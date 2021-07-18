#include "transformer_encoder_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
TransformerEncoderLayer<T>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _training(true),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _qkv_linear(
          typename FeedForward<T>::Config(3 * hidden_size, hidden_size)),
      _attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
               _max_batch_tokens),
      _ffn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
              _max_batch_tokens),
      _ff1(typename FeedForward<T>::Config(_intermediate_size, hidden_size)),
      _ff2(typename FeedForward<T>::Config(hidden_size, _intermediate_size)),
      _softmax(typename Softmax<T>::Config(num_heads)),
      _attn_prob_dropout(typename Dropout<T>::Config(attn_prob_dropout_ratio),
                         _max_batch_tokens * _heads * _max_seq_len),
      _attn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                    _max_batch_tokens * _hidden_size),
      _ffn_activation_dropout(
          typename Dropout<T>::Config(activation_dropout_ratio),
          _max_batch_tokens * _intermediate_size),
      _ffn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                   _max_batch_tokens * _hidden_size),
      _attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)) {
  assert(_hidden_size % _heads == 0);
  allocate_mem_buffer();
}

template <typename T>
TransformerEncoderLayer<T>::~TransformerEncoderLayer() {
  free_mem_buffer();
}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_fw(const T *input_ptr,
                                               const T *input_mask_ptr,
                                               T *output_ptr, T *buffer) {//buffer=attn_buffer=_shared_mem_ptr
  T *q_tf_ptr = _qkv_ptr; //_qkv_ptr=max_batch_tokens*hidden_size*3
  T *k_tf_ptr = q_tf_ptr + _batch_dim; //_batch_dim=batch_tokens*hidden_size
  T *v_tf_ptr = k_tf_ptr + _batch_dim;

  if (_pre_or_postLayerNorm) {
    _attn_ln.Forward(_gemmQKV_inp_ptr, input_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.Forward(_batch_tokens, gemmQKV_inp_ptr, _attn_qkvw_ptr, buffer,//btbt _batch_tokens=batch_size*seq_len, _attn_qkvw_ptr在assign_weight_ptr()中指向_hidden_size * _hidden_size * 3部分,也就是负责把token dim变成QKV dim的线性映射的权重
                      _cublasHandle);//btbt [maybug] ???FeedForward调用cublasGemmEx好像浪费了一次标量乘和矩阵加?
  //btbt 上行A=_attn_qkvw_ptr[hid_sz,hid_sz*3]^T,postLN:B=gemmQKV_inp_ptr=input_ptr[hid_sz,_batch_tokens],out=C=buffer[3*hid_sz,_batch_tokens]???貌似C有可能没占满buffer,此时buffer多出部分有可能是有上次计算的结果的,会不会对当前计算产生影响?
  launch_bias_add_transform_20314<T>(q_tf_ptr, buffer, _attn_qkvb_ptr,//_attn_qkvb_ptr在assign_weight_ptr()中占大整块权重的_hidden_size * 3部分,也就是负责把token dim变成QKV dim的线性映射的Bias
                                     _batch_size, _seq_len, 3, _heads,
                                     _hidden_size / _heads, _stream);
  //dim0=_batch_size,dim1=_seq_len,dim2=3,dim3=heads,dim4=headSz=hidden_size/heads,input=buffer也就是上次FF矩阵乘后的结果,bias=_attn_qkvb_ptr,out=q_tf_ptr[]
  //grid[batch_sz,seq_len,3]按每个tok的每个QKV角色划分,blk[headSz*heads=hid_sz or mx_thread]就是每个thread把buffer中特定(batch中的)sample特定token的特定QKV角色的dim(即hid_sz)加上相应的bias(来自_attn_qkvb_ptr),由于cuda内用float4所以相加是4个4个来的,
  //相加结果放在q_tf_ptr中某个QKV角色的(batch中)某sample的某头的某token的某位headSz上,也就是output的q_tf_ptr[KQV数*batchSz*头数*seqLen*headSz],其中batchSz*seqLen=batch_tokens,头数*headSz=hidden_size,KQVs数=3,所以输出的大小是在q_tf_ptr指向的内存区域内
  // attention scores, q*k
  _attn_scores.Forward(_batch_heads, _soft_out_ptr, k_tf_ptr, q_tf_ptr,//_batch_heads=batchSz*heads, _soft_out_ptr=max_batch_tokens*heads*max_seq_len
                       _cublasHandle);//就是调用cublasGemmStridedBatchedEx把batch_heads个K矩阵与同样数量的Q矩阵相乘,对于每个batch中的sample的每个head的每个token会与该sample的其它token得到一个atten分数
  //batch=_batch_heads,m=seqLen,n=seqLen,k=headSz,stridA=m*k=seqLen*headSz,stridB=n*k=seqLen*headSz,stridC=m*n=seqLen*seqLen,alph=1/sqrt(headSz)s,A=[(batchSz*头数=batch_heads)*headSz*seqLen]的第i个strid的转置是[seqLen*headSz],B=q_tf_ptr(形状同k_tf_ptr)
  //out=C=_soft_out_ptr[(batchSz*头数=batch_heads)*seqLen*seqLen] stride窗口是在倒数的维度上滑动的,前面的维度(batch heads)是窗口的数目
  // Softmax + Mask //btbt ?*? Q*K后没有除以sqrt(d_k)?见创建_attn_scores时的'(T(1.0) / T(sqrt(_hidden_size / _heads)))',sqrt(d_k)已经作为alpha传给cublasGemmStridedBatchedEx去计算除法
  _softmax.Forward(_soft_out_ptr, input_mask_ptr, _batch_size, _seq_len,
                   _seq_len, _stream);
  //val=inp=out=_soft_out_ptr(即是输入又保存了计算结果作为输出),attn_mask=input_mask_ptr,from_len=to_len=该次forward的_seq_len(假设512),nhead=config.heads假设12,mask_future=true
  //out=_soft_out_ptr[(batchSz*头数=batch_heads)*seqLen*seqLen] 保存了Q对应每个K的softmax后的attn score
  // attn prob dropout.
  _attn_prob_dropout.dropout(_ctx_bufB_ptr, _soft_out_ptr,//_ctx_bufB_ptr=[max_batch_tokens*heads*max_seq_len]
                             _batch_heads * _seq_len * _seq_len, _stream);
  //out=_ctx_bufB_ptr 输出dropout后的结果[(batchSz*头数=batch_heads)*seqLen*seqLen],in=vals=_soft_out_ptr,count=batch_heads*seq_len*seq_len含义类似Dropout.max_ele_num,
  //btbt [REFACTOR]???上面的dropout能否在计算softmax时就做了,_ctx_bufB_ptr 的shape和soft_out_ptr一样的,用soft_out_ptr保存drop后的数据就可以,既省时间又省内存
  // attention context, score * v
  _attn_context.Forward(_batch_heads, buffer, v_tf_ptr, _ctx_bufB_ptr,//C=out=buffer但由于_attn_context的beta=0所以buffer原有的数据没排上用场而单纯作为out,A=v_tf_ptr[batchSz*头数*seqLen*headSz],B=_ctx_bufB_ptr[(batchSz*头数=batch_heads)*seqLen*seqLen]
                        _cublasHandle);//m=headSz=hidden_size/heads,n=seq_len,k=seq_len,stridA=m*k=headSz*seqLen,stridB=n*k=headSz*seqLen,stridC=m*n=headSz*seqLen
  //上行结束后out=buffer[(batchSz*头数=batch_heads)*seqLen*headSz]=softed_atten * v
  // [b, nh, s, ad] -> [b, s, nh, ad] //[(batchSz*头数=batch_heads)*seqLen*headSz]转成[batchSz*seqLen*头数*headSz]也就是[atchSz*seqLen*(hid_sz=头数*headSz)]
  launch_transform4d_0213<T>(_attn_o_inp_ptr, buffer, _batch_size, _seq_len,
                             _hidden_size, _heads, 1, _stream);

  _attn_out_linear.Forward(_batch_tokens, _attn_o_inp_ptr, _attn_ow_ptr,
                           output_ptr, _cublasHandle);

  _attn_dropout.bias_dropout_residual(output_ptr, output_ptr, input_ptr,
                                      _attn_ob_ptr, _batch_tokens, _hidden_size,
                                      _stream);//out=output_ptr=postLN:_ff1_inp_ptr[batch_tokens*hidden_size]
  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _attn_ln.Forward(output_ptr, output_ptr, _attn_nw_ptr, _attn_nb_ptr,//max_rows=_max_batch_tokens,ln_res=output_ptr,vars=malloc(max_rows),use_mean=false,inp=output_ptr,gamma=scale=_attn_nw_ptr,betta=bias=_attn_nb_ptr
                     _batch_tokens, _stream);//batch_size=_batch_tokens,hidden_dim=hidden_size
  }
}

template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_fw(T *inp_ptr, T *out_ptr) {
  // save _ff1_inp_ptr, _relu_inp_ptr, _ff2_inp_ptr for backward
  if (_pre_or_postLayerNorm) {
    _ffn_ln.Forward(_ff1_inp_ptr, inp_ptr, _ffn_nw_ptr, _ffn_nb_ptr,
                    _batch_tokens, _stream);
  }
  _ff1.Forward(_batch_tokens, _ff1_inp_ptr, _inter_w_ptr, _relu_inp_ptr,//batch_size=n=_batch_tokens,input=B=_ff1_inp_ptr=inp_ptr if postLN,weight=A=_inter_w_ptr[hidden_size*intermediate_size]
               _cublasHandle);//out=C=_relu_inp_ptr[batch_tokens*intermediate_size]

  _ffn_activation_dropout.bias_relu_dropout(_ff2_inp_ptr, _relu_inp_ptr,//max_ele_num=max_batch_tokens*intermediate_size,in=vals=_relu_inp_ptr,
                                            _inter_b_ptr, _batch_tokens,//bias=_inter_b_ptr[_intermediate_size],rows=total_count=_batch_tokens
                                            _intermediate_size, _stream);//cols=dim=_intermediate_size,mask=malloc(max_ele_num)[max_batch_tokens*intermediate_size]
  //上行out=_ff2_inp_ptr[batch_tokens*intermediate_size] //btbt [REFACTOR]??? 貌似很多内存块没有复用,是为了backward计算更快么?
  _ff2.Forward(_batch_tokens, _ff2_inp_ptr, _output_w_ptr, out_ptr,
               _cublasHandle);
  //上行out_ptr[batch_tokens*hidden_size]
  _ffn_dropout.bias_dropout_residual(out_ptr, out_ptr, inp_ptr, _output_b_ptr,
                                     _batch_tokens, _hidden_size, _stream);

  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _ffn_ln.Forward(out_ptr, out_ptr, _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens,
                    _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::Forward(const T *input_ptr,
                                         const T *input_mask_ptr, T *out_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *attn_buffer = _shared_mem_ptr;  // 3 * _batch_dim //btbt ??? _shared_mem_ptr 在allocate_mem_buffer()中为何要如此设置?仅给atten做内存么?怎么感觉也给ffn做内存了?
  // _batch_dim
  T *ffn_inp_ptr = //btbt {post_LN(_pre_or_postLayerNorm=False):ff1_inp_ptr=max_batch_tokens*hidden_size}
      _pre_or_postLayerNorm ? _shared_mem_ptr + 3 * _batch_dim : _ff1_inp_ptr; //btbt ??? _pre_or_postLayerNorm为true时为pre layerNorm,但为何要酱紫?pre LN的话之前malloc的_ff1_inp_ptr内存就用不上了?

  attn_layer_fw(input_ptr, input_mask_ptr, ffn_inp_ptr, attn_buffer);

  ffn_layer_fw(ffn_inp_ptr, out_ptr);
}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_bw(const T *input_ptr,
                                               const T *input_mask_ptr,
                                               const T *grad_output_ptr,
                                               T *grad_input_ptr, T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};
  const T *q_tf_ptr = _qkv_ptr;
  const T *k_tf_ptr = q_tf_ptr + _batch_dim;
  const T *v_tf_ptr = k_tf_ptr + _batch_dim;
  // batch_dim = batch_size * seq_len * hidden_size
  // buffer size: batch_dim * 3 + max(batch_dim * 3,
  //     batch_size * head_num * seq_len * seq_len)
  T *grad_residual_ptr = buffer;
  buffer += _batch_dim;

  T *grad_input_buf_ptr = buffer;  // batch_dim
  T *grad_qkv_5d_ptr = buffer;     // batch_dim * 3
  buffer += 3 * _batch_dim;

  T *grad_qkv_4d_ptr = buffer;   // batch_dim * 3
  T *grad_softmax_ptr = buffer;  // batch_size * head_num * seq_len * seq_len
  // buffer += max(3 * _batch_dim,
  //   batch_size * head_num * seq_len * seq_len);

  if (_pre_or_postLayerNorm) {
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_output_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  } else {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_residual_ptr,
                      grad_output_ptr, nullptr, _ff1_inp_ptr, _attn_nw_ptr,
                      _attn_nb_ptr, _batch_tokens, streams);//OUTPUT=>LN在fw时的输入的导数:grad_residual_ptr[batch_size * seq_len, hidden_size],LN的w的导数:_grad_attn_nw_ptr[hidden_size],LN的bias的导数:_grad_attn_nb_ptr[hidden_size]
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_residual_ptr, _batch_tokens,
                                          _hidden_size, _stream);//OUTPUT=>Drop在fw时顺便加了,这是对该bias的导数:_grad_attn_ob_ptr[hidSz],在fw时的输入的导数:grad_input_ptr[batch_size, seq_len, hidden_size]
  }

  // bw of output project
  _attn_out_linear.Backward(_batch_tokens, grad_input_ptr, _attn_o_inp_ptr,
                            _attn_ow_ptr, _grad_attn_ow_ptr, _grad_attn_ob_ptr,
                            _cublasHandle, _stream, grad_input_buf_ptr, nullptr,
                            false);//OUTPUT=>_grad_attn_ow_ptr[hidden_size*intermediate_size]权重导数,grad_input_buf_ptr 该层在fw时的输入的导数
  launch_transform_0213<T>(grad_input_ptr, grad_input_buf_ptr, _batch_size,
                           _seq_len, _hidden_size, _heads, _stream);// [b, s, h] -> [b, nh, s, ad]//OUTPUT=>grad_input_ptr[batch_size*heads*seq_len*headSz]

  // bw of score * v
  _attn_context.Backward(_batch_heads, grad_input_ptr, v_tf_ptr, _ctx_bufB_ptr,//m=headSz=hidden_size/heads,n=seq_len,k=seq_len,opA=opB=N,alpha=1,beta=0,bsz=batch=_batch_heads[batch_size*heads]
                         _cublasHandle, grad_qkv_5d_ptr + 2 * _batch_dim,//d_output=A1=grad_input_ptr[batch_size*heads*seq_len*headSz]上行linear的fw的输入的导数转成[b,nh,s,ad]格式,_buffer_b=B1=_ctx_bufB_ptr[batchSz*heads*seq_len*seq_len],inpGradA=C1=grad_qkv_5d_ptr[2*_batch_dim:]=[_batch_dim]=[batch_size*heads*seq_len*headSz]???,stridA1=headSz*seq_len,stridB1=seq_len*seq_len,stridC1=headSz*seq_len
                         grad_softmax_ptr);//_buffer_a=A2=v_tf_ptr,B2=grad_input_ptr,inpGradB=C2=grad_softmax_ptr[batch_size*head_num*seq_len*seq_len],stridA2=headSz*seq_len,stridB2=headSz*seq_len,stridC2=seq_len*seq_len,
  //OUTPUT=>对V:v_tf_ptr的导数 保存在grad_qkv_5d_ptr[2*_batch_dim:]中[batch_size*heads*seq_len*headSz],对softmax并dropout后的attn score从fw保存在_ctx_bufB_ptr中,对它的导数在grad_softmax_ptr中
  _attn_prob_dropout.d_dropout(grad_softmax_ptr,
                               _batch_heads * _seq_len * _seq_len, _stream);
  //BTBT *** softmax的backward和dropout&LN的backward使用tiled_partition的方式不同,从一开始就把gridDim和blockDim的处理变成列优先的,所以后面不用做别扭的从行优先变成列优先的转换,省了不少时间
  _softmax.Backward(grad_softmax_ptr, _soft_out_ptr, _batch_size, _seq_len,//out_grad=grad_softmax_ptr,soft_out=soft_inp=_soft_out_ptr保存了softmax后的attn score,rows=batch_size*heads*seq_len,to_len=softmax_len=seq_len
                    _seq_len, _stream);//OUTPUT=>grad_softmax_ptr保存了对softmax后的attn score的导数

  // bw of q * k
  _attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_tf_ptr, q_tf_ptr,
                        _cublasHandle, grad_qkv_5d_ptr + _batch_dim,
                        grad_qkv_5d_ptr);

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_qkv_4d_ptr, grad_qkv_5d_ptr, _batch_size,
                             _seq_len, _hidden_size, _heads, 3, _stream);

  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr, gemmQKV_inp_ptr,
                       _attn_qkvw_ptr, _grad_attn_qkvw_ptr, _grad_attn_qkvb_ptr,
                       _cublasHandle, _stream, grad_input_buf_ptr);

  if (_pre_or_postLayerNorm) {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr,
                      grad_input_buf_ptr, grad_output_ptr, gemmQKV_inp_ptr,
                      _attn_nw_ptr, _attn_nb_ptr, _batch_tokens, streams);
  } else {
    // FIXME later
    launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr, grad_residual_ptr,
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_bw(const T *grad_output_ptr,
                                              const T *output_ptr,
                                              T *grad_inp_ptr, T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};//BTBT ??? 这啥?一个stream fork成两个sub stream?

  T *grad_residual_ptr = buffer;//grad_residual_ptr:_batch_dim=[batch_size * seq_len, hidden_size]
  buffer += _batch_dim;

  T *grad_ff1_inp_ptr = buffer;//grad_ff1_inp_ptr:_batch_dim=[batch_size * seq_len, hidden_size]
  buffer += _batch_dim;

  T *grad_ff1_out_ptr = buffer;//btbt ???shape
  // buffer += _batch_size * _seq_len * _intermediate_size;

  if (_pre_or_postLayerNorm) {
    _ffn_dropout.d_bias_dropout_residual(grad_inp_ptr, _grad_output_b_ptr,
                                         grad_output_ptr, _batch_tokens,
                                         _hidden_size, _stream);
  } else {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_residual_ptr,//gamma_grad=_grad_ffn_nw_ptr[hidden_size],betta_grad=_grad_ffn_nb_ptr[hidden_size],inp_grad=grad_residual_ptr[batch_size * seq_len, hidden_size]
                     grad_output_ptr, nullptr, output_ptr, _ffn_nw_ptr,//out_grad=grad_output_ptr 上一级的grad_out[batch_size * seq_len, hidden_size],residual_grad=null,inp_or_out=output_ptr,gamma=_ffn_nw_ptr,betta=_ffn_nb_ptr
                     _ffn_nb_ptr, _batch_tokens, streams);//batch_size=_batch_tokens,OUTPUT=>LN在fw时的输入的导数:grad_residual_ptr[batch_size * seq_len, hidden_size],LN的w的导数:_grad_ffn_nw_ptr[hidden_size],LN的bias的导数:_grad_ffn_nb_ptr[hidden_size]
    _ffn_dropout.d_bias_dropout_residual(grad_inp_ptr, _grad_output_b_ptr,//d_input=in_grad=grad_inp_ptr[batch_size, seq_len, hidden_size],d_bias=bias_grad=_grad_output_b_ptr[hid_sz]因为这个drop在fw时顺便加了bias再drop
                                         grad_residual_ptr, _batch_tokens,//d_output,rows=out_grad=grad_residual_ptr[batch_size * seq_len, hidden_size]上行LN对fw输入的导数,rows=row_sz=_batch_tokens,cols=dim=hidSz
                                         _hidden_size, _stream);//mask用的是fw时产生的mask,max_ele_num=max_batch_tokens*hidden_size,OUTPUT=>Drop在fw时顺便加了,这是对该bias的导数:_grad_output_b_ptr[hidSz],在fw时的输入的导数:grad_inp_ptr[batch_size, seq_len, hidden_size]
  }

  _ff2.Backward(_batch_tokens, grad_inp_ptr, _ff2_inp_ptr, _output_w_ptr,//out_grad=B2=grad_inp_ptr[batch_tokens,hidSz]倒数上一级在fw时输入的导数,input_ptr=A=_ff2_inp_ptr[batch_tokens*intermediate_size]保存了fw时ff2的输入,weights=A2=_output_w_ptr fw时的权重
                _grad_output_w_ptr, _grad_output_b_ptr, _cublasHandle, _stream,//weights_grad=C=_grad_output_w_ptr[hidden_size*intermediate_size]权重导数,_grad_output_b_ptr[hidSz]bias导数
                grad_ff1_out_ptr, nullptr, false);//inp_grad_out=C2=grad_ff1_out_ptr[intermediate_size*batch_tokens], OUTPUT=>_grad_output_w_ptr[hidden_size*intermediate_size]权重导数,grad_ff1_out_ptr 该ff2在fw时的输入的导数

  _ffn_activation_dropout.d_bias_relu_dropout(//max_ele_num=batch_tokens*intermediate_size,d_inp_out=in_grad&out_grad=grad_ff1_out_ptr,d_bias_out=bias_grad=_grad_inter_b_ptr[intermediate_size],
      grad_ff1_out_ptr, _grad_inter_b_ptr, _relu_inp_ptr, _inter_b_ptr,//input=_relu_inp_ptr[batch_tokens*intermediate_size]保存了fw时的输入,bias=inter_b_ptr[_intermediate_size]偏移的参数值,rows=row_sz=_batch_tokens,cols=dim=_intermediate_size
      _batch_tokens, _intermediate_size, _stream);//OUTPUT=>grad_ff1_out_ptr保存了该drop层在fw时的输入的导数,_grad_inter_b_ptr该drop层bias的导数

  _ff1.Backward(_batch_tokens, grad_ff1_out_ptr, _ff1_inp_ptr, _inter_w_ptr,//input_ptr=A=_ff1_inp_ptr[batch_tokens*hidden_size]保存了fw时ff1的输入
                _grad_inter_w_ptr, _grad_inter_b_ptr, _cublasHandle, _stream,
                grad_ff1_inp_ptr, nullptr, false);//OUTPUT=>_grad_inter_w_ptr[intermediate_size*hidden_size]ff1权重的导数,grad_ff1_inp_ptr[batch_tokens*hidden_size]ff1在fw的输入的导数

  /* ln signature:
  grad_gamma_grad, grad_betta, grad_inp,
  grad_out, grad_residual, output, gamma, betta,
  */
  const T *add_res_ptr = _ff1_inp_ptr;
  if (_pre_or_postLayerNorm) {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_inp_ptr,
                     grad_ff1_inp_ptr, grad_output_ptr, _ff1_inp_ptr,
                     _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens, streams);
  } else {
    launch_fused_add2<T>(grad_inp_ptr, grad_ff1_inp_ptr, grad_residual_ptr,//OUTPUT=>对整个ffn_layer的输入进行求导后放在grad_inp_ptr[batch_tokens,hidSz]
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::Backward(const T *grad_output_ptr,[batch_size * seq_len, hidden_size]
                                          const T *input_ptr,
                                          const T *output_ptr,
                                          const T *input_mask_ptr,
                                          T *grad_input_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *grad_ffn_inp_ptr = _shared_mem_ptr;//btbt ??? _shared_mem_ptr的大小如何确定? 
  T *buffer = grad_ffn_inp_ptr + _batch_dim; //batch_dim=batch_tokens*hidden_size btbt ??? 为何要加batch_dim,说明grad_ffn_inp_ptr是[_batch_dim]即[batch_size, seq_len, hidden_size]

  /*
  buffer size needed by ffn bw:
      2 * _batch_dim + _batch_size * _seq_len * _intermediate_size
  */
  ffn_layer_bw(grad_output_ptr, output_ptr, grad_ffn_inp_ptr, buffer);

  /*
  buffer size needed by attn bw:
      4 * _batch_dim + max(3 * _batch_dim,
      _batch_size * _head_num * _seq_len * _seq_len);
  */
  attn_layer_bw(input_ptr, input_mask_ptr, grad_ffn_inp_ptr, grad_input_ptr,//grad_ffn_inp_ptr[batch_tokens,hidSz]上行ffn_layer在fw的输入的导数
                buffer);//OUTPUT=>grad_input_ptr???
}

template <typename T>
void TransformerEncoderLayer<T>::SetTrainingMode(bool training) {
  // Dropout will be skipped when not in training model.
  _attn_prob_dropout.SetTrainingMode(training);
  _attn_dropout.SetTrainingMode(training);
  _ffn_activation_dropout.SetTrainingMode(training);
  _ffn_dropout.SetTrainingMode(training);
}

template <typename T>
T *TransformerEncoderLayer<T>::_shared_mem_ptr = nullptr;

template class TransformerEncoderLayer<float>;
template class TransformerEncoderLayer<__half>;
