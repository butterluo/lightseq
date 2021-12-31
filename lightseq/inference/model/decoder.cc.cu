#include <unistd.h>

#include "3rdparty/cub/cub/cub.cuh"
#include "decoder.h"
#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels.h"

/**
@file
Transformer decoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
Decoder<OpType_>::Decoder(int max_batch_size, const int* p_d_padding_mask,
                          const _DataType* p_d_encoder_output, int* p_d_result,
                          TransformerWeight<OpType_>& tw, cudaStream_t stream,
                          cublasHandle_t hd, bool output_topk,
                          const int* p_d_lang_id)
    : _max_batch_size(max_batch_size),
      _max_thread_per_block(1024),
      _h_can_num_batch(0),
      _cub_sort_buffer_bytes(max_batch_size * tw._beam_size *
                             tw._trg_vocab_size * sizeof(_DataType)),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_encoder_output(p_d_encoder_output),
      _p_d_result(p_d_result),
      _p_d_trg_emb_wei(tw.get_trg_emb_wei()),
      _p_d_dec_wei(tw.get_dec_wei()),
      _tw(tw),
      _stream(stream),
      _hd(hd),
      _output_topk(output_topk),
      _p_d_lang_id(p_d_lang_id),  // source token id
      _layer_size_encdec_k(max_batch_size * tw._max_step * tw._hidden_size),
      _layer_size_self_k(max_batch_size * tw._max_step * tw._hidden_size *
                         tw._beam_size),
      _type_one(1.f),
      _type_zero(0.f),
      _fzero(0.f),
      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _logit_scaler(_tw._no_scale_embedding ? 1.f
                                            : sqrt(1.f / tw._hidden_size)),
      _h_alive_seq_probs(max_batch_size * tw._beam_size,
                         min_log_probability / 2),//BTBT ??? 为何这里把值初始化为(min_log_probability / 2),下三行又把每个beam sz开始值初始化为0
      _h_length_norm(tw._max_step, 1.f),
      _h_unfinished(1) {
  for (int i = 0; i < _h_alive_seq_probs.size(); i += tw._beam_size) {
    _h_alive_seq_probs[i] = 0.f;
  }
  if (tw._length_penalty >= 0) {
    for (int i = 0; i < _h_length_norm.size(); i++) {
      _h_length_norm[i] = length_norm(i + 1, tw._length_penalty);
    }
  }
  return;
}

/**
Compute GPU memory size needed by transformer decoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long Decoder<OpType_>::compute_buffer_bytesize() {
  long cache_bytesize = 4 * _tw._n_dec_layer * _layer_size_self_k +
                        2 * _tw._n_dec_layer * _layer_size_encdec_k +
                        _max_batch_size * _tw._beam_size * _tw._hidden_size;
  cache_bytesize *= sizeof(_DataType);

  long decode_buffer_bytesize =
      _max_batch_size * _tw._beam_size * _tw._hidden_size * 4 +
      _max_batch_size * _tw._beam_size *
          max(_tw._hidden_size, _tw._inner_size) +
      _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step;
  decode_buffer_bytesize *= sizeof(_DataType);

  long sf = _max_batch_size * _tw._beam_size * _tw._trg_vocab_size * 2 +
            _max_batch_size * _tw._beam_size * 2;
  long si = _max_batch_size * _tw._beam_size * _tw._max_step * 2 +
            _max_batch_size * _tw._beam_size * _tw._trg_vocab_size +
            _max_batch_size * _tw._beam_size + 1;
  long beam_buffer_bytesize = sf * sizeof(float) + si * sizeof(int);

  return cache_bytesize + max(decode_buffer_bytesize, beam_buffer_bytesize);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by decoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void Decoder<OpType_>::init_buffer(void* pbuf) {
  std::cout << "decoder buffer init start" << std::endl;
  _DataType* curp = reinterpret_cast<_DataType*>(pbuf);

  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "key" of enc_dec attention
    _p_d_encdec_k_bgeem.push_back(curp);/* 分层保持cros attn要到的由ecd out过来的k值,[batch_size, head_num, dim_per_head, batch_seq_len] */
    curp += _layer_size_encdec_k;
  }
  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "value" of enc_dec attention
    _p_d_encdec_v_bgeem.push_back(curp);/* 分层保持cros attn要到的由ecd out过来的v值,[batch_size, head_num, dim_per_head, batch_seq_len] */
    curp += _layer_size_encdec_k;
  }
  // reused buffer with _p_d_self_k_bgeem _p_d_self_v_bgeem,
  // no need to add curp because _p_d_encoder_out_buf is smaller
  // and no need to use it any more after get _p_d_encdec_k_bgeem
  // and _p_d_encdec_v_bgeem
  _p_d_encoder_out_buf = curp;

  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    // the "key" of decoder self attention, we need to maintain it by twice
    // one for current step's "key", one for "key" of beam_search cache
    // after finishing current step's search, we will copy the first one
    // to the second one to refresh beam_search cache
    // based on the selected beam id
    _p_d_self_k_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    // the "value" of decoder self attention, we need to maintain it by twice
    // one for current step's "value", one for "value" of beam_search cache
    // after finishing current step's search, we will copy the first one
    // to the second one to refresh beam_search cache
    // based on the selected beam id
    _p_d_self_v_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  _p_d_self_k_bgeem1 = _p_d_self_k_bgeem.data();
  _p_d_self_k_bgeem2 = _p_d_self_k_bgeem.data() + _tw._n_dec_layer;
  _p_d_self_v_bgeem1 = _p_d_self_v_bgeem.data();
  _p_d_self_v_bgeem2 = _p_d_self_v_bgeem.data() + _tw._n_dec_layer;

  // GPU memory buffer to save "query",
  // In all layers, using the same buffer
  _p_d_cur_step_query = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;

  // we can use the same buffer for decoder network computation
  // and beam search, since they're serial.
  _DataType* reuse_p = curp;

  // for decode network computation
  _p_d_self_step_qkv = curp;  // [q, k, v], result of gemm [max_batch_size*beam_size*hidden_size*3]
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size * 3;
  _p_d_query_buf1 = curp;  // "query" buffer[max_batch_size*beam_size*hidden_size]
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;
  _p_d_query_buf2 = curp;  // "query" buffer[max_batch_size*beam_size*max(hidden_size,inner_size)]
  curp +=
      _max_batch_size * _tw._beam_size * max(_tw._hidden_size, _tw._inner_size);
  _p_d_c = curp;  // correlation(attention score) buffer
  curp += _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step;

  // for beam search
  curp = reuse_p;
  _p_d_logit_buf = curp;  // vocab ligit
  curp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  // always be float
  float* fcurp = (float*)curp;
  // seq score ended with every target token for current step
  _p_d_can_score = fcurp;
  fcurp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  _p_d_alive_seq_probs = fcurp;  // alive seq probability
  fcurp += _max_batch_size * _tw._beam_size;
  _p_d_alive_seq_score = fcurp;  // alive seq score
  fcurp += _max_batch_size * _tw._beam_size;

  int* pint = reinterpret_cast<int*>(fcurp);
  // FIXME
  std::vector<int> start_id_vec(
      _max_batch_size * _tw._beam_size * _tw._max_step * 2, _tw._start_id);//BTBT 用于初始化 _p_d_alive_seq 和 _p_d_alive_seq_buf,初始值为ModelConf.trg_start_id,即target seq的开始标记
  usleep(3000);
  CHECK_GPU_ERROR(cudaMemcpyAsync(pint, start_id_vec.data(),
                                  sizeof(int) * start_id_vec.size(),
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  // token id of alive seq, we need to maintain it by twice
  // one for current step, one for beam_search cache
  // after finishing current step's search, we will copy the first one
  // to the second one to refresh beam_search cache
  // based on the selected beam id
  _p_d_alive_seq = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;
  _p_d_alive_seq_buf = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;

  // candidate token id for every beam, selected by rough top-beam_size op
  _p_d_can_idx = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  // candidate token number for every beam, selected by rough top-beam_size op
  _p_d_can_num = pint;
  pint += _max_batch_size * _tw._beam_size + 1;

  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_sample_unfinished, sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_curandstate,
                             _max_batch_size * sizeof(curandState)));
  ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);

  CHECK_GPU_ERROR(cudaGetLastError());
  std::cout << "decoder buffer init succeed" << std::endl;
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string Decoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_tw._multilg_type == 0 && _p_d_trg_emb_wei.size() != 7) {
    return "violate p_d_trg_emb_wei.size() = 7";
  }
  if (_tw._multilg_type != 0 && _p_d_trg_emb_wei.size() != 8) {
    return "violate p_d_trg_emb_wei.size() = 8";
  }
  if (_p_d_dec_wei.size() != _tw._weight_per_dec_layer * _tw._n_dec_layer) {
    return "violate p_d_dec_wei.size() = weight_per_dec_layer * n_dec_layer";
  }
  if (_output_topk && _tw._length_penalty < 0) {
    return "not support length_penlty < 0 for generate topk currently !";
  }
  bool btmp = false;
  for (int i = 1; i < 64; i *= 2) {
    if (i == _tw._beam_size) {
      btmp = true;
      break;
    }
  }
  if (!btmp) {
    return "wrong beam_size, should be 1, 2, 4, 8, 16 or 32";
  }
  if (_tw._multilg_type != 0 && _p_d_lang_id == nullptr) {
    return "lang id should not be null when multilg";
  }
  return "";
}

/**
Decoder inference
*/
template <OperationType OpType_>
void Decoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
  /* ---step1. init--- */
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;
  _step_token_num = batch_size * _tw._beam_size;
  _batch_max_decode_length =
      min(_tw._max_step, batch_seq_len + _tw._extra_decode_length) - 1; //BTBT batch_seq_len??? max_step??? _extra_decode_length??? 为何减1??? _batch_max_decode_length???
  _is_sampling =
      (_tw._sampling_method == "topk" || _tw._sampling_method == "topp" ||
       _tw._sampling_method == "topk_greedy");
  if (_is_sampling) {
    _batch_max_decode_length = _tw._max_step;
  }

  project_encoder_output();  // project encoder output
  // init the first step's token id with target start_id
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_alive_seq_probs,
                                  _h_alive_seq_probs.data(),  //BTBT ??? _h_alive_seq_probs貌似只用于这里对_p_d_alive_seq_probs的值初始化
                                  sizeof(float) * _batch_size * _tw._beam_size,
                                  cudaMemcpyHostToDevice, _stream));//BTBT REFACTOR 一些常量(比如_h_alive_seq_probs)能否常驻显存,数值内存分配和初始化能否直接就在显存中做了

  /* ---step2. autoregressive decoding--- */
  for (_cur_step = 0; _cur_step < _batch_max_decode_length - 1; _cur_step++) {
#ifdef DEBUG_RESULT
    std::cout << "*** run step " << _cur_step << " ***" << std::endl;
#endif
    if (run_step()) {  // one step
      break;
    }
  }

  /* ---step3. output the decoding result--- */
  if (_output_topk || _is_sampling) {//BTBT 看了此分支_output_topk=true,_is_sampling=='beam_search'
    if (_cur_step == _batch_max_decode_length) {
      _cur_step -= 1;
    }
    ker_write_topk_result<<<_batch_size * _tw._beam_size, _cur_step + 1, 0,
                            _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._trg_vocab_size,//??? p_d_alive_seq_score的用法,是要返回么
        _tw._max_step, _tw._beam_size, _tw._end_id);
    return;
  }
  if (_tw._length_penalty >= 0.f || _cur_step == _batch_max_decode_length) {
    ker_write_trg_tokenid_pos_penalty<<<_batch_size, _cur_step + 1, 0,
                                        _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._max_step,
        _tw._beam_size);
  } else {
    ker_write_trg_tokenid_neg_penalty<<<_batch_size, _cur_step + 1, 0,
                                        _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._max_step,
        _tw._beam_size, _tw._trg_vocab_size, _tw._end_id);
  }
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {
    print_vec(_p_d_result + i * (_cur_step + 1), "finial res", _cur_step + 1);
  }
#endif
  return;
}

/**
Project encoder output 
*/
template <OperationType OpType_>
void Decoder<OpType_>::project_encoder_output() {
  int kv_dim = _tw._hidden_size * 2 * _tw._n_dec_layer; //BTBT 把所有layer的cross attn计算所需的k,v用cublasGemmEx先算出来,保存在_p_d_encoder_out_buf
#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  print_vec(_p_d_encoder_output, "_p_d_encoder_output(head):", 5);
  print_vec(_p_d_encoder_output + _batch_token_num * _tw._hidden_size - 5,
            "_p_d_encoder_output(tail)", 5);
  print_vec(_p_d_trg_emb_wei[4], "encoder project(head):", 10);
#endif
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, _batch_token_num, _tw._hidden_size,
      &_type_one, _p_d_trg_emb_wei[4], _AType, kv_dim, _p_d_encoder_output, //BTBT _p_d_trg_emb_wei[4]是enc_out_kernel_kv
      _BType, _tw._hidden_size, &_type_zero, _p_d_encoder_out_buf, _CType,
      kv_dim, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // _p_d_encoder_out_buf: [batch_size, batch_seq_len, layer_num, 2,
  // hidden_size]

#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  print_vec(_p_d_encoder_out_buf, "encoder out(head):", 5);
  print_vec(_p_d_encoder_out_buf +
                _batch_token_num * _tw._hidden_size * _tw._n_dec_layer - 5,
            "encoder out(tail):", 5);
#endif
  ker_arrange_encdec_kv_launcher<_DataType>(//BTBT _p_d_trg_emb_wei[5]是enc_out_bias_kv,这里是把bias加到上面矩阵相乘得到的cros attn所需的kv值上
      _batch_token_num, _tw._n_dec_layer, _tw._hidden_size, _stream,
      _p_d_encoder_out_buf, _p_d_trg_emb_wei[5], _p_d_encdec_k_bgeem[0],// _p_d_encdec_k_bgeem, _p_d_encdec_v_bgeem ctrl有注释
      _p_d_encdec_v_bgeem[0], _layer_size_encdec_k, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);
  return;
}

/**
Decode one step
*/
template <OperationType OpType_>
bool Decoder<OpType_>::run_step() {
  embedding();
  decoder_stack();
  /* --- Project hidden states to vocab logits--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._trg_vocab_size, _step_token_num,
      _tw._hidden_size, &_logit_scaler, _p_d_trg_emb_wei[0], _AType,
      _tw._trg_vocab_size, _p_d_cur_step_query, _BType, _tw._hidden_size,
      //&_type_zero, _p_d_logit_buf, _CType, _tw._trg_vocab_size, _computeType,
      &_fzero, _p_d_logit_buf, _CType, _tw._trg_vocab_size, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _tw._beam_size; j++) {  // beam_id
      std::cout << "decoder output: batch-" << i << ", beam-" << j << std::endl;
      print_vec(_p_d_cur_step_query + i * _tw._beam_size * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden", 10);
      print_vec(_p_d_logit_buf + i * _tw._beam_size * _tw._trg_vocab_size +
                    j * _tw._trg_vocab_size,
                "logits", 10);
    }
  }
#endif
  if (_tw._sampling_method == "topk") {
    return sample();
  } else if (_tw._sampling_method == "topp") {
    return sample();
  } else if (_tw._sampling_method == "topk_greedy") {
    return topk_greedy_search();
  } else if (_tw._sampling_method == "beam_search") {
    return beam_search();
  } else {
    throw std::runtime_error("not supported sampling_method");
  }
}  // namespace cuda

/**
Decode embedding
*/
template <OperationType OpType_>
void Decoder<OpType_>::embedding() {
  // _p_d_trg_emb_wei: {token_emb, position_emb, norm_scale, norm_bias,
  // enc_out_kernel_kv, enc_out_bias_kv, logit_bias} //BTBT 当_tw._multilg_type == 0时,只有两种lang type,此时没有_p_d_trg_emb_wei[7]; _tw._multilg_type != 0 时 _p_d_trg_emb_wei[7] 里面是lang type emb,也可做tkn typ emb
  launch_dec_emb<_DataType>(_p_d_trg_emb_wei[0], _p_d_trg_emb_wei[1],
                            _p_d_alive_seq, _p_d_trg_emb_wei[7], _p_d_lang_id, //当_tw._multilg_type == 0时,_p_d_lang_id为nullptr,其它情况要指定target emb对应的tkn tye 类型
                            _p_d_cur_step_query, _batch_size, _tw._beam_size, //BTBT _p_d_cur_step_query作为emb的output,用于attn的Query,只处理当前_cur_step对应的那个emb
                            _tw._hidden_size, _tw._trg_vocab_size, _cur_step,
                            _tw._max_step, _tw._multilg_type, _stream);
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _tw._beam_size; j++) {  // beam_id
      std::cout << "decoder emb: batch-" << i << ", beam-" << j << std::endl;
      print_vec(_p_d_cur_step_query + i * _tw._beam_size * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb", 10);
    }
  }
#endif
  return;
}

/**
Decoder feedforward, composed by self_atten,
  enc-dec-atten, ffn
*/
template <OperationType OpType_>
void Decoder<OpType_>::decoder_stack() {
  // _p_d_dec_wei = {self_norm_scale, self_norm_bias,
  // self_qkv_kernel, self_qkv_bias, self_output_kernel, self_output_bias
  // encdec_norm_scale, encdec_norm_bias,
  // encdec_q_kernel, encdec_q_bias, encdec_output_kernel, encdec_output_bias
  // ffn_norm_scale, ffn_norm_bias, ffn_first_kernel, ffn_first_bias,
  // ffn_second_kernel, ffn_second_bias} * encoder_layer_num
  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_dec_layer;

    self_attention();

    encdec_attention();

    ffn_add_norm();
  }

  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_trg_emb_wei[2], _p_d_trg_emb_wei[3], _max_thread_per_block);
  return;
}

/**
Decoder self attention
*/
template <OperationType OpType_>
void Decoder<OpType_>::self_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,//input
      _p_d_query_buf1/*output*/, _p_d_dec_wei[_weight_offset],//self_norm_scale
      _p_d_dec_wei[_weight_offset + 1]/*self_norm_bias*/, _p_d_dec_wei[_weight_offset + 5]/*self_project_bias_output*/,
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn ln(tail): ", 5);
#endif

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(//BTBT ??? 只把当前step的q扩展为qkv
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 2]/*self_project_kernel_qkv*/, _AType,
      _tw._hidden_size * 3, _p_d_query_buf1, _BType, _tw._hidden_size,
      &_type_zero, _p_d_self_step_qkv, _CType, _tw._hidden_size * 3,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_self_step_qkv, "self qkv(head): ", 5);
  print_vec(_p_d_self_step_qkv + _step_token_num * _tw._hidden_size * 3 - 5,
            "self qkv(tail): ", 5);
#endif
  //BTBT 计算当前step的qkv+bias,并分割成_p_d_query_buf1(qOfCurrentStep),_p_d_self_k_bgeem1[_layer_id](kOfCurrentStepAndHistoryStep)_p_d_self_v_bgeem1也是包含历史step(虽然该方法只处理当前step)
  // get q, k, v by split and reshape qkv
  ker_arrange_decself_qkv_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_self_step_qkv,
      _p_d_dec_wei[_weight_offset + 3]/*self_project_bias_qkv*/, _p_d_query_buf1,
      _p_d_self_k_bgeem1[_layer_id], _p_d_self_v_bgeem1[_layer_id],
      _tw._head_num, _tw._dim_per_head, _tw._max_step, _cur_step,
      _max_thread_per_block);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn q(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn q(tail): ", 5);
  print_vec(_p_d_self_k_bgeem1[_layer_id] +
                _cur_step * _tw._hidden_size / _tw._head_num,
            "self attn k(head): ", 5);
  print_vec(
      _p_d_self_k_bgeem1[_layer_id] +
          _step_token_num * _tw._hidden_size * _tw._max_step -
          ((_tw._max_step - _cur_step - 1) * _tw._hidden_size / _tw._head_num) -
          5,
      "self attn k(tail): ", 5);
  print_vec(_p_d_self_v_bgeem1[_layer_id] +
                _cur_step * _tw._hidden_size / _tw._head_num,
            "self attn v(head): ", 5);
  print_vec(
      _p_d_self_v_bgeem1[_layer_id] +
          _step_token_num * _tw._hidden_size * _tw._max_step -
          ((_tw._max_step - _cur_step - 1) * _tw._hidden_size / _tw._head_num) -
          5,
      "self attn v(tail): ", 5);
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(//BTBT 沿headsOfCurrentStep的轴线,按head计算当前step的q与之前step(含当前)的k的点积分数
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _cur_step + 1, 1, _tw._dim_per_head,  //??? 这里CUBLAS_OP_T和CUBLAS_OP_N的影响貌似正相反?是column-major的影响?
      &_atten_scaler, _p_d_self_k_bgeem1[_layer_id]/*xbatchSz*beamSz*headNum*xstep*dimHead */, _AType, _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _p_d_query_buf1/*xbatchSz*beamSz*hidSz */, _BType,
      _tw._dim_per_head, _tw._dim_per_head, &_type_zero, _p_d_c/*xbatchSz*beamSz*headNum*xstep */, _CType,
      _cur_step + 1, _cur_step + 1, (_step_token_num * _tw._head_num)/*batchCount=(batchSz*beamSz)*headNum=>headsOfStep */,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_correlation_softmax_decself_launcher(_step_token_num * _tw._head_num,
                                           _cur_step + 1, _stream, _p_d_c);//只有当前step和之前的step相关的correlation参与sftmx计算

#ifdef DEBUG_RESULT
  print_vec(_p_d_c, "self attn corr(head): ", 5);
  print_vec(_p_d_c + _step_token_num * _tw._head_num * (_cur_step + 1) - 5,
            "self attn corr(tail): ", 5);
#endif

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(//BTBT 沿headsOfCurrentStep的轴线,按head计算当前step的新q值,correlation * v的计算均涉及当前和历史step
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, 1, _cur_step + 1,
      &_type_one, _p_d_self_v_bgeem1[_layer_id]/*xbatchSz*beamSz*headNum*xstep*dimHead */, _AType, _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _p_d_c/*xbatchSz*beamSz*headNum*xstep */, _BType, _cur_step + 1,
      _cur_step + 1, &_type_zero, _p_d_query_buf1/*xbatchSz*beamSz*headNum*dimHead */, _CType, _tw._dim_per_head,
      _tw._dim_per_head, _step_token_num * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn before ffn(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn before ffn(tail): ", 5);
#endif

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(//BTBT 只涉及当前step的q计算
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 4]/*self_project_kernel_output*/, _AType,
      _tw._hidden_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_one,
      _p_d_cur_step_query/*xbatchSz*beamSz*hidSz,已经在前面ker_norm_layer_resual_launcher时+bias*/, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "self attn out(head): ", 3);
  print_vec(_p_d_cur_step_query + _step_token_num * _tw._hidden_size - 3,
            "self attn out(tail): ", 3);
#endif
}

/**
Encode-Decoder attention
*/
template <OperationType OpType_>
void Decoder<OpType_>::encdec_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 6]/*encdec_norm_scale*/,
      _p_d_dec_wei[_weight_offset + 7]/*encdec_norm_bias*/, _p_d_dec_wei[_weight_offset + 11]/*encdec_project_bias_output*/,
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "encdec attn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "encdec attn ln(tail): ", 5);
#endif

  /* ---step 1. new_q = ori_q * q_wei + bias, reshape new_q for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 8]/*encdec_project_kernel_q*/, _AType,
      _tw._hidden_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_zero,
      _p_d_query_buf2, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_arrange_encdec_q_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_query_buf2/*in:ori_q*/,
      _p_d_dec_wei[_weight_offset + 9]/*encdec_project_bias_q*/, _p_d_query_buf1/*out:new_q*/, _tw._beam_size,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _tw._beam_size,
      _tw._dim_per_head, &_atten_scaler, _p_d_encdec_k_bgeem[_layer_id], _AType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, _p_d_query_buf1,
      _BType, _tw._dim_per_head, _tw._beam_size * _tw._dim_per_head,
      &_type_zero, _p_d_c, _CType, _batch_seq_len,
      _tw._beam_size * _batch_seq_len, _batch_size * _tw._head_num,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_correlation_softmax_encdec_launcher<_DataType>(
      _batch_size, _tw._head_num * _tw._beam_size, _batch_seq_len, _stream,
      _p_d_c, _p_d_padding_mask);

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _tw._beam_size,
      _batch_seq_len, &_type_one, _p_d_encdec_v_bgeem[_layer_id], _AType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, _p_d_c, _BType,
      _batch_seq_len, _tw._beam_size * _batch_seq_len, &_type_zero,
      _p_d_query_buf1, _CType, _tw._dim_per_head,
      _tw._beam_size * _tw._dim_per_head, _batch_size * _tw._head_num,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_arrange_atten_output_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_query_buf1,
      _p_d_query_buf2, _tw._beam_size, _tw._dim_per_head, _tw._head_num,
      _max_thread_per_block);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_query_buf2, _BType, _tw._hidden_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
void Decoder<OpType_>::ffn_add_norm() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13], _p_d_dec_wei[_weight_offset + 17],
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "ffn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "ffn ln(tail): ", 5);
#endif

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 14], _AType,
      _tw._inner_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_zero,
      _p_d_query_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (_tw._use_gelu) {
    ker_bias_gelu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15], _tw._inner_size);
  } else {
    ker_bias_relu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15], _tw._inner_size);
  }

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._inner_size, &_type_one, _p_d_dec_wei[_weight_offset + 16], _AType,
      _tw._hidden_size, _p_d_query_buf2, _BType, _tw._inner_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
bool Decoder<OpType_>::sample() {
  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  if (_tw._sampling_method == "topk") {
    ker_topk_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
        _p_d_alive_seq_buf, _tw._trg_vocab_size, _tw._topk,
        _p_d_sample_unfinished, _p_d_curandstate, _tw._end_id);
  } else {
    ker_topp_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
        _p_d_alive_seq_buf, _tw._trg_vocab_size, _tw._topp,
        _p_d_sample_unfinished, _p_d_curandstate, _tw._end_id);
  }
#ifdef DEBUG_RESULT
  print_vec(_p_d_sample_unfinished, "unfinished flag", 1);
  for (int ii = 0; ii < _batch_size; ii++) {
    print_vec(_p_d_alive_seq + ii * _tw._max_step,
              "Batch token ids: ", _cur_step + 2);
  }
#endif

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_sample_unfinished,
                                  sizeof(int), cudaMemcpyDeviceToHost,
                                  _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _h_unfinished == 1 ? false : true;
}

template <OperationType OpType_>
bool Decoder<OpType_>::beam_search() {
  /*
    step 1. logits bias and softmax,
      select rough topk candidate for every batch item,
      record the candidate's beam_id, vocab_id and probability
  */
  update_new_seq_probs();

  /* ---step 2. sort the candidate with their probability--- */
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_can_num_batch, _p_d_can_num, sizeof(int),//BTBT _p_d_can_num指向的第一个数是该step该batch中所有beam所找到的topK数值的总个数
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  if (_tw._diverse_lambda != 0) {
    if (_h_can_num_batch < _cub_sort_buffer_bytes / 160) {
      CHECK_GPU_ERROR(cub::DeviceRadixSort::SortPairsDescending(
          (float*)_p_d_logit_buf, _cub_sort_buffer_bytes, _p_d_can_score,
          _p_d_can_score, _p_d_can_idx, _p_d_can_idx, _h_can_num_batch, 0,
          sizeof(float) * 8, _stream));
    } else {
      thrust::sort_by_key(thrust::cuda::par.on(_stream), _p_d_can_score,
                          _p_d_can_score + _h_can_num_batch, _p_d_can_idx,
                          thrust::greater<float>());
    }
    ker_diverse_beam_search_launcher(_p_d_can_score, _p_d_can_idx, _p_d_can_num,
                                     _step_token_num, _max_thread_per_block,
                                     _stream, _tw._beam_size,
                                     _tw._diverse_lambda, _tw._trg_vocab_size);
  }
  
  thrust::sort_by_key(thrust::cuda::par.on(_stream), _p_d_can_score,//BTBT 因在select_beam_rough_topk()中'batch_id * min_log_probability',所以这排序后batch_id(也就是该batch中各样本的顺序位置)不变,只是样本中的那几个beam对应的tkn id按score从大到小排序了
                      _p_d_can_score + _h_can_num_batch, _p_d_can_idx, //BTBT 根据 _p_d_can_score 里的key对_p_d_can_idx 进行排序
                      thrust::greater<float>()); // 排序方法 https://thrust.github.io/doc/group__sorting_gaec4e3610a36062ee3e3d16607ce5ad80.html#gaec4e3610a36062ee3e3d16607ce5ad80

#ifdef DEBUG_RESULT
  print_vec(_p_d_can_score, "can score", _h_can_num_batch);
  print_vec(_p_d_can_idx, "can idx", _h_can_num_batch);
#endif

  /*
    step 3. refresh alive_seq, seq_probs, seq_score, num_finish_beam
      based on sorted candidate.
      Deciding whether early stop based on num_finish_beam
  *///BTBT ??? 若diverse_lambda==0,每个beam只选择top1分数的候选
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_can_num, 0, sizeof(int), _stream));
  ker_refresh_result<<<dim3(_batch_size, _tw._beam_size), _tw._max_step, 0,
                       _stream>>>(
      _p_d_can_idx, _p_d_can_score, _p_d_can_num + 1, _p_d_alive_seq/*old_alive_seq*/,
      _p_d_alive_seq_buf/*new_alive_seq*/, _p_d_alive_seq_probs, _p_d_alive_seq_score,//??? p_d_alive_seq_score只有遇到trg_end_id才更新?
      _p_d_can_num, _tw._trg_vocab_size, _cur_step, _h_length_norm[_cur_step],
      _tw._diverse_lambda, _tw._end_id);
  int* tmp = _p_d_alive_seq_buf;
  _p_d_alive_seq_buf = _p_d_alive_seq;
  _p_d_alive_seq = tmp;
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_can_num_batch, _p_d_can_num, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

#ifdef DEBUG_RESULT
  for (int ii = 0; ii < _batch_size; ii++) {
    for (int jj = 0; jj < _tw._beam_size; jj++) {
      print_vec(_p_d_alive_seq + (ii * _tw._beam_size + jj) * _tw._max_step,
                "Batch token ids: ", _cur_step + 2);
      print_vec(_p_d_alive_seq_probs + ii * _tw._beam_size + jj,
                "Batch probs: ", 1);
      print_vec(_p_d_alive_seq_score + ii * _tw._beam_size + jj,
                "Batch scores: ", 1);
    }
  }
#endif

  if (_h_can_num_batch == _step_token_num) {//BTBT 当batch中每个beam选的tkn都是end_id的时候,就会h_can_num_batch==step_token_num,否则都是>step_token_num
#ifdef DEBUG_RESULT
    std::cout << "early stop beam search!" << std::endl;
#endif
    return true;
  }

  /* ---step 4. refresh cache: k, v for decoder self attention--- */
  if (_cur_step > 0) {
    ker_refresh_cache_launcher<_DataType>(
        _tw._n_dec_layer * (_cur_step + 1) /*grid_dim_x*/ , _step_token_num * 2 /*grid_dim_y*/,
        _max_thread_per_block /*block_dim*/, _stream, _p_d_can_num + 1, _p_d_can_idx,
        _p_d_self_k_bgeem1[0]/*self_k_bgeem*/, _p_d_self_v_bgeem1[0], _p_d_self_k_bgeem2[0]/*new_self_k_bgeem*/,
        _p_d_self_v_bgeem2[0], _layer_size_self_k, _tw._beam_size,
        _tw._dim_per_head, _tw._head_num, _tw._trg_vocab_size, _cur_step,
        _tw._max_step, _tw._diverse_lambda != 0, _tw._end_id);
    _DataType** ftmp = _p_d_self_k_bgeem2;
    _p_d_self_k_bgeem2 = _p_d_self_k_bgeem1;//??? _p_d_self_k_bgeem1和_p_d_self_k_bgeem2的用途?ker_refresh_cache_launcher的意图?
    _p_d_self_k_bgeem1 = ftmp;
    ftmp = _p_d_self_v_bgeem2;
    _p_d_self_v_bgeem2 = _p_d_self_v_bgeem1;
    _p_d_self_v_bgeem1 = ftmp;
  }
  return false;
}

/**
Logits bias and softmax.
Select rough topk candidate for every batch item.
Record the candidate's beam_id, vocab_id and probability
*/
template <OperationType OpType_>
void Decoder<OpType_>::update_new_seq_probs() {
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_can_num, 0, sizeof(int), _stream));

  select_beam_rough_topk_launcher(
      _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,//???_p_d_alive_seq_probs和_p_d_alive_seq_score的作用与更新机制
      _p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score,
      _p_d_can_num, _tw._trg_vocab_size, _tw._max_step,
      _h_length_norm[_cur_step], _cur_step, _step_token_num,
      _max_thread_per_block, _stream, _tw._beam_size, _tw._diverse_lambda,
      _tw._end_id);
  thrust::exclusive_scan(thrust::cuda::par.on(_stream), _p_d_can_num + 1, https://thrust.github.io/doc/group__prefixsums_ga0f1b7e1931f6ccd83c67c8cfde7c8144.html#ga0f1b7e1931f6ccd83c67c8cfde7c8144 , https://www.cnblogs.com/carekee/articles/2409514.html
                         _p_d_can_num + 1 + _step_token_num, _p_d_can_num + 1);//BTBT 原来_p_d_can_num保存了该batch每个beam的各个topK数目,exclusive_scan后变成了从第一个beam开始累计的topK数,也就是可直接作为查_p_d_can_idx和_p_d_can_score中候选(topK)tkn id及其分数的下标
  return;
}

template <OperationType OpType_>
bool Decoder<OpType_>::topk_greedy_search() {
  _tw._diverse_lambda = 0;
  if (_cur_step == 0) {
    return beam_search();
  }

  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  ker_topk_sample_launcher<_DataType>(
      _step_token_num, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
      _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
      _p_d_alive_seq_buf, _tw._trg_vocab_size, 1, _p_d_sample_unfinished,
      _p_d_curandstate, _tw._end_id);

#ifdef DEBUG_RESULT
  print_vec(_p_d_sample_unfinished, "unfinished flag", 1);
  for (int ii = 0; ii < _batch_size; ii++) {
    for (int jj = 0; jj < _tw._beam_size; jj++) {
      print_vec(_p_d_alive_seq + (ii * _tw._beam_size + jj) * _tw._max_step,
                "Batch token ids: ", _cur_step + 2);
    }
  }
#endif

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_sample_unfinished,
                                  sizeof(int), cudaMemcpyDeviceToHost,
                                  _stream));

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _h_unfinished == 1 ? false : true;
}

template class Decoder<OperationType::FP16>;
template class Decoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
