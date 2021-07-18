#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>//btbt https://nvlabs.github.io/cub/index.html

#include "block_reduce.h"
#include "kernels.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

/**
@brief: softmax_kernel
Softmax forward kernel for
  enc-self-attn, dec-self-attn, encdec-attn

@thread
gridDim.x = dynamic
gridDim.y = batch_size
gridDim.z = nhead
blockDim.x = from_len

@param
inp: [batch_size, nhead, from_len, to_len], softmax input.
attn_mask: [batch_size, to_len], padding tokens are -inf,
  non padding tokens are 0.
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
template <typename T, int block_dim, int ele_per_thread> //BTBT [BUG] ??? 用BLOCK_STORE_VECTORIZE的话item per thread应该是4的倍数会好些?
__global__ void ker_attn_softmax(T *inp, const T *attn_mask, int from_len,
                                 int to_len, bool mask_future) {//btbt ??? 这里Q的seq_len为from_len,K的为to_len,可以设为不同,是为了兼容cross attn?
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,//btbt https://nvlabs.github.io/cub/classcub_1_1_block_load.html
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,//https://nvlabs.github.io/cub/classcub_1_1_block_store.html
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;//btbt attn_mask指向的值为const但指针可移动,现移动到该sample的token起始位置(不用headId是因为每个头的mask都一样)
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);//btbt mval指向哪个to tkn,与下面的inp_val一样,开始指针指向个to tkn基于(该block中的)threadIdx*ele_per_thread计算
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);//指针移动到该sample该头的token起始位置
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;//gridDim.x控制Q的from_len上token的循环,每次取token_per_reduce个token与K的to_len个token计算得到的attn score做softmax计算,下一轮又取相隔gridDim.x * token_per_reduce的下一个token进行计算,用于token多而一个grid计算不完的情况
       token_id += gridDim.x * token_per_reduce) {//这样在from_len=512,gridDim.x=64,token_per_reduce=1的情况下相当于每个thread计算Q的from_len=512个token中的8个,每个block有256个thrad处理一个Qtoken的softmax,每个thread处理该Qtkn对应的2个K token的attn score求和,最终汇总到block对应的QK的softmax值
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,//btbt inp指针移动到第token_id个token的attn score起始位置,因每个from tkn要和to_len个to tkn做attn score,所以要乘以to_len
                              REDUCE_FLOAT_INF_NEG);//btbt inp_val存的是第token_id个from tkn对应的to_len个to tkn的ele_per_thread个to tkn的attn score,开始指针指向个to tkn基于(该block中的)threadIdx*ele_per_thread计算
    }

    /* step 1. compute max */
    // thread local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];//btbt ??? 该thread的每个reduce的最大值?
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        if (attn_mask) {
          val[i][j] = (float)inp_val[i][j] + (float)mval[j];//加上mask
        } else {
          if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
            val[i][j] = REDUCE_FLOAT_INF_NEG;
          } else {
            val[i][j] = (float)inp_val[i][j];
          }
        }
        l_max[i] = fmaxf(l_max[i], val[i][j]);
      }
    }
    // block reduce max //btbt [maybug]???l_max不是数组么,这种传参只适合token_per_reduce=1的情况吧?还是都可以?
    blockReduce<ReduceType::kMax, token_per_reduce>(l_max);//btbt 该block的每个thread传入本地最大值l_max,然后'blockReduce<ReduceType::kMax'计算出block的最大值保存在l_max中
    // write shared
    __shared__ float s_max[token_per_reduce];//共享内存s_max保存该block的最大值
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_max[i] = l_max[i];
      }
    }
    __syncthreads();//btbt [REFACTOR] __syncthreads有点多,blockReduce里面也还有,这样会影响性能

    /* step 2. compute sum */
    // thread local sum
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {//block的每个thread基于block最大值归一化各自负责的val,然后对这些val求和(thread local sum)
        val[i][j] = __expf(val[i][j] - s_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // block reduce sum
    blockReduce<ReduceType::kSum, token_per_reduce>(l_sum);//对整个block的所有thread的l_sum进行求和,block和放在l_sum中. 这种处理方式 //btbt [REFACTOR]???这种自己写的方式会比cub::BlockReduce快?
    // write shared
    __shared__ float s_sum[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);//求blck的和l_sum的倒数
      }
    }
    __syncthreads();

    /* step 3. compute final result */
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * s_sum[i]);//每个thrad所负责的val归一化后再乘以block和的倒数,得到softmax最后结果
      }
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],//把block中每个thrad所负责的token对应的softmax结果都保存在inp中对应的Qtoken的各个相应的Ktoken中
                                 to_len);
    }
  }  // blockIdx.x
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(T *inp, const T *attn_mask, int from_len,
                                      int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        if (attn_mask) {
          val[i][j] = (float)inp_val[i][j] + (float)mval[j];
        } else {
          if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
            val[i][j] = REDUCE_FLOAT_INF_NEG;
          } else {
            val[i][j] = (float)inp_val[i][j];
          }
        }
        l_max[i] = fmaxf(l_max[i], val[i][j]);
      }
    }
    // warp reduce max
    warpReduce<ReduceType::kMax, token_per_reduce>(l_max);

    /* step 2. compute sum */
    // thread local sum
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - l_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // warp reduce sum
    warpReduce<ReduceType::kSum, token_per_reduce>(l_sum);

    /* step 3. compute final result */
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      l_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * l_sum[i]);
      }
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
  }  // blockIdx.x
}

/*
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
template <>
void launch_attn_softmax<float>(float *inp, const float *attn_mask,
                                int batch_size, int nhead, int from_len,
                                int to_len, bool mask_future,
                                cudaStream_t stream) {
  dim3 grid_dim(1, batch_size, nhead);//btbt grid dim相对固定,但block dim要基于序列长度和每个thread要处理的token数调整,保证一个block能处理完单个sample序列的所有seq_len个token,
  if (to_len <= 32) {
    ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 128) {
    grid_dim.x = 16;
    ker_attn_softmax<float, 64, 2><<<grid_dim, 64, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 256) {
    grid_dim.x = 32;
    ker_attn_softmax<float, 128, 2><<<grid_dim, 128, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 512) {
    grid_dim.x = 64;//btbt ??? 为何此处grid_dim.x要为64 grid<64,batchSz,heads>,block<256> ??? 序列长点又要多一个else if?
    ker_attn_softmax<float, 256, 2><<<grid_dim, 256, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 512 is currently not supported");
  }
}

template <>
void launch_attn_softmax<__half>(__half *inp, const __half *attn_mask,
                                 int batch_size, int nhead, int from_len,
                                 int to_len, bool mask_future,
                                 cudaStream_t stream) {
  dim3 grid_dim(1, batch_size, nhead);
  if (to_len <= 32) {
    ker_attn_softmax_lt32<__half, 32, 1><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<__half, 32, 2><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 128) {
    grid_dim.x = 8;
    ker_attn_softmax<__half, 64, 2><<<grid_dim, 64, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 256) {
    grid_dim.x = 16;
    ker_attn_softmax<__half, 128, 2><<<grid_dim, 128, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 512) {
    grid_dim.x = 32;
    ker_attn_softmax<__half, 256, 2><<<grid_dim, 256, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 512 is currently not supported");
  }
}

/**
@brief: ker_attn_softmax_bw
Softmax backward in self attention.

@thread
gridDim.x = batch_size * nhead * seq_len / warps_per_block
blockDim.x = WARP_SIZE
blockDim.y = warps_per_block

@param
grad: [batch_size, nhead, seq_len, seq_len], output grad.
output: [batch_size, nhead, seq_len, seq_len], output of softmax forward.
*/
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(T *grad, const T *inp, int softmax_length) {//BTBT *** 这里和dropout&LN的backward使用tiled_partition的方式不同,从一开始就把gridDim和blockDim的处理变成列优先的,所以后面不用做别扭的从行优先变成列优先的转换,省了不少时间
  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int offset = batch_idx * softmax_length + threadIdx.x;

  grad += offset;
  inp += offset;

  T grad_reg[ITERATIONS];
  T inp_reg[ITERATIONS];
  float sum = 0.0;

#pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length) {
      grad_reg[i] = grad[i * WARP_SIZE];
      inp_reg[i] = inp[i * WARP_SIZE];
      sum += (float)grad_reg[i] * (float)inp_reg[i];
    }
  }

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

#pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length)
      grad[i * WARP_SIZE] = (T)((float)inp_reg[i] * ((float)grad_reg[i] - sum));
  }
}

template <typename T>
void launch_attn_softmax_bw(T *out_grad, const T *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream) {
  const int warps_per_block = 4;
  // rows = batch_size * nhead * from_len
  dim3 grid_dim(rows / warps_per_block);
  dim3 block_dim(WARP_SIZE, warps_per_block);

  if (softmax_len <= 32)
    ker_attn_softmax_bw<T, 1>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 64)
    ker_attn_softmax_bw<T, 2>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 128)
    ker_attn_softmax_bw<T, 4>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 256)
    ker_attn_softmax_bw<T, 8>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 384)
    ker_attn_softmax_bw<T, 12>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 512)
    ker_attn_softmax_bw<T, 16>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 768)
    ker_attn_softmax_bw<T, 24>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 1024)
    ker_attn_softmax_bw<T, 32>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 2048)
    ker_attn_softmax_bw<T, 64>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else
    throw std::runtime_error(
        std::string(
            "Special sequence length found in softmax backward, seq_len: ") +
        std::to_string(softmax_len));
}

template void launch_attn_softmax_bw<__half>(__half *out_grad,
                                             const __half *soft_inp, int rows,
                                             int softmax_len,
                                             cudaStream_t stream);
template void launch_attn_softmax_bw<float>(float *out_grad,
                                            const float *soft_inp, int rows,
                                            int softmax_len,
                                            cudaStream_t stream);
