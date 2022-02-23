/* Copyright 2021 The LightSeq Team
   Copyright Tencent/TurboTransformers
   This block_reduce_n is adapted from Tencent/TurboTransformers
*/
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum class ReduceType { kMax = 0, kSum };
const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const float REDUCE_FLOAT_INF_NEG = -100000000.f;
const float REDUCE_FLOAT_INF_POS = 100000000.f;
const unsigned int WARP_REDUCE_SIZE = 32;

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int mask = (WARP_REDUCE_SIZE >> 1); mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_REDUCE_SIZE);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum<T>(val);
  return val;
}

template <ReduceType Rtype, int Num>
__inline__ __device__ void blockReduce(float *pval);

// use template to make code more concise
template <ReduceType Rtype, int Num>
__inline__ __device__ void warpReduce(float *pval);

// static
template <>
__inline__ __device__ void warpReduce<ReduceType::kMax, 1>(float *pval) {
  *pval = max(*pval, __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 16, 32));
  *pval = max(*pval, __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 8, 32));
  *pval = max(*pval, __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 4, 32));
  *pval = max(*pval, __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 2, 32));
  *pval = max(*pval, __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 1, 32));
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kMax, 2>(float *pval) {
  float val0_tmp, val1_tmp;
#define WarpReduceMaxOneStep(a, b)                                 \
  val0_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval), a, b);     \
  val1_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), a, b); \
  *(pval) = max(val0_tmp, *(pval));                                \
  *(pval + 1) = max(val1_tmp, *(pval + 1));

  WarpReduceMaxOneStep(16, 32);
  WarpReduceMaxOneStep(8, 32);
  WarpReduceMaxOneStep(4, 32);
  WarpReduceMaxOneStep(2, 32);
  WarpReduceMaxOneStep(1, 32);
#undef WarpReduceMaxOneStep
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 1>(float *pval) {
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 16, 32);//BTBT 见 [https://zhuanlan.zhihu.com/p/203648521]
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 8, 32);//__shfl_xor_sync()通过对调用者的通道ID与laneMask进行按位异或（XOR）运算来计算源通道ID。返回值为计算所得源通道中的var值。此模式实现了蝶形寻址模式。
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 4, 32);//mask=WARP_REDUCE_MASK=0xffffffff表示warp内所有线程都是活跃线程, val=*pval, laneMask=4, width=32=warpSize
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 2, 32);
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 1, 32);
}

/*
 * Unorll for loop for warpreduce to
 * imporve instruction issue efficiency
 * ElemX means there are X numbers to be summed
 */

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 2>(float *pval) {
  float val0_tmp, val1_tmp;
#define WarpReduceSumOneStep(a, b)                                 \
  val0_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), a, b); \
  val1_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), a, b); \
  *(pval + 0) += val0_tmp;                                         \
  *(pval + 1) += val1_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);

#undef WarpReduceSumOneStep
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 4>(float *pval) {
  float val0_tmp, val1_tmp, val2_tmp, val3_tmp;
#define WarpReduceSumOneStep(a, b)                                 \
  val0_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), a, b); \
  val1_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), a, b); \
  val2_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 2), a, b); \
  val3_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 3), a, b); \
  *(pval + 0) += val0_tmp;                                         \
  *(pval + 1) += val1_tmp;                                         \
  *(pval + 2) += val2_tmp;                                         \
  *(pval + 3) += val3_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);
#undef WarpReduceSumOneStep
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kSum, 1>(float *pval) {
  const int num = 1;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;//BTBT 十六进制0x1f=31,这里的意思是模32
  int wid = threadIdx.x >> 5;//BTBT 2的5次方=32,这里是除以32, 这里可理解为求warpId

  warpReduce<ReduceType::kSum, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);//把blk内每个warp做allReduce得到的值保存到share mem相应的位置[wid]上
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {//该blk能分成的warp数目就是'blockDim.x >> 5',这里是以blk内threaIdx==warpId的thread作为对应的warp的reduce值的处理者,对应的pval的值为对应的warp做reduce后的结果
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {//threadIdx!=warpId的对应的pval为0
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = 0.f;
    }
  }
  warpReduce<ReduceType::kSum, num>(pval);//该操作后,该blk中threadIdx==warpId的thread中的pval就是blk中各个thread的pval的allReduce结果了
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kSum, 2>(float *pval) {
  const int num = 2;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kSum, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = 0.f;
    }
  }
  warpReduce<ReduceType::kSum, num>(pval);
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kSum, 4>(float *pval) {
  const int num = 4;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kSum, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = 0.f;
    }
  }
  warpReduce<ReduceType::kSum, num>(pval);
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kMax, 1>(float *pval) {
  const int num = 1;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kMax, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = REDUCE_FLOAT_INF_NEG;
    }
  }
  warpReduce<ReduceType::kMax, num>(pval);
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kMax, 2>(float *pval) {
  const int num = 1;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kMax, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = REDUCE_FLOAT_INF_NEG;
    }
  }
  warpReduce<ReduceType::kMax, num>(pval);
}

template <>
__inline__ __device__ void blockReduce<ReduceType::kMax, 4>(float *pval) {
  const int num = 1;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kMax, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = REDUCE_FLOAT_INF_NEG;
    }
  }
  warpReduce<ReduceType::kMax, num>(pval);
}
