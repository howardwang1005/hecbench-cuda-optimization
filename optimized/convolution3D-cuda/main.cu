/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cuda.h>

#define TILE_WIDTH 16

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

#ifdef CUDNN_CONV
#include <cudnn.h>
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
    }                                                        \
  }
#endif

template <typename T>
void verify (const T* Y, T* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Optimized conv3d (FP32). Profiling the BASELINE's heavy layer
// (32 96 256 26 26 5, job 952657) showed compute/instruction-bound: DRAM 1.4%,
// L1/L2 hit 95/97%, SM 79.7%, occupancy 95%, 51 waves. The per-FMA address
// integer math (recomputing II/WI each iteration) and the W reads through L1
// dilute the FMA throughput. Two changes per kernel:
//   1. Stage this block's filter slice W[m] (C*K*K floats, fixed m per block)
//      into shared memory once, reused by all 256 threads. (m fits: 9.6KB for
//      the heavy layer, 600B for the small layer; W=2.4MB is too big for
//      __constant__, but one m-slice fits in shared.)
//   2. Template on the compile-time kernel size KK so the K x K inner loops
//      unroll, and hoist xbase/wbase per channel so the inner loop uses only
//      constant offsets -> far fewer address instructions per FMA.
// KK = 0 is a runtime fallback for uncommon K.
// ---------------------------------------------------------------------------
template<typename T, int KK>
__device__ __forceinline__
T conv3d_acc(const T* __restrict__ X, const float* __restrict__ sW,
             int n, int h, int w, int C, int K, int Hin, int Win)
{
  T s = 0;
  const int kk = (KK > 0) ? KK : K;
  const T* xc = X + (size_t)n * C * Hin * Win + (size_t)h * Win + w;  // II(n,0,h,w)
  const float* wc = sW;                                              // sW[c*K*K]
  for (int c = 0; c < C; c++) {
    if constexpr (KK > 0) {
      #pragma unroll
      for (int p = 0; p < KK; p++)
        #pragma unroll
        for (int q = 0; q < KK; q++)
          s += xc[p * Win + q] * wc[p * KK + q];
    } else {
      for (int p = 0; p < kk; p++)
        for (int q = 0; q < kk; q++)
          s += xc[p * Win + q] * wc[p * kk + q];
    }
    xc += Hin * Win;     // next channel of X
    wc += kk * kk;       // next channel of W slice in shared
  }
  return s;
}

// cooperatively load filter slice W[m] (C*K*K floats) into shared memory
template<typename T>
__device__ __forceinline__
void load_filter(float* sW, const T* __restrict__ W, int m, int C, int K)
{
  const int wsz = C * K * K;
  const T* Wm = W + (size_t)m * wsz;
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < wsz;
       i += blockDim.x * blockDim.y)
    sW[i] = (float)Wm[i];
  __syncthreads();
}

template<typename T, int KK>
__global__
void conv3d_s1(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C, const int M, const int K,
               const int Hin, const int Win, const int Hout, const int Wout,
               const int W_grid)
{
  extern __shared__ float sW[];
  int m = blockIdx.y;
  load_filter(sW, W, m, C, K);
  int n = blockIdx.x;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout)
    Y[OI(n, m, h, w)] = conv3d_acc<T,KK>(X, sW, n, h, w, C, K, Hin, Win);
}

template<typename T, int KK>
__global__
void conv3d_s2(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C, const int M, const int K,
               const int Hin, const int Win, const int Hout, const int Wout,
               const int W_grid)
{
  extern __shared__ float sW[];
  int m = blockIdx.x;
  load_filter(sW, W, m, C, K);
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout)
    Y[OI(n, m, h, w)] = conv3d_acc<T,KK>(X, sW, n, h, w, C, K, Hin, Win);
}

template<typename T, int KK>
__global__
void conv3d_s3(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C, const int M, const int K,
               const int Hin, const int Win, const int Hout, const int Wout,
               const int W_grid)
{
  extern __shared__ float sW[];
  int m = blockIdx.z;
  load_filter(sW, W, m, C, K);
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  if (h < Hout && w < Wout)
    Y[OI(n, m, h, w)] = conv3d_acc<T,KK>(X, sW, n, h, w, C, K, Hin, Win);
}


// Hin = Hout-1+K; max(h+p) is Hin - 1 as max(h) = Hout-1 and max(p) = K-1
template <typename T>
void reference(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int N,
               const int M,
               const int C,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout)
{
  for(int n = 0; n < N; n++)
    for(int m = 0; m < M; m++)
      for(int h = 0; h < Hout; h++)
        for(int w = 0; w < Wout; w++) {
          Y[OI(n, m, h, w)] = 0;
          for(int c = 0; c < C; c++)
            for(int p = 0; p < K; p++)
              for(int q = 0; q < K; q++)
                Y[OI(n, m, h, w)] += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
}

template <typename T>
void conv3D(const int N, const int C, const int M, const int Win, const int Hin, const int K, const int repeat)
{
  const int Hout = Hin-K+1;
  const int Wout = Win-K+1;

  size_t X_size = N * C * Hin * Win;
  size_t W_size = M * C * K * K;
  size_t Y_size = N * M * Hout * Wout;
  size_t X_bytes = X_size * sizeof(T);
  size_t W_bytes = W_size * sizeof(T);
  size_t Y_bytes = Y_size * sizeof(T);

  T *X, *W, *Y, *Y_ref;
  X = (T *)malloc(X_bytes); // input
  W = (T *)malloc(W_bytes); // filter
  Y = (T *)malloc(Y_bytes); // output
  Y_ref = (T *)malloc(Y_bytes);

  srand(123);

  for (size_t i = 0; i < W_size; i++) W[i] = rand() % 31;
  for (size_t i = 0; i < X_size; i++) X[i] = rand() % 13;

  for (size_t i = 0; i < Y_size; i++) {
    Y[i] = -1;
    Y_ref[i] = -1;
  }

  reference(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);

  T *dX, *dW, *dY;
  cudaMalloc((void **)&dX, X_bytes);
  cudaMalloc((void **)&dW, W_bytes);
  cudaMalloc((void **)&dY, Y_bytes);

  cudaMemcpy(dX, X, X_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W, W_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, Y, Y_bytes, cudaMemcpyHostToDevice);

  int W_grid = (Wout + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (Hout + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);
  printf("3D grid dimensions: N=%d M=%d Z=%d\n", N, M, Z);

  // try grid organizations
  dim3 grids_s1 (N, M, Z);
  dim3 grids_s2 (M, Z, N);
  dim3 grids_s3 (Z, N, M);
  dim3 blocks (TILE_WIDTH, TILE_WIDTH, 1);

  // shared memory holds one filter slice W[m] = C*K*K floats
  size_t shmem = (size_t)C * K * K * sizeof(float);

  // dispatch runtime K to a compile-time KK kernel (KK=0 = runtime fallback)
  #define LAUNCH(KERN, GRID) do { switch (K) { \
      case 1:  KERN<T,1 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      case 3:  KERN<T,3 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      case 5:  KERN<T,5 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      case 7:  KERN<T,7 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      case 9:  KERN<T,9 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      case 11: KERN<T,11><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
      default: KERN<T,0 ><<<GRID,blocks,shmem>>>(dX,dW,dY,C,M,K,Hin,Win,Hout,Wout,W_grid); break; \
    } } while (0)

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    LAUNCH(conv3d_s1, grids_s1);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    LAUNCH(conv3d_s2, grids_s2);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    LAUNCH(conv3d_s3, grids_s3);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);
  #undef LAUNCH

#ifdef CUDNN_CONV
  #include "conv3d_s4.cu"
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);
#endif

  free(X);
  free(W);
  free(Y);
  free(Y_ref);
  cudaFree(dX);
  cudaFree(dW);
  cudaFree(dY);
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat>\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int W = atoi(argv[4]);
  int H = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);

  printf("3D convolution (FP32)\n");
  printf("\n========== Warmup start ==========\n");
  conv3D<float>(N, C, M, W, H, K, 1000);
  printf("\n========== Warmup done ==========\n");
  conv3D<float>(N, C, M, W, H, K, repeat);

  return 0;
}
