/*
  Reference
  Chapter 7 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define GPU_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define MAX_MASK_WIDTH 10
#define MAX_BLOCK_SIZE 1024

template<typename T>
__constant__ T mask [MAX_MASK_WIDTH];

// ---------------------------------------------------------------------------
// Optimized conv1d: keep the baseline's fully-coalesced one-element-per-thread
// access pattern (for tap j, thread i reads in[i-h+j]; addresses are contiguous
// across a warp and the overlap is served by L2), but remove the per-tap
// boundary branch for interior blocks and unroll the mask loop by templating on
// the (compile-time) mask width.
//
// Profiling (job 952532) showed double already saturates DRAM (~89% peak) while
// float (65%) and int16 (29%) under-saturate -- not a coalescing problem but a
// per-element instruction-overhead problem: the 2 boundary comparisons x
// mask_width per output dominate when few bytes are moved. Removing them on the
// interior fast path cuts that overhead without changing the memory pattern.
//
// (A 128-bit vectorized variant with per-thread scalar halo loads was tried and
//  REGRESSED badly -- the halo loads became stride-E uncoalesced; see analysis.)
// ---------------------------------------------------------------------------
template<typename T, int MW>
__global__
void conv1d(const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int start = i - MW / 2;
  T s = 0;
  if (start >= 0 && start + (MW - 1) < input_width) {
    // interior fast path: no bounds check, fully unrolled, coalesced loads
    #pragma unroll
    for (int j = 0; j < MW; j++) s += in[start + j] * mask<T>[j];
  } else {
    #pragma unroll
    for (int j = 0; j < MW; j++) {
      int idx = start + j;
      if (idx >= 0 && idx < input_width) s += in[idx] * mask<T>[j];
    }
  }
  out[i] = s;
}

// dispatch the runtime mask_width (3,5,7,9) to the templated kernel
template<typename T>
static void launch_conv1d(dim3 grids, dim3 blocks,
                          const T* d_a, T* d_b,
                          int input_width, int mask_width)
{
  switch (mask_width) {
    case 3: conv1d<T,3><<<grids,blocks>>>(d_a, d_b, input_width); break;
    case 5: conv1d<T,5><<<grids,blocks>>>(d_a, d_b, input_width); break;
    case 7: conv1d<T,7><<<grids,blocks>>>(d_a, d_b, input_width); break;
    case 9: conv1d<T,9><<<grids,blocks>>>(d_a, d_b, input_width); break;
  }
}

template<typename T>
__global__
void conv1d_tiled(const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  extern __shared__ unsigned char smem[]; // TILE_SIZE + MAX_MASK_WIDTH - 1;
  T *tile = reinterpret_cast<T*>(smem);
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  int n = mask_width / 2;  // last n cells of the previous tile

  // load left cells 
  int halo_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  if (threadIdx.x >= blockDim.x - n)
     tile[threadIdx.x - (blockDim.x - n)] = halo_left < 0 ? 0 : in[halo_left];

  // load center cells
  tile[n + threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];

  // load right cells
  int halo_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  if (threadIdx.x < n)
     tile[threadIdx.x + blockDim.x + n] = halo_right >= input_width ? 0 : in[halo_right];

  __syncthreads();

  T s = 0;
  for (int j = 0; j < mask_width; j++)
    s += tile[threadIdx.x + j] * mask<T>[j];

  out[i] = s;
}

template<typename T>
__global__
void conv1d_tiled_caching(const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  extern __shared__ unsigned char smem[]; // TILE_SIZE
  T *tile = reinterpret_cast<T*>(smem);

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  tile[threadIdx.x] = in[i];
  __syncthreads();

  int this_tile_start = blockIdx.x * blockDim.x;
  int next_tile_start = (blockIdx.x + 1) * blockDim.x;
  int start = i - (mask_width / 2);
  T s = 0;
  for (int j = 0; j < mask_width; j++) {
    int in_index = start + j;
    if (in_index >= 0 && in_index < input_width) {
      if (in_index >= this_tile_start && in_index < next_tile_start) {
        // in_index = (start + j) = (i - mask_width/2 +j) >= 0,
        // then map in_index to tile_index
        s += tile[threadIdx.x + j - (mask_width / 2)] * mask<T>[j];
      } else {
        s += in[in_index] * mask<T>[j];
      }
    }
  }
  out[i] = s;
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width)
{
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (fabs(s - d_out[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); // input
  b = (T *)malloc(size_bytes); // output

  T h_mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *d_a, *d_b;
  GPU_CHECK(cudaMalloc((void **)&d_a, size_bytes));
  GPU_CHECK(cudaMalloc((void **)&d_b, size_bytes));

  GPU_CHECK(cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpyToSymbol(mask<T>, h_mask, mask_width * sizeof(T)));

  GPU_CHECK(cudaDeviceSynchronize());

  // conv1D basic (one element per thread; coalesced; interior fast path)
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    dim3 grids (input_width / bs);
    dim3 blocks (bs);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      launch_conv1d(grids, blocks, d_a, d_b, input_width, mask_width);
    }
    GPU_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d kernel (block size %d): %f (us)\n",
           bs, (time * 1e-3f) / repeat);
    GPU_CHECK(cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost));
    reference(a, b, h_mask, input_width, mask_width);
  }

  // conv1D tiling
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    dim3 grids (input_width / bs);
    dim3 blocks (bs);
    size_t sm_bytes = (bs + MAX_MASK_WIDTH - 1) * sizeof(T);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      conv1d_tiled <<< grids, blocks, sm_bytes, 0 >>> (d_a, d_b, input_width, mask_width);
    }
    GPU_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled kernel (block size %d): %f (us)\n",
           bs, (time * 1e-3f) / repeat);
    GPU_CHECK(cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost));
    reference(a, b, h_mask, input_width, mask_width);
  }

  // conv1D tiling and caching
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    dim3 grids (input_width / bs);
    dim3 blocks (bs);
    size_t sm_bytes = bs * sizeof(T);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      conv1d_tiled_caching <<< grids, blocks, sm_bytes, 0 >>> (d_a, d_b, input_width, mask_width);
    }
    GPU_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled-caching kernel (block size %d): %f (us)\n",
           bs, (time * 1e-3f) / repeat);
    GPU_CHECK(cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost));
    reference(a, b, h_mask, input_width, mask_width);
  }

  free(a);
  free(b);
  GPU_CHECK(cudaFree(d_a));
  GPU_CHECK(cudaFree(d_b));
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  // a multiple of MAX BLOCK_SIZE
  input_width = (input_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;

  const int repeat = atoi(argv[2]);

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width); 

    printf("1D convolution (FP64)\n");
    conv1D<double>(input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(input_width, mask_width, repeat);
  }

  return 0;
}
