#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <chrono>
#include "reference.h"

// ---------------------------------------------------------------------------
// Optimized bilateral filter. Profiling (job 952714) shows compute-bound
// (DRAM ~1%, SM ~78%). Three changes vs baseline:
//   1. expf -> __expf (single MUFU instruction).
//   2. The spatial weight exp(-(i^2+j^2)/(2*var_s)) depends only on the window
//      offset (i,j) for interior pixels -> precompute into constant memory, so
//      the inner loop drops the spatial computation and one transcendental.
//   3. Interior fast path (no mirror-edge branches); boundary pixels (a thin
//      border, R<=9 on a 2960x1440 image) keep the exact mirror path.
// Exact except the __expf approximation, which still passes the 1e-3 check.
// ---------------------------------------------------------------------------
__constant__ float c_spatial[(2*9+1)*(2*9+1)];   // exp(spatial) table, max R=9

template<int R>
__global__ void bilateralFilter(
    const float *__restrict__ in,
    float *__restrict__ out,
    int w,
    int h,
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if(idx >= w || idy >= h) return;

  const int id = idy*w + idx;
  const float I = in[id];
  const float rI = -1.f / (2.f * variance_I);     // hoisted reciprocal
  constexpr int D = 2*R + 1;
  float res = 0, normalization = 0;

  if (idx >= R && idx < w-R && idy >= R && idy < h-R) {
    // interior fast path: no mirroring, spatial weight from constant table
    #pragma unroll
    for(int i = -R; i <= R; i++) {
      #pragma unroll
      for(int j = -R; j <= R; j++) {
        float I_w    = in[(idy+j)*w + (idx+i)];
        float diff   = I - I_w;
        float range  = diff*diff*rI;
        float weight = a_square * c_spatial[(i+R)*D + (j+R)] * __expf(range);
        normalization += weight;
        res += I_w * weight;
      }
    }
  } else {
    // boundary path: exact mirror-edge handling (matches the CPU reference)
    const float rS = -1.f / (2.f * variance_spatial);
    for(int i = -R; i <= R; i++) {
      for(int j = -R; j <= R; j++) {
        int idk = idx+i;
        int idl = idy+j;
        if( idk < 0) idk = -idk;
        if( idl < 0) idl = -idl;
        if( idk > w - 1) idk = w - 1 - i;
        if( idl > h - 1) idl = h - 1 - j;
        float I_w     = in[idl*w + idk];
        float diff    = I - I_w;
        float range   = diff*diff*rI;
        float spatial = ((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) * rS;
        float weight  = a_square * __expf(spatial + range);
        normalization += weight;
        res += I_w * weight;
      }
    }
  }
  out[id] = res/normalization;
}

// fill the constant spatial-weight table for radius R
template<int R>
static void setup_spatial(float variance_spatial)
{
  const int D = 2*R + 1;
  float tbl[(2*9+1)*(2*9+1)];
  for(int i = -R; i <= R; i++)
    for(int j = -R; j <= R; j++)
      tbl[(i+R)*D + (j+R)] = expf(-(float)(i*i + j*j) / (2.f * variance_spatial));
  cudaMemcpyToSymbol(c_spatial, tbl, sizeof(float) * D * D);
}

//
// reference https://en.wikipedia.org/wiki/Bilateral_filter
//
int main(int argc, char *argv[]) {

  if (argc != 6) {
    printf("Usage: %s <image width> <image height> <intensity> <spatial> <repeat>\n",
            argv[0]);
    return 1;
  }

  // image dimensions
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);
  const int img_size = w*h;

   // As the range parameter increases, the bilateral filter gradually 
   // approaches Gaussian convolution more closely because the range 
   // Gaussian widens and flattens, which means that it becomes nearly
   // constant over the intensity interval of the image.
  float variance_I = atof(argv[3]);

   // As the spatial parameter increases, the larger features get smoothened.
  float variance_spatial = atof(argv[4]);

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  int repeat = atoi(argv[5]);

  float *d_src, *d_dst;
  cudaMalloc((void**)&d_dst, img_size * sizeof(float));
  cudaMalloc((void**)&d_src, img_size * sizeof(float));

  float *h_src = (float*) malloc (img_size * sizeof(float));
  // host and device results
  float *h_dst = (float*) malloc (img_size * sizeof(float));
  float *r_dst = (float*) malloc (img_size * sizeof(float));

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

  cudaMemcpy(d_src, h_src, img_size * sizeof(float), cudaMemcpyHostToDevice); 

  dim3 threads (16, 16);
  dim3 blocks ((w+15)/16, (h+15)/16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  setup_spatial<3>(variance_spatial);
  for (int i = 0; i < repeat; i++)
    bilateralFilter<3><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (3x3) %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_dst, d_dst, img_size * sizeof(float), cudaMemcpyDeviceToHost); 

  // verify
  bool ok = true;
  reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  setup_spatial<6>(variance_spatial);
  for (int i = 0; i < repeat; i++)
    bilateralFilter<6><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (6x6) %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_dst, d_dst, img_size * sizeof(float), cudaMemcpyDeviceToHost); 

  reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  setup_spatial<9>(variance_spatial);
  for (int i = 0; i < repeat; i++)
    bilateralFilter<9><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (9x9) %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_dst, d_dst, img_size * sizeof(float), cudaMemcpyDeviceToHost); 

  reference<9>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_dst);
  free(r_dst);
  free(h_src);
  cudaFree(d_dst);
  cudaFree(d_src);
  return 0;
}
