// statistical kernel
//
// Profiling (job 952757) showed `reduce` is the #1 kernel (39% of GPU time,
// 3000 launches). The baseline used the slowest textbook reduction:
// interleaved addressing with a `(tx+1) % i == 0` modulo test -> highly
// divergent, strided shared accesses with bank conflicts, and only a few lanes
// active per step. This rewrite uses the standard sequential-addressing
// reduction (contiguous active lanes, no modulo, no bank conflicts), keeping the
// exact same multi-level load/store indexing so the host driver loop is
// unchanged. Inactive lanes (ei >= d_no) are zero-padded so the partial last
// block needs no special case.
__global__ void reduce(const  long d_Ne,  // number of elements in array
                    const int d_no,       // number of sums to reduce
                    const int d_mul,      // increment
                    fp *d_sums,           // pointer to partial sums variable (DEVICE GLOBAL MEMORY)
                    fp *d_sums2){

  int bx = blockIdx.x;                     // current block index
  int tx = threadIdx.x;                    // current thread index
  int ei = (bx*NUMBER_THREADS)+tx;         // unique thread id

  __shared__ fp d_psum[NUMBER_THREADS];
  __shared__ fp d_psum2[NUMBER_THREADS];

  // load (zero-pad the extra threads so the reduction needs no last-block case)
  if(ei < d_no){
    d_psum[tx]  = d_sums[ei*d_mul];
    d_psum2[tx] = d_sums2[ei*d_mul];
  }
  else{
    d_psum[tx]  = 0;
    d_psum2[tx] = 0;
  }
  __syncthreads();

  // sequential-addressing tree reduction
  for(int s = NUMBER_THREADS/2; s > 0; s >>= 1){
    if(tx < s){
      d_psum[tx]  += d_psum[tx + s];
      d_psum2[tx] += d_psum2[tx + s];
    }
    __syncthreads();
  }

  // block result stored in global memory (same slot as the baseline)
  if(tx == 0){
    d_sums[bx*d_mul*NUMBER_THREADS]  = d_psum[0];
    d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[0];
  }
}
