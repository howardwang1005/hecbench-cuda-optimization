// Optimized O(N^2) force kernel. Profiling (job 952721) showed the baseline is
// occupancy-limited (12.4% -> ~1 block/SM) from copying the whole 40-byte
// Particle into registers, while DRAM is idle (L2 hit 98.8%). We stage each tile
// of (pos,mass) as a coalesced float4 in shared memory and keep only pos+mass
// per thread -> far fewer registers -> higher occupancy to hide the rsqrt
// latency, and p[j] reads become shared-memory reads. Math is identical to the
// baseline (self- and padding-particles contribute 0 via mass=0 / dx=0).
__global__ void
accelerate_particles( Particle* p, const int n, const float kSofteningSquared, const float kG )
{
  extern __shared__ float4 tile[];               // blockDim.x entries: (x,y,z,mass)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load this thread's particle position (guard i>=n; result not stored back)
  float px = 0.f, py = 0.f, pz = 0.f;
  if (i < n) { px = p[i].pos[0]; py = p[i].pos[1]; pz = p[i].pos[2]; }

  RealType acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;

  for (int base = 0; base < n; base += blockDim.x) {
    int j = base + threadIdx.x;
    if (j < n) {
      tile[threadIdx.x] = make_float4(p[j].pos[0], p[j].pos[1], p[j].pos[2], p[j].mass);
    } else {
      tile[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);  // mass 0 -> no contribution
    }
    __syncthreads();

    int cnt = min((int)blockDim.x, n - base);
    #pragma unroll 4
    for (int k = 0; k < cnt; k++) {
      float4 pj = tile[k];
      RealType dx = pj.x - px;
      RealType dy = pj.y - py;
      RealType dz = pj.z - pz;
      RealType distance_sqr = dx*dx + dy*dy + dz*dz + kSofteningSquared;
      RealType distance_inv = rsqrtf(distance_sqr);
      RealType strength = kG * pj.w * distance_inv * distance_inv * distance_inv;
      acc0 += dx * strength;
      acc1 += dy * strength;
      acc2 += dz * strength;
    }
    __syncthreads();
  }

  if (i < n) {
    p[i].acc[0] = acc0;     // baseline acc is 0 at step start (reset in update)
    p[i].acc[1] = acc1;
    p[i].acc[2] = acc2;
  }
}

__global__ void
update_particles(Particle *__restrict__ p,
                 RealType *__restrict__ e,
                 const int n, RealType dt)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  auto pi = p[i];

  pi.vel[0] += pi.acc[0] * dt;  // 2flops
  pi.vel[1] += pi.acc[1] * dt;  // 2flops
  pi.vel[2] += pi.acc[2] * dt;  // 2flops

  pi.pos[0] += pi.vel[0] * dt;  // 2flops
  pi.pos[1] += pi.vel[1] * dt;  // 2flops
  pi.pos[2] += pi.vel[2] * dt;  // 2flops

  pi.acc[0] = 0.f;
  pi.acc[1] = 0.f;
  pi.acc[2] = 0.f;

  e[i] = pi.mass *
    (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
     pi.vel[2] * pi.vel[2]);  // 7flops

  p[i] = pi;
}

// Parallel reduction of e[0..n-1] into e[0] (baseline did this serially on a
// single thread, which profiling showed cost ~17% of GPU time). Launch with one
// block of BLOCK threads. All elements are read before e[0] is overwritten.
__global__ void
accumulate_energy(RealType *e, const int n)
{
  __shared__ RealType s[256];
  int t = threadIdx.x;
  RealType sum = 0;
  for (int i = t; i < n; i += blockDim.x) sum += e[i];
  s[t] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) e[0] = s[0];
}
