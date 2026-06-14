/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    __global__ void histogram_smem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
    {
        // global position and size
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int nx = blockDim.x * gridDim.x;
        int ny = blockDim.y * gridDim.y;

        // threads in workgroup
        int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
        int nt = blockDim.x * blockDim.y; // total threads in workgroup

        // group index in 0..ngroups-1
        int g = blockIdx.x + blockIdx.y * gridDim.x;

        // initialize smem
        __shared__ unsigned int smem[ACTIVE_CHANNELS * NUM_BINS + 3];
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
            smem[i] = 0;
        __syncthreads();

        // process pixels
        // updates our group's partial histogram in smem
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                    atomicAdd(&smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL], 1);
            }
        }

        __syncthreads();

        // move to our workgroup's slice of output
        out += g * NUM_PARTS;

        // store local output to global
        for (int i = t; i < NUM_BINS; i += nt)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    __global__ void histogram_smem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range
        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];
        out[i] = total;
    }



template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_smem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist, 
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);

    // The baseline fixed grid(16,16) = 256 blocks leaves the GPU at ~20%
    // occupancy (ncu): block is 128 threads, V100 holds up to 16 blocks/SM x 80
    // SMs = 1280 blocks, so 256 fills only a fraction of the SMs. Size the grid
    // to the kernel's max active blocks per SM so kernel1 fills the device. More
    // partial histograms slightly grows the accum kernel's reduction loop, but
    // accum is tiny relative to the pixel-scanning kernel.
    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        histogram_smem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>,
        block.x * block.y, /*dynamic smem*/0);
    if (blocksPerSM < 1) blocksPerSM = 1;
    int target = blocksPerSM * props.multiProcessorCount;   // fill all SM slots
    // keep it a 2D grid similar in shape to the original (16 wide)
    dim3 grid(16, (target + 15) / 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block2.x - 1) / block2.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    histogram_smem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist);

    histogram_smem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_part_hist);

    return elapsed_millis;
}

