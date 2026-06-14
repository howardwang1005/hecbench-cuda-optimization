# gaussian-cuda — Optimization v2 (shrinking grid)

## v1 (FAILED): fan1+fan2 fusion
Folding fan1 into fan2 made every column thread recompute a[row][t]/a[t][t]
(~4096x redundant divisions). Measured ~0.85x (slower) vs same-compiler baseline.
Reverted.

## v2 (current): shrink the launch grid per elimination step
The kernels are UNCHANGED (byte-identical to baseline). Only the launch grid
changes: at step t the live submatrix is (size-1-t) rows x (size-t) cols, but
the baseline launched ceil(size/16)^2 blocks every step. Late steps scheduled
~256x256 blocks that immediately returned on the internal bounds check. We size
the grid to the live submatrix each iteration:

    rows = size-1-t;  cols = size-t;
    gridDim_fan1 = ceil(rows/256);
    gridDim_fan2 = (ceil(cols/16), ceil(rows/16));

No algorithm change, no extra work, no fusion -> identical result, fewer idle
blocks scheduled. Correctness preserved by construction (kernels' own guards
still hold; we only stop launching blocks that would have returned immediately).

Verify with the program's built-in check (PASS) and compare Total kernel
execution time vs baseline on the same compiler (CUDA 12.3).
