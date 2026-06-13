# Baseline Metric Overview

Primary metrics are the main optimization targets. Supporting metrics provide end-to-end context.

## all-pairs-distance-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `k1_register_atomic` | lower | 0.188743 | ms | 5 |
| primary | `k2_shared_tree_reduction` | lower | 0.159191 | ms | 5 |
| primary | `k3_cub_block_reduction` | lower | 0.157361 | ms | 5 |
| secondary | `cpu_reference_time` | lower | 119.095 | ms | 5 |

## atomicReduction-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `atomic_reduction.block_1024` | higher | 770.511 | GB/s | 5 |
| primary | `atomic_reduction.block_128` | higher | 690.845 | GB/s | 5 |
| primary | `atomic_reduction.block_256` | higher | 702.97 | GB/s | 5 |
| primary | `atomic_reduction.block_512` | higher | 752.64 | GB/s | 5 |
| primary | `atomic_reduction_v16.block_1024` | higher | 588.284 | GB/s | 5 |
| primary | `atomic_reduction_v16.block_128` | higher | 722.35 | GB/s | 5 |
| primary | `atomic_reduction_v16.block_256` | higher | 728.311 | GB/s | 5 |
| primary | `atomic_reduction_v16.block_512` | higher | 686.959 | GB/s | 5 |
| primary | `atomic_reduction_v2.block_1024` | higher | 848.954 | GB/s | 5 |
| primary | `atomic_reduction_v2.block_128` | higher | 845.385 | GB/s | 5 |
| primary | `atomic_reduction_v2.block_256` | higher | 837.293 | GB/s | 5 |
| primary | `atomic_reduction_v2.block_512` | higher | 825.838 | GB/s | 5 |
| primary | `atomic_reduction_v4.block_1024` | higher | 848.148 | GB/s | 5 |
| primary | `atomic_reduction_v4.block_128` | higher | 850.581 | GB/s | 5 |
| primary | `atomic_reduction_v4.block_256` | higher | 863.378 | GB/s | 5 |
| primary | `atomic_reduction_v4.block_512` | higher | 850.928 | GB/s | 5 |
| primary | `atomic_reduction_v8.block_1024` | higher | 827.979 | GB/s | 5 |
| primary | `atomic_reduction_v8.block_128` | higher | 834.299 | GB/s | 5 |
| primary | `atomic_reduction_v8.block_256` | higher | 846.104 | GB/s | 5 |
| primary | `atomic_reduction_v8.block_512` | higher | 787.501 | GB/s | 5 |

## bfs-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `total_kernel_execution_time` | lower | 1.00126 | ms | 5 |

## bilateral-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time.3x3` | lower | 0.920257 | ms | 5 |
| primary | `average_kernel_execution_time.6x6` | lower | 3.4139 | ms | 5 |
| primary | `average_kernel_execution_time.9x9` | lower | 7.27637 | ms | 5 |

## bscan-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `block_1024.execution_time` | lower | 0.676225 | ms | 5 |
| primary | `block_128.execution_time` | lower | 0.341159 | ms | 5 |
| primary | `block_256.execution_time` | lower | 0.676147 | ms | 5 |
| primary | `block_32.execution_time` | lower | 0.173924 | ms | 5 |
| primary | `block_512.execution_time` | lower | 0.675997 | ms | 5 |
| primary | `block_64.execution_time` | lower | 0.341103 | ms | 5 |
| secondary | `block_1024.throughput` | higher | 91.5841 | GElements/s | 5 |
| secondary | `block_128.throughput` | higher | 22.6916 | GElements/s | 5 |
| secondary | `block_256.throughput` | higher | 22.8987 | GElements/s | 5 |
| secondary | `block_32.throughput` | higher | 11.1276 | GElements/s | 5 |
| secondary | `block_512.throughput` | higher | 45.8076 | GElements/s | 5 |
| secondary | `block_64.throughput` | higher | 11.3477 | GElements/s | 5 |

## convolution1D-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `mask_3.fp32.conv1d-tiled-caching.block_1024` | lower | 1.63723 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled-caching.block_128` | lower | 1.72592 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled-caching.block_256` | lower | 1.64216 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled-caching.block_512` | lower | 1.66231 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled-caching.block_64` | lower | 2.96832 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled.block_1024` | lower | 1.70651 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled.block_128` | lower | 1.74885 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled.block_256` | lower | 1.68037 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled.block_512` | lower | 1.67275 | ms | 5 |
| primary | `mask_3.fp32.conv1d-tiled.block_64` | lower | 2.96914 | ms | 5 |
| primary | `mask_3.fp32.conv1d.block_1024` | lower | 1.43418 | ms | 5 |
| primary | `mask_3.fp32.conv1d.block_128` | lower | 1.50545 | ms | 5 |
| primary | `mask_3.fp32.conv1d.block_256` | lower | 1.41965 | ms | 5 |
| primary | `mask_3.fp32.conv1d.block_512` | lower | 1.4242 | ms | 5 |
| primary | `mask_3.fp32.conv1d.block_64` | lower | 2.96794 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled-caching.block_1024` | lower | 2.73437 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled-caching.block_128` | lower | 2.70362 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled-caching.block_256` | lower | 2.70752 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled-caching.block_512` | lower | 2.72428 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled-caching.block_64` | lower | 2.98687 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled.block_1024` | lower | 2.80975 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled.block_128` | lower | 2.77428 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled.block_256` | lower | 2.79692 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled.block_512` | lower | 2.80392 | ms | 5 |
| primary | `mask_3.fp64.conv1d-tiled.block_64` | lower | 2.98422 | ms | 5 |
| primary | `mask_3.fp64.conv1d.block_1024` | lower | 2.71165 | ms | 5 |
| primary | `mask_3.fp64.conv1d.block_128` | lower | 2.67924 | ms | 5 |
| primary | `mask_3.fp64.conv1d.block_256` | lower | 2.6834 | ms | 5 |
| primary | `mask_3.fp64.conv1d.block_512` | lower | 2.69142 | ms | 5 |
| primary | `mask_3.fp64.conv1d.block_64` | lower | 2.98292 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled-caching.block_1024` | lower | 1.4703 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled-caching.block_128` | lower | 1.49256 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled-caching.block_256` | lower | 1.34578 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled-caching.block_512` | lower | 1.39396 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled-caching.block_64` | lower | 2.96753 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled.block_1024` | lower | 1.31504 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled.block_128` | lower | 1.49715 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled.block_256` | lower | 1.3405 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled.block_512` | lower | 1.29221 | ms | 5 |
| primary | `mask_3.int16.conv1d-tiled.block_64` | lower | 2.96675 | ms | 5 |
| primary | `mask_3.int16.conv1d.block_1024` | lower | 1.11927 | ms | 5 |
| primary | `mask_3.int16.conv1d.block_128` | lower | 1.48753 | ms | 5 |
| primary | `mask_3.int16.conv1d.block_256` | lower | 1.13021 | ms | 5 |
| primary | `mask_3.int16.conv1d.block_512` | lower | 1.1003 | ms | 5 |
| primary | `mask_3.int16.conv1d.block_64` | lower | 2.96595 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled-caching.block_1024` | lower | 1.84236 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled-caching.block_128` | lower | 1.86963 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled-caching.block_256` | lower | 1.77008 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled-caching.block_512` | lower | 1.7811 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled-caching.block_64` | lower | 2.97027 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled.block_1024` | lower | 1.80987 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled.block_128` | lower | 1.80743 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled.block_256` | lower | 1.75653 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled.block_512` | lower | 1.76518 | ms | 5 |
| primary | `mask_5.fp32.conv1d-tiled.block_64` | lower | 2.97046 | ms | 5 |
| primary | `mask_5.fp32.conv1d.block_1024` | lower | 1.45039 | ms | 5 |
| primary | `mask_5.fp32.conv1d.block_128` | lower | 1.48866 | ms | 5 |
| primary | `mask_5.fp32.conv1d.block_256` | lower | 1.41435 | ms | 5 |
| primary | `mask_5.fp32.conv1d.block_512` | lower | 1.41161 | ms | 5 |
| primary | `mask_5.fp32.conv1d.block_64` | lower | 2.9886 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled-caching.block_1024` | lower | 2.75313 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled-caching.block_128` | lower | 2.72642 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled-caching.block_256` | lower | 2.72439 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled-caching.block_512` | lower | 2.73505 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled-caching.block_64` | lower | 3.00189 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled.block_1024` | lower | 2.91666 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled.block_128` | lower | 2.83392 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled.block_256` | lower | 2.85935 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled.block_512` | lower | 2.88339 | ms | 5 |
| primary | `mask_5.fp64.conv1d-tiled.block_64` | lower | 2.98951 | ms | 5 |
| primary | `mask_5.fp64.conv1d.block_1024` | lower | 2.71075 | ms | 5 |
| primary | `mask_5.fp64.conv1d.block_128` | lower | 2.68451 | ms | 5 |
| primary | `mask_5.fp64.conv1d.block_256` | lower | 2.68453 | ms | 5 |
| primary | `mask_5.fp64.conv1d.block_512` | lower | 2.688 | ms | 5 |
| primary | `mask_5.fp64.conv1d.block_64` | lower | 2.98566 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled-caching.block_1024` | lower | 1.74867 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled-caching.block_128` | lower | 1.59263 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled-caching.block_256` | lower | 1.56917 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled-caching.block_512` | lower | 1.61774 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled-caching.block_64` | lower | 2.96702 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled.block_1024` | lower | 1.49867 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled.block_128` | lower | 1.56222 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled.block_256` | lower | 1.50061 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled.block_512` | lower | 1.47251 | ms | 5 |
| primary | `mask_5.int16.conv1d-tiled.block_64` | lower | 2.96725 | ms | 5 |
| primary | `mask_5.int16.conv1d.block_1024` | lower | 1.28578 | ms | 5 |
| primary | `mask_5.int16.conv1d.block_128` | lower | 1.48624 | ms | 5 |
| primary | `mask_5.int16.conv1d.block_256` | lower | 1.1661 | ms | 5 |
| primary | `mask_5.int16.conv1d.block_512` | lower | 1.16853 | ms | 5 |
| primary | `mask_5.int16.conv1d.block_64` | lower | 2.96755 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled-caching.block_1024` | lower | 2.13055 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled-caching.block_128` | lower | 2.06262 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled-caching.block_256` | lower | 1.97788 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled-caching.block_512` | lower | 2.00901 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled-caching.block_64` | lower | 2.97994 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled.block_1024` | lower | 1.9301 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled.block_128` | lower | 1.89902 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled.block_256` | lower | 1.88629 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled.block_512` | lower | 1.9068 | ms | 5 |
| primary | `mask_7.fp32.conv1d-tiled.block_64` | lower | 2.97419 | ms | 5 |
| primary | `mask_7.fp32.conv1d.block_1024` | lower | 1.58043 | ms | 5 |
| primary | `mask_7.fp32.conv1d.block_128` | lower | 1.567 | ms | 5 |
| primary | `mask_7.fp32.conv1d.block_256` | lower | 1.52393 | ms | 5 |
| primary | `mask_7.fp32.conv1d.block_512` | lower | 1.53706 | ms | 5 |
| primary | `mask_7.fp32.conv1d.block_64` | lower | 2.99227 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled-caching.block_1024` | lower | 2.80459 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled-caching.block_128` | lower | 2.76965 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled-caching.block_256` | lower | 2.76328 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled-caching.block_512` | lower | 2.77868 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled-caching.block_64` | lower | 3.01953 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled.block_1024` | lower | 3.03168 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled.block_128` | lower | 2.90476 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled.block_256` | lower | 2.96198 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled.block_512` | lower | 2.99763 | ms | 5 |
| primary | `mask_7.fp64.conv1d-tiled.block_64` | lower | 2.99534 | ms | 5 |
| primary | `mask_7.fp64.conv1d.block_1024` | lower | 2.71086 | ms | 5 |
| primary | `mask_7.fp64.conv1d.block_128` | lower | 2.70002 | ms | 5 |
| primary | `mask_7.fp64.conv1d.block_256` | lower | 2.70278 | ms | 5 |
| primary | `mask_7.fp64.conv1d.block_512` | lower | 2.71141 | ms | 5 |
| primary | `mask_7.fp64.conv1d.block_64` | lower | 2.99877 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled-caching.block_1024` | lower | 2.10413 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled-caching.block_128` | lower | 1.99317 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled-caching.block_256` | lower | 1.96044 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled-caching.block_512` | lower | 1.97588 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled-caching.block_64` | lower | 2.96609 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled.block_1024` | lower | 1.708 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled.block_128` | lower | 1.77653 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled.block_256` | lower | 1.71139 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled.block_512` | lower | 1.68409 | ms | 5 |
| primary | `mask_7.int16.conv1d-tiled.block_64` | lower | 2.96742 | ms | 5 |
| primary | `mask_7.int16.conv1d.block_1024` | lower | 1.6193 | ms | 5 |
| primary | `mask_7.int16.conv1d.block_128` | lower | 1.57358 | ms | 5 |
| primary | `mask_7.int16.conv1d.block_256` | lower | 1.52979 | ms | 5 |
| primary | `mask_7.int16.conv1d.block_512` | lower | 1.52201 | ms | 5 |
| primary | `mask_7.int16.conv1d.block_64` | lower | 2.99911 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled-caching.block_1024` | lower | 2.28077 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled-caching.block_128` | lower | 2.20487 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled-caching.block_256` | lower | 2.1246 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled-caching.block_512` | lower | 2.16375 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled-caching.block_64` | lower | 2.97429 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled.block_1024` | lower | 1.91128 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled.block_128` | lower | 1.84446 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled.block_256` | lower | 1.80977 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled.block_512` | lower | 1.82517 | ms | 5 |
| primary | `mask_9.fp32.conv1d-tiled.block_64` | lower | 2.97322 | ms | 5 |
| primary | `mask_9.fp32.conv1d.block_1024` | lower | 1.62577 | ms | 5 |
| primary | `mask_9.fp32.conv1d.block_128` | lower | 1.58147 | ms | 5 |
| primary | `mask_9.fp32.conv1d.block_256` | lower | 1.54776 | ms | 5 |
| primary | `mask_9.fp32.conv1d.block_512` | lower | 1.57839 | ms | 5 |
| primary | `mask_9.fp32.conv1d.block_64` | lower | 3.00325 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled-caching.block_1024` | lower | 2.86443 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled-caching.block_128` | lower | 2.81655 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled-caching.block_256` | lower | 2.80773 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled-caching.block_512` | lower | 2.82744 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled-caching.block_64` | lower | 3.04236 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled.block_1024` | lower | 3.0805 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled.block_128` | lower | 2.91671 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled.block_256` | lower | 2.98225 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled.block_512` | lower | 3.04107 | ms | 5 |
| primary | `mask_9.fp64.conv1d-tiled.block_64` | lower | 2.99596 | ms | 5 |
| primary | `mask_9.fp64.conv1d.block_1024` | lower | 2.71966 | ms | 5 |
| primary | `mask_9.fp64.conv1d.block_128` | lower | 2.70443 | ms | 5 |
| primary | `mask_9.fp64.conv1d.block_256` | lower | 2.70602 | ms | 5 |
| primary | `mask_9.fp64.conv1d.block_512` | lower | 2.71746 | ms | 5 |
| primary | `mask_9.fp64.conv1d.block_64` | lower | 3.0061 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled-caching.block_1024` | lower | 2.21764 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled-caching.block_128` | lower | 2.21418 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled-caching.block_256` | lower | 2.16969 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled-caching.block_512` | lower | 2.17727 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled-caching.block_64` | lower | 2.96569 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled.block_1024` | lower | 1.67468 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled.block_128` | lower | 1.73694 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled.block_256` | lower | 1.67214 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled.block_512` | lower | 1.65748 | ms | 5 |
| primary | `mask_9.int16.conv1d-tiled.block_64` | lower | 2.96833 | ms | 5 |
| primary | `mask_9.int16.conv1d.block_1024` | lower | 1.83237 | ms | 5 |
| primary | `mask_9.int16.conv1d.block_128` | lower | 1.72662 | ms | 5 |
| primary | `mask_9.int16.conv1d.block_256` | lower | 1.68898 | ms | 5 |
| primary | `mask_9.int16.conv1d.block_512` | lower | 1.66876 | ms | 5 |
| primary | `mask_9.int16.conv1d.block_64` | lower | 3.00608 | ms | 5 |

## convolution3D-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `c6_w14_h14.conv3d_s1` | lower | 0.0176334 | ms | 5 |
| primary | `c6_w14_h14.conv3d_s2` | lower | 0.0192304 | ms | 5 |
| primary | `c6_w14_h14.conv3d_s3` | lower | 0.0175847 | ms | 5 |
| primary | `c96_w26_h26.conv3d_s1` | lower | 13.6125 | ms | 5 |
| primary | `c96_w26_h26.conv3d_s2` | lower | 13.1328 | ms | 5 |
| primary | `c96_w26_h26.conv3d_s3` | lower | 14.0959 | ms | 5 |

## convolutionSeparable-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 1.526 | ms | 5 |

## fdtd3d-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 0.191 | ms | 5 |

## filter-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `filter.global_aggregate` | lower | 2.46262 | ms | 5 |
| primary | `filter.shared_memory` | lower | 0.958458 | ms | 5 |

## floydwarshall-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 8.507 | ms | 5 |

## gaussian-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `total_kernel_execution_time` | lower | 490.962 | ms | 5 |
| secondary | `device_offloading_time` | lower | 756.357 | ms | 5 |

## heat-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `solve_time` | lower | 340.85 | ms | 5 |
| secondary | `bandwidth` | higher | 787.547 | GB/s | 5 |
| secondary | `total_time` | lower | 1007.16 | ms | 5 |

## histogram-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `1_channel_uchar1.global_memory_atomics` | lower | 0.071309 | ms | 5 |
| primary | `1_channel_uchar1.shared_memory_atomics` | lower | 0.035929 | ms | 5 |
| primary | `3/4_channel_float4.global_memory_atomics` | lower | 0.189893 | ms | 5 |
| primary | `3/4_channel_float4.shared_memory_atomics` | lower | 0.082673 | ms | 5 |
| primary | `3/4_channel_uchar4.global_memory_atomics` | lower | 0.186102 | ms | 5 |
| primary | `3/4_channel_uchar4.shared_memory_atomics` | lower | 0.065776 | ms | 5 |

## hotspot-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `total_kernel_execution_time` | lower | 1.183 | ms | 5 |
| secondary | `device_offloading_time` | lower | 188 | ms | 5 |

## hotspot3D-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 0.0428491 | ms | 5 |
| secondary | `device_offloading_time` | lower | 393 | ms | 5 |

## jaccard-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `unweighted_jaccard_pipeline` | lower | 5.61527 | ms | 5 |
| primary | `weighted_jaccard_pipeline` | lower | 6.67691 | ms | 5 |

## jacobi-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_execution_time_per_iteration` | lower | 0.0738951 | ms | 5 |
| secondary | `total_elapsed_time` | lower | 637 | ms | 5 |

## laplace3d-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `grid_128x128x128.kernel_time` | lower | 0.114 | ms | 5 |
| primary | `grid_512x512x512.kernel_time` | lower | 2.16 | ms | 5 |

## lavaMD-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `kernel_execution_time` | lower | 29.117 | ms | 5 |
| secondary | `device_offloading_time` | lower | 211.141 | ms | 5 |

## miniWeather-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `main_time_step_loop` | lower | 1202.35 | ms | 5 |

## nbody-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `total_time` | lower | 27.7052 | ms | 5 |
| secondary | `average_performance` | higher | 1956.83 | GFLOP/s | 5 |

## nw-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `total_kernel_execution_time` | lower | 21.998 | ms | 5 |

## scan-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `block_1024.bytes_1.with_conflicts` | lower | 3.85924 | ms | 5 |
| primary | `block_1024.bytes_1.without_conflicts` | lower | 3.98011 | ms | 5 |
| primary | `block_1024.bytes_2.with_conflicts` | lower | 4.18532 | ms | 5 |
| primary | `block_1024.bytes_2.without_conflicts` | lower | 4.15986 | ms | 5 |
| primary | `block_1024.bytes_4.with_conflicts` | lower | 5.60482 | ms | 5 |
| primary | `block_1024.bytes_4.without_conflicts` | lower | 4.01445 | ms | 5 |
| primary | `block_1024.bytes_8.with_conflicts` | lower | 8.28242 | ms | 5 |
| primary | `block_1024.bytes_8.without_conflicts` | lower | 6.58818 | ms | 5 |
| primary | `block_128.bytes_1.with_conflicts` | lower | 3.09003 | ms | 5 |
| primary | `block_128.bytes_1.without_conflicts` | lower | 3.52071 | ms | 5 |
| primary | `block_128.bytes_2.with_conflicts` | lower | 3.64822 | ms | 5 |
| primary | `block_128.bytes_2.without_conflicts` | lower | 3.71027 | ms | 5 |
| primary | `block_128.bytes_4.with_conflicts` | lower | 4.89144 | ms | 5 |
| primary | `block_128.bytes_4.without_conflicts` | lower | 3.51966 | ms | 5 |
| primary | `block_128.bytes_8.with_conflicts` | lower | 8.37233 | ms | 5 |
| primary | `block_128.bytes_8.without_conflicts` | lower | 6.13065 | ms | 5 |
| primary | `block_2048.bytes_1.with_conflicts` | lower | 4.03148 | ms | 5 |
| primary | `block_2048.bytes_1.without_conflicts` | lower | 5.13996 | ms | 5 |
| primary | `block_2048.bytes_2.with_conflicts` | lower | 4.68511 | ms | 5 |
| primary | `block_2048.bytes_2.without_conflicts` | lower | 5.39885 | ms | 5 |
| primary | `block_2048.bytes_4.with_conflicts` | lower | 6.11823 | ms | 5 |
| primary | `block_2048.bytes_4.without_conflicts` | lower | 5.61678 | ms | 5 |
| primary | `block_2048.bytes_8.with_conflicts` | lower | 8.84487 | ms | 5 |
| primary | `block_2048.bytes_8.without_conflicts` | lower | 8.00459 | ms | 5 |
| primary | `block_256.bytes_1.with_conflicts` | lower | 3.19858 | ms | 5 |
| primary | `block_256.bytes_1.without_conflicts` | lower | 3.36364 | ms | 5 |
| primary | `block_256.bytes_2.with_conflicts` | lower | 3.77102 | ms | 5 |
| primary | `block_256.bytes_2.without_conflicts` | lower | 3.5347 | ms | 5 |
| primary | `block_256.bytes_4.with_conflicts` | lower | 5.09006 | ms | 5 |
| primary | `block_256.bytes_4.without_conflicts` | lower | 3.25242 | ms | 5 |
| primary | `block_256.bytes_8.with_conflicts` | lower | 8.35323 | ms | 5 |
| primary | `block_256.bytes_8.without_conflicts` | lower | 8.02414 | ms | 5 |
| primary | `block_512.bytes_1.with_conflicts` | lower | 3.36007 | ms | 5 |
| primary | `block_512.bytes_1.without_conflicts` | lower | 3.82723 | ms | 5 |
| primary | `block_512.bytes_2.with_conflicts` | lower | 4.2157 | ms | 5 |
| primary | `block_512.bytes_2.without_conflicts` | lower | 3.81644 | ms | 5 |
| primary | `block_512.bytes_4.with_conflicts` | lower | 5.22583 | ms | 5 |
| primary | `block_512.bytes_4.without_conflicts` | lower | 3.71597 | ms | 5 |
| primary | `block_512.bytes_8.with_conflicts` | lower | 8.3592 | ms | 5 |
| primary | `block_512.bytes_8.without_conflicts` | lower | 6.35859 | ms | 5 |

## snake-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `threshold_0.kernel_time` | lower | 0.005671 | ms | 5 |
| primary | `threshold_1.kernel_time` | lower | 0.008832 | ms | 5 |
| primary | `threshold_10.kernel_time` | lower | 0.122234 | ms | 5 |
| primary | `threshold_11.kernel_time` | lower | 0.145828 | ms | 5 |
| primary | `threshold_12.kernel_time` | lower | 0.170834 | ms | 5 |
| primary | `threshold_13.kernel_time` | lower | 0.196347 | ms | 5 |
| primary | `threshold_14.kernel_time` | lower | 0.234456 | ms | 5 |
| primary | `threshold_15.kernel_time` | lower | 0.26641 | ms | 5 |
| primary | `threshold_16.kernel_time` | lower | 0.305142 | ms | 5 |
| primary | `threshold_17.kernel_time` | lower | 0.348417 | ms | 5 |
| primary | `threshold_18.kernel_time` | lower | 0.394466 | ms | 5 |
| primary | `threshold_19.kernel_time` | lower | 0.435662 | ms | 5 |
| primary | `threshold_2.kernel_time` | lower | 0.013177 | ms | 5 |
| primary | `threshold_20.kernel_time` | lower | 0.485145 | ms | 5 |
| primary | `threshold_21.kernel_time` | lower | 0.540868 | ms | 5 |
| primary | `threshold_22.kernel_time` | lower | 0.575705 | ms | 5 |
| primary | `threshold_23.kernel_time` | lower | 0.686878 | ms | 5 |
| primary | `threshold_24.kernel_time` | lower | 0.767515 | ms | 5 |
| primary | `threshold_25.kernel_time` | lower | 0.823347 | ms | 5 |
| primary | `threshold_3.kernel_time` | lower | 0.021412 | ms | 5 |
| primary | `threshold_4.kernel_time` | lower | 0.035715 | ms | 5 |
| primary | `threshold_5.kernel_time` | lower | 0.044455 | ms | 5 |
| primary | `threshold_6.kernel_time` | lower | 0.059397 | ms | 5 |
| primary | `threshold_7.kernel_time` | lower | 0.07276 | ms | 5 |
| primary | `threshold_8.kernel_time` | lower | 0.089613 | ms | 5 |
| primary | `threshold_9.kernel_time` | lower | 0.106205 | ms | 5 |

## sobel-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 0.00796448 | ms | 5 |

## srad-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `compute_time` | lower | 56.163 | ms | 5 |
| secondary | `total_time` | lower | 197.562 | ms | 5 |

## stencil3d-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 36.017 | ms | 5 |

## thomas-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `average_kernel_execution_time` | lower | 2.35223 | ms | 5 |

## xsbench-cuda

| Role | Metric | Direction | Median | Unit | Runs |
|---|---|---|---:|---|---:|
| primary | `kernel_execution_time` | lower | 342 | ms | 5 |
| secondary | `kernel_lookups_per_second` | higher | 4.96782e+07 | lookups/s | 5 |
| secondary | `runtime` | lower | 56725 | ms | 5 |
| secondary | `total_lookups_per_second` | higher | 299689 | lookups/s | 5 |

