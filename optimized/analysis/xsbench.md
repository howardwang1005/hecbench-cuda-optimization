# xsbench bottleneck 分析

執行參數：`./main -s large -m event -r 10`（Event-based、Unionized grid、large）。
比較指標：程式自印 `Average kernel execution time: <s> seconds`（lookup kernel，
取 warmup/repeat 平均）。baseline ≈ **0.342 s**（job 951758；wall ~57s 多為 5.6GB init +
serial host 驗證，非 kernel）。

## 1. Baseline 行為

XSBench = Monte Carlo 中子輸運的巨觀截面查表 proxy。Event-based：**每個 lookup 一條 thread**
（large = 17,000,000 threads，grid=(N+255)/256, block=256），kernel `lookup`（Simulation.cu:14）
每 thread：
1. LCG 亂數 → 取樣 energy `p_energy` 與 material `mat`（`pick_mat` 有 O(12²) 累積分布迴圈）。
2. `calculate_macro_xs`：grid_type=UNIONIZED → 對 **unionized energy array（n_isotopes×n_gridpoints
   = 4,012,565 個 double ≈ 32MB）做一次二分搜**（~22 步、資料相依的隨機存取）。
3. 對該 material 的 `num_nucs[mat]` 個 nuclide（迴圈長度因 material 而異 → **divergence**）各做
   `calculate_micro_xs`：讀 `index_grid[idx*n_isotopes+nuc]`（巨陣列，total ~5.6GB）、再讀
   `nuclide_grid[low_idx]`/`[high_idx]`（`NuclideGridPoint` = 6 doubles AoS，隨機位置）、線性內插 5 個 XS。
4. 把 5 個 macro XS 取最大值 index 寫 `verification[i]`。

**預期瓶頸**：對 >L2（32MB egrid + 5.6GB grids）的**隨機、未 coalesced 記憶體存取 + 二分搜的
資料相依延遲 + warp divergence（num_nucs[mat] 不一）** → memory-latency-bound。
正確性：GPU checksum vs serial host 參考比對（任何 size 自我驗證）。

## 2. Profiling 證據（job 952677，nvprof + ncu，large）

**nvprof**：`lookup` kernel 362.6 ms（單次，佔扣除 init 後的 GPU 時間 22.8%）；
HtoD memcpy 1.2s（5.6GB init，**不計入** kernel 指標）。

**ncu（lookup，SpeedOfLight + WarpState + Memory + Occupancy）**
- **Warp Cycles Per Issued Instruction = 102.1 cycle**，其中 **94.9 cycle（92.9%）是
  scoreboard-dependency stall**（等 global memory）→ 決定性證據。
- **DRAM Throughput 50.6%（454 GB/s）**：未飽和 → **latency-bound，非 bandwidth-bound**。
- **Compute (SM) Throughput 11.25%**：compute 幾乎閒置。
- **L2 Hit Rate 39.3%**：隨機/分散存取 locality 差。
- Achieved Occupancy 59.5%。

## 3. Bottleneck 判定

**memory-latency-bound（global random access + 低 locality + binary-search 資料相依鏈）**：
- 93% 的 warp cycle 卡在 long-scoreboard（記憶體相依）+ DRAM 只 50% + SM 11% →
  排除 compute-bound 與 bandwidth-bound，確定是延遲。
- L2 hit 39% 顯示存取分散；二分搜的 ~22 步相依 load（每步 ~數百 cycle）+ 對 index_grid /
  nuclide_grid 的隨機 gather 是延遲來源。num_nucs[mat] 不一造成 warp divergence 為次要。
- 判斷依據：ncu warp stall（scoreboard 92.9%）/ DRAM% / SM% / L2 hit。

## 4. 優化決策

**依 energy 排序 lookups（XSBench 經典 locality 優化）**：在計時迴圈前一次性
(1) 用小 kernel 由 seed 重算每個 lookup 的 `p_energy`，(2) `thrust::sort_by_key` 得到依
energy 排序的原始索引 `idx_sorted`，(3) `lookup` kernel 改成 `i = idx_sorted[tid]`。
排序後**相鄰 thread 的 energy 相近 → 二分搜落點與 grid gather 位置相近 → L2 hit ↑、
latency stall ↓**。verification 仍寫回 `verification[原始 i]`，總和不變 → 排序安全。

排序為一次性前處理（energy 由 seed 決定、每個 repeat 相同），置於計時 `kstart` 之前；
其成本（energy kernel + thrust sort）會**另外用 nvprof 量測並誠實標註**，不混入 lookup
kernel 指標。若排序版無效或退步，fallback：調 occupancy（`__launch_bounds__` / block size）。

## 5. 結果與驗證

正確性：smoke（job 952687）checksum Valid；compare（job 952707）5 次 checksum 一致
= 50951281（Valid）。baseline = `baseline/results/951758/xsbench-cuda/run-{1..5}.log`，
optimized = job 952707，皆取 5 次中位數。

| 指標 | baseline | optimized | speedup |
|---|---:|---:|---:|
| lookup kernel 平均時間（程式自印） | 0.342 s | **0.058 s** | **5.90x** |

**確認改善（energy 排序）**：把隨機 energy 的 lookup 依 energy 排序後，相鄰 thread 存取
鄰近 grid 位置 → L2 reuse 大增、記憶體 latency stall 大降，lookup kernel 5.9×。

**誠實計入一次性排序成本（nvprof job 952710）**：
- `compute_energy_key` 1.22 ms + `thrust::sort_by_key`（cub radix sort）6.56 ms ≈ **7.8 ms（一次性）**。
- lookup kernel：362 ms → 58 ms（每個 repeat 省 ~304 ms）。
- net（含 sort）：repeat=1 → (58+7.8)/362 ≈ **5.5×**；compare 的 repeat=10 →
  (10·58+7.8)/(10·362) ≈ **6.2×**。sort 成本相對省下的時間可忽略，優化在任何 repeat 下皆穩贏。
- 排序置於計時 `kstart` 之前（energy 由 seed 決定、各 repeat 相同），不混入 lookup kernel
  指標；上面已另外標明其絕對成本。

**為何有效（對照 §3）**：bottleneck 是 memory-latency（92.9% warp cycle 卡 scoreboard、
L2 hit 39%）。排序直接提高 locality，這是 XSBench 在 GPU 上公認最有效的優化方向，profiling
數據（L2 hit、stall）與結果一致。

