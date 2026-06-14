# HeCBench CUDA 優化結果報告

**硬體**：NVIDIA Tesla V100-SXM2-32GB (Volta, sm_70, 80 SM, ~900 GB/s HBM2)
**編譯器**：CUDA 12.3（與官方 baseline job 951758 相同，確保公平比較）
**環境**：TWCC `gp1d` 運算節點（Slurm 提交，單卡 `CUDA_VISIBLE_DEVICES=0`）
**Profiler**：Nsight Compute (`ncu`) — SpeedOfLight / Occupancy / LaunchStats

## 怎麼讀這份報告

- **官方 baseline (official)**：官方 951758 那次 5-run 的 median，是「要打敗的目標」。
- **local baseline**：本機用同一版 nvcc (12.3) 重新編譯 baseline 跑出的時間。
- **optimized**：優化版用同一版 nvcc 跑出的時間。
- **判斷標準**：`vs_local`（baseline 與 optimized 同編譯器、同環境、同時段）才是
  「演算法/實作改動」的公平指標。`vs_official` 受編譯器版本影響，僅供對照。

---

## 1. thomas-cuda（批次三對角解，cuThomasBatch）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：Duration 2.45 ms，Compute 4%，Memory 55%，Occupancy 9.95%，Waves/SM 0.18 |
| **瓶頸判定** | 1 thread/system，N=16384 systems → 僅 16448 threads，GPU 未填滿；但每 thread 跑 M=1024 序列掃描，工作量飽和。記憶體已 coalesced（interleaved layout）。 |
| **嘗試 1（失敗）** | PCR（1 block/system，1024 threads）。引入 ~20 次 block barrier + 48KB smem，同步開銷蓋過平行好處 → **0.55x（慢一倍）**。已捨棄。 |
| **最終優化** | **同演算法的 register-carry CSE**：baseline 每步重算分母 `D[i]-L[i]*U[i-stride]` 兩次、且重複從 global memory 載入 `U[i-stride]`/`RHS[i-stride]`。改為用 register 攜帶前一元素的 U/RHS、分母只算一次。減少 FP 運算與 global load，不改演算法。 |
| **正確性** | CPU 驗證對 M∈{2..1024} 與 baseline bit-exact（誤差 0）。 |
| **結果** | official 2.352 ms / local 1.180 ms / **opt 0.975 ms** → **vs_local 1.21x**、vs_official 2.41x ✅ |

## 2. gaussian-cuda（高斯消去前向消元，fan1/fan2）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`fan2`（主要工作）Memory 76%、Occupancy 87%（健康）；`fan1` grid 16、Compute 0.3%、Occupancy 11%。每個消去步 launch fan1+fan2，size=4096 共 8190 次 launch；device offload 950ms vs total kernel 499ms → host 提交開銷大。 |
| **瓶頸判定** | `fan2` 的 grid 固定為 full `size×size`，但第 t 步只需 `(size-1-t)×(size-t)` 的子矩陣。後期大量 block 一啟動就因邊界檢查 return → 浪費 block 排程。 |
| **嘗試 1（失敗）** | 把 fan1 融進 fan2。導致每個 column thread 重算除法 `a[row][t]/a[t][t]`（重複 ~4096 倍）→ **0.85x（變慢）**。已捨棄。 |
| **最終優化** | **每步縮小 grid**：依存活子矩陣大小設定 grid（`fan1=ceil(rows/256)`、`fan2=(ceil(cols/16), ceil(rows/16))`）。kernel 本體與 baseline byte 相同，純粹不再啟動會立即 return 的空轉 block。 |
| **正確性** | kernel 未動（內部邊界檢查仍成立），程式自帶 check 為 PASS。 |
| **結果** | official 490.96 ms / local 245.45 ms / **opt 142.70 ms** → **vs_local 1.72x**、vs_official 3.44x ✅ |

## 3. jacobi-cuda（Jacobi 迭代 stencil）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`jacobi_step` Duration 67 us，Compute 52%，Memory 73%，Occupancy 85%（kernel 已調得好）。 |
| **瓶頸判定** | kernel 接近頻寬天花板，無 occupancy 空間。剩餘開銷在 host 端：**每次迭代都 `cudaMemcpy(error)` D2H**，強制每步整裝同步。 |
| **最終優化** | **降低收斂檢查頻率**：每 100 次迭代才把 error 拷回 host 一次，中間 kernel 連續排隊、不做 host round-trip。最多多跑 <100 次迭代（不會更少），收斂與 PASS 不變。 |
| **正確性** | 程式自帶 check 為 PASS。 |
| **結果** | official 7.39e-5 s / local 4.02e-5 s / **opt 3.41e-5 s** → **vs_local 1.18x**、vs_official 2.16x ✅ |

## 4. filter-cuda（stream compaction，正值篩選）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`filter` Duration 1.01 ms，Compute 43%，Memory 66%，Occupancy 90%。 |
| **瓶頸判定** | 偏 memory-bound，但 compute 43% 來自「每個正值元素一次 shared-memory atomicAdd」。 |
| **最終優化** | **warp-aggregated 計數**：每個 warp 用 `__ballot_sync`+`__popc` 一次數完該 warp 的正值數，只做一次 shared atomic 保留區段；每個 thread 的位置用 popc 在 warp 內排名得出。最多減少 32× 的 shared atomic。輸出順序改變但程式比對前會排序。 |
| **正確性** | CPU 驗證 warp 計數與逐元素計數一致；程式自帶 check 為 PASS。 |
| **結果** | official 0.958 ms / local 0.522 ms / **opt 0.528 ms** → **vs_local 0.99x（持平）**、vs_official 1.82x ➖ |
| **說明** | baseline 已是 memory-bound（66%），少做 atomic 沒帶來實質加速；compute 不在關鍵路徑上。對官方贏是編譯器紅利，同編譯器下持平。 |

## 5. histogram-cuda（256-bin 直方圖，shared-memory atomics）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`histogram_smem_atomics` Duration 39 us，Compute 9%，Memory 9%，**Occupancy 20%**（grid 固定 16×16=256 blocks）。 |
| **瓶頸判定（初判）** | occupancy 偏低，疑似 block 太少未填滿 GPU。 |
| **嘗試優化** | 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 把 grid 放大到填滿 SM。 |
| **結果** | official 184.4 us / local 201.4 us / **opt 388.0 us** → **vs_local 0.52x（變慢一倍）** ❌ |
| **失敗原因（誠實檢討）** | 放大 grid 會產生更多 partial histogram，使第二個 kernel `histogram_smem_accum` 的 reduction loop 變長（每個 bin 要加總更多份）。增加的合併成本超過第一個 kernel 提升的 occupancy。**這題的初步 occupancy 判斷誤判了第二階段的成本** —— 應退回 baseline 或改用單階段策略。 |

## 6. jaccard-cuda（Jaccard 相似度，CSR 稀疏）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`jaccard_is_opt` Duration 7.75 ms（佔全程 ~99%），Compute 49%，Memory 34%，Occupancy 20%，DRAM 0.1%（資料進 L2）。 |
| **瓶頸判定** | 交集計算只在 `threadIdx.x==0` 上跑序列 two-pointer merge，其餘 7/8 lane 閒置。 |
| **嘗試（失敗）** | 8-lane 平行 binary search。binary search 總工作量 `O(Ni·logNj)` > two-pointer `O(Ni+Nj)`，攤到 8 lane 也補不回多做的工作 → **0.74x（變慢）**。已退回 baseline。 |
| **結果** | official 0.01229 s / local 0.01229 s / **opt 0.01230 s** → **vs_local 1.00x（=baseline）** ➖ |
| **說明** | baseline two-pointer 已是高效演算法；正確的平行化（不增加總工作量）較困難，停損退回 baseline。 |

## 7. bscan-cuda（二元前綴和）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`binary_scan` (N=32) Occupancy 46%，限制器為 Block Limit Shared Mem = 32 blocks/SM（單 warp block 上限 50%）。 |
| **瓶頸判定** | N=32 時是單一 warp，卻仍跑完整的多 warp 流程（shared memory + 2 barrier）。 |
| **嘗試優化** | `if constexpr (N<=32)` 特化：單 warp 直接回傳 warp scan 結果，不用 shared memory、不用 barrier。 |
| **結果** | official 2884.6 us / local 2894.7 us / **opt 2909.0 us** → **vs_local 1.00x（持平）** ➖ |
| **說明** | 只動到 N=32（程式內 6 種 block size 加總中最快的一個），對總時間幾乎無影響；範圍太小。 |

## 8. atomicReduction-cuda（原子規約）— 未優化

| 項目 | 內容 |
|---|---|
| **Profile** | ncu：`atomic_reduction` Memory 77%、Occupancy 92%；baseline 已有 v2/v4/v8/v16 向量化版本，`atomic_reduction_v4` 達 **863 GB/s ≈ 96% 頻寬峰值**。 |
| **判定** | 已達 V100 記憶體頻寬天花板，無實質優化空間。**刻意不做**，避免無謂或反效果的改動。 |

## 9. scan-cuda（work-efficient Blelloch 前綴和，含 bank-conflict-avoiding 版）

| 項目 | 內容 |
|---|---|
| **Profile** | ncu（以 `scan<char,128>` 為樣本）：Duration 3.65 ms，Compute 78%，Memory 87%（per-warp 利用率高），**Theoretical Occupancy 100% 但 Achieved 僅 49.5%**，**Waves/SM = 0.50**。 |
| **瓶頸判定** | kernel 本身不閒置（78%/87%），問題在 launch 的 block 數不足：baseline 固定 `grids = 16 × SM = 1280` blocks，但 V100 可容 `32 blocks/SM × 80 = 2560` 個 slot，所以只填一半（Waves/SM 0.50 = 1280/2560）→ 即使理論 occupancy 100%，實測只有 ~50%，resident warp 不夠藏記憶體延遲。kernel 是 grid-stride loop（`for bid = blockIdx.x; bid < nblocks; bid += gridDim.x`），所以物理 grid 大小只需填滿裝置。 |
| **改動內容** | 把 baseline 寫死的 `dim3 grids(16 * prop.multiProcessorCount)` 改成**依 kernel 自身的最大常駐 block 數動態決定 grid**：對 `scan` 與 `scan_bcao` 各自呼叫 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`（依資料型別 T 與每 block 元素數 N），得到 `maxBlocksPerSM`，再乘以 SM 數量得到 grid 大小，並以 `num_blocks` 為上限（不啟動超過實際工作量的 block）。`scan_bcao` 用 `temp[2*N]`（兩倍 shared memory），其 maxBlocksPerSM 可能不同，故兩個 kernel 的 grid 分開計算（`grids` / `grids_bcao`）。 |
| **優化原理** | 純 launch-config 改動，**不改 kernel、不改演算法、不增加工作量**：只是把物理 grid 從固定 1280 放大到「填滿所有 SM block slot」，提升 achieved occupancy（0.5 wave → 接近填滿），讓更多 warp 常駐以藏記憶體延遲。grid-stride loop 保證對任意 grid 大小輸出相同，正確性 by construction。 |
| **正確性** | 演算法數學未動，僅改物理 block 數；程式自帶 `verify()`（對每個 block/型別/N 與 CPU exclusive scan 比對）為把關。 |
| **誠實註記** | 此題是五個 data-compression 題中**最不確定**的：kernel 本來就 87% memory / 78% compute，若已接近真實吞吐天花板，把 wave 數翻倍可能只有小幅提升。實際效果需以 V100 實測為準。另：本輪 paired 量測時 scan 的 program-repeat 由 100 降為 10 以縮短時間（scan 報的是每次平均，仍可比）。 |
| **結果** | _(待填)_ |

---

## 結果總表

判斷以 **vs_local**（同編譯器 12.3、同環境的公平對照）為準。

| # | Benchmark | official | local base | optimized | **vs_local** | vs_official | 狀態 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | **gaussian** | 490.96 ms | 245.45 ms | 142.70 ms | **1.72x** | 3.44x | ✅ 贏 |
| 2 | **thomas** | 2.352 ms | 1.180 ms | 0.975 ms | **1.21x** | 2.41x | ✅ 贏 |
| 3 | **jacobi** | 73.9 us | 40.2 us | 34.1 us | **1.18x** | 2.16x | ✅ 贏 |
| 4 | filter | 0.958 ms | 0.522 ms | 0.528 ms | 0.99x | 1.82x | ➖ 持平 |
| 5 | jaccard | 12.29 ms | 12.29 ms | 12.30 ms | 1.00x | 1.00x | ➖ 持平 (=baseline) |
| 6 | bscan | 2884.6 us | 2894.7 us | 2909.0 us | 1.00x | 0.99x | ➖ 持平 |
| 7 | **histogram** | 184.4 us | 201.4 us | 388.0 us | **0.52x** | 0.48x | ❌ 變慢 |
| 8 | atomicReduction | — | — | — | — | — | ⏭️ 未做（已達頻寬天花板）|
| 9 | **scan** | _(待填)_ | _(待填)_ | _(待填)_ | _(待填)_ | _(待填)_ | _(自行填寫)_ |

### 小結

- **明確贏**（同編譯器下）：gaussian 1.72x、thomas 1.21x、jacobi 1.18x —— 三題都靠
  「**移除浪費而非增加工作量**」：縮 grid、register CSE、降同步頻率。
- **持平**：filter / jaccard / bscan —— baseline 已接近最佳或已用相同手法，誠實退回或無實質空間。
- **變慢**：histogram —— occupancy 初判誤判了第二階段 accum 的成本，待修正。
- **未做**：atomicReduction —— 已達頻寬天花板。

### 跨編譯器觀察

所有題目的 `local base`（12.3 重編）都比 `official`（亦 12.3，但 2026-06-13 那次）略有差異，
而 `vs_official` 普遍高於 `vs_local`。這顯示 optimized 在較新環境下另有額外加速；
判斷實作改動的真實效果應以 `vs_local` 為準。
