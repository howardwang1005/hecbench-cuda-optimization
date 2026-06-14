# convolutionSeparable bottleneck 分析

執行參數：`./main 8192 8192 1000`。比較指標：自印 `Average kernel execution time (s)`
（包住 conv_rows + conv_cols 各 numIterations）。baseline ≈ **0.001526 s**（job 951758）。
內建驗證：`L2norm < 1e-6 → PASS`。

## 1. Baseline 行為

NVIDIA SDK 可分離卷積範例：`conv_rows` + `conv_cols` 兩 pass，已用 shared-memory tiling
（halo steps + 每 thread 8 個 result steps、conv_cols shared 有 +1 padding 避 bank conflict）。
**但 filter `kernel[]` 以 global pointer 傳入**（`kernel[KERNEL_RADIUS - j]`），
非 `__constant__`（原始 NVIDIA 範例用 `c_Kernel` 常數陣列）。

## 2. Profiling 證據（job 952732，nvprof + ncu）

- nvprof：conv_rows 923 µs、conv_cols 737 µs（其餘為 init memcpy）。
- ncu conv_rows：**DRAM 64.7%**、SM 48.9%、L2 54.5%、**occupancy 96.5%**。
- ncu conv_cols：**DRAM 83.5%（近 roofline）**、SM 41.8%、L2 53%。

## 3. Bottleneck 判定

**memory-bound、且已被 NVIDIA 範例充分優化**：conv_cols 已達 83.5% DRAM、occupancy 96.5%；
conv_rows 64.7% 略有空間但非 filter 造成。filter 僅 17 個 float、被 cache 完全吸收，
不是瓶頸。

## 4. 優化決策與結果

**attempt 1：filter → `__constant__` memory**（warp 廣播、對齊原始 NVIDIA 設計）。
→ 實測 **1.006×（noise）**：證實 filter 讀取本就不是瓶頸（太小、已快取）。保留此修改
（更乾淨、符合原設計）但**無可量測加速**。

## 5. 結果與驗證

正確性：smoke（job 952735）PASS、compare（job 952737）PASS（內建 L2norm<1e-6）。

| | baseline | optimized | speedup |
|---|---:|---:|---:|
| conv_rows+conv_cols 平均時間 | 1.526 ms | 1.517 ms | **1.006×（noise）** |

**推測為何無顯著改善**：此為 NVIDIA 高度優化的 separable conv 範例，conv_cols 已近 memory
roofline（83.5% DRAM）、occupancy 96.5%；conv_rows 的 64.7% 受 row halo 載入/位址模式而非
filter 限制。要再快需重排 row pass 的存取或改 tiling 參數，屬範例級微調、邊際效益低。
誠實記為**已優化、近 roofline**。baseline = `baseline/results/951758/convolutionSeparable-cuda/run-{1..5}.log`。
