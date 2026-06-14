# convolution1D bottleneck 分析

執行參數：`./main 134217728 1000`（input_width 進位到 1024 倍數，repeat=1000）
baseline measured wall：~578 s（job 951758）。比較指標：各程式自印的
`Average kernel execution time of ... (us)`。

## 1. Baseline 行為

`src/convolution1D-cuda/main.cu` 是一支「教學比較 harness」，對同一個 1D 卷積問題
提供三種 kernel：

| kernel | 作法 | shared mem |
|---|---|---|
| `conv1d` | 每 thread 算 1 個 output，直接從 global 讀 `mask_width` 個鄰居（含邊界檢查） | 無 |
| `conv1d_tiled` | 先把 tile + 左右 halo 載入 shared，再從 shared 卷積 | `(bs+9)*sizeof(T)` |
| `conv1d_tiled_caching` | 只載入 center tile 到 shared，halo 直接讀 global（靠 L2） | `bs*sizeof(T)` |

mask 放 `__constant__ mask[MAX_MASK_WIDTH]`（值全為 1）。輸入 `a[i]=rand()%256`。

**問題規模 / launch 次數**：外層掃 `mask_width ∈ {3,5,7,9}`（4）×型別
`{double, float, int16_t}`（3）×3 種 kernel ×`block size ∈ {64,128,256,512,1024}`（5）
×`repeat=1000`
→ 約 **180,000 次 kernel launch**，每次處理 ~1.34e8 個元素。

**算術強度極低**：每個 output 僅 `mask_width`(3~9) 次乘加，但要讀 `mask_width` 個
input + 寫 1 個 output → 屬於典型 **memory-bound streaming**。`conv1d` 的相鄰 thread
讀取位址連續、重疊（每個元素被相鄰 thread 重複讀 mask_width 次），重疊部分主要靠
L2 吸收；輸出寫入 coalesced。

**注意（量測解讀）**：每個 (kernel,blocksize) 組合在 repeat 迴圈後會呼叫一次
`reference()` 做 CPU 驗證（O(input_width × mask_width)），共約 180 次 → baseline 的
578 s wall 有相當比例是 **CPU 驗證**，不是 GPU。我們的 speedup 只看各 kernel 自印的
us，已隔離此干擾。

## 2. Profiling 證據（job 952532，nvprof + ncu，CUDA 12.8 / sm_70）

**nvprof `--print-gpu-summary`（PROF_ARGS = `134217728 2`）**
- GPU activity 90.87% 是 `[CUDA memcpy DtoH]`（180 次，每次 ~128MB）——這是 harness
  每個 (kernel,blocksize) 都做一次 CPU `reference()` 前的回傳，是**量測 artifact**，
  與 kernel 效能無關（我們的指標只看各程式自印的 kernel us）。
- 各 kernel 平均時間（40 launches 各）：

  | kernel | double | float | int16 |
  |---|---:|---:|---:|
  | conv1d (basic) | 2.86 ms | **1.84 ms** | 2.08 ms |
  | conv1d_tiled | 3.02 ms | 2.15 ms | 2.21 ms |
  | conv1d_tiled_caching | 2.86 ms | 2.32 ms | 2.58 ms |

  → **basic `conv1d` 一律最快**；tiled / tiled_caching 因多了 shared 載入 + `__syncthreads`
  反而更慢（小 mask 時 L2 已能吸收重疊讀取，tiling 純屬額外開銷）。
- 也注意 **`conv1d<int16>` (2.08ms) 比 `conv1d<float>` (1.84ms) 還慢**，儘管位元組數只有一半。

**有效 DRAM 頻寬（由 conv1d 平均時間反推，最小流量 = 讀一次+寫一次）**

| 型別 | bytes/elem | conv1d | 最小流量 | 有效 BW | % of ~900GB/s peak |
|---|---:|---:|---:|---:|---:|
| double | 8 | 2.86 ms | 2.15 GB | ~752–803 GB/s | **84–89%** |
| float  | 4 | 1.84 ms | 1.07 GB | 582 GB/s | **65%** |
| int16  | 2 | 2.08 ms | 0.54 GB | 258 GB/s | **29%** |

**ncu（`conv1d<double>` 各 block size）**
- block 128–1024：DRAM Throughput **89.5%**、Memory Throughput **803 GB/s**、
  Compute(SM) Throughput 27%、Achieved Occupancy 84%。→ double 已接近記憶體 roofline。
- block 64：DRAM 67.7%、607 GB/s、occupancy 48.8%。→ 小 block occupancy 不足、未飽和。
- L2 Hit Rate ~54%、L1/TEX ~50%（halo 重疊讀取的命中；主體為 streaming 必然 miss）。

## 3. Bottleneck 判定

**memory-bound，但飽和度隨型別寬度而異**：
- `double`（8B 載入）已達 ~89% DRAM peak → **bandwidth-bound、近 roofline，headroom 小**。
- `float`（4B）只到 65%、`int16`（2B）只到 29% → **未飽和**。根因是
  **每 thread 純量窄載入產生的 memory-level parallelism 不足**：4B/2B 交易無法像 8B 那樣
  把 DRAM 排滿，於是 latency-bound 而非 bandwidth-bound。
- 小 block size（64）另外受 **occupancy 限制**。
- tiled / tiled_caching：對小 mask 是**反效果**（shared+sync overhead），屬 rejected。

判斷依據：ncu 顯示 double 89% / float·int16 由時間反推僅 65% / 29%，且 SM throughput 全程
低（~27%）排除 compute-bound；DtoH memcpy 佔比高但屬驗證 artifact、不計入 kernel 指標。

## 4. 優化決策

**對 `conv1d` 做 16-byte 向量化載入/儲存**（`double2` / `float4` / `short8`），
每 thread 處理 `16/sizeof(T)` 個連續 output：中心元素以單一 128-bit 交易讀入（跨 thread
連續 → coalesced），左右 halo（≤4 個）以純量讀取靠 L2。如此把 float/int16 的 memory-level
parallelism 拉到與 double 同級，預期推向 DRAM roofline。

- 預期：**int16 ~3×、float ~1.4×、double 持平**（已近 roofline）。
- `tiled` / `tiled_caching` 維持原樣（profiling 已證明它們較慢，列為 rejected）。
- 正確性：out-of-range halo 補 0，與 `reference()` 的邊界處理一致；input_width 為 1024 倍數，
  E∈{2,4,8} 與各 block size 皆整除，向量存取對齊（cudaMalloc ≥256B 對齊）。

## 5. 結果與驗證

正確性：smoke test（job 952620）180/180 全 PASS；compare（job 952621）全 PASS。
baseline = `baseline/results/951758/convolution1D-cuda/run-{1..5}.log`（per-kernel 平均
時間與 repeat 無關，可直接重用，不必重跑）；optimized = job 952621，各取 5 次中位數。

**最終採用：保持 coalesced 的 interior fast-path + 編譯期展開 mask（templated mask width）。**

`conv1d` kernel（平均 over mask×blocksize；及 mask9 最佳 block size）：

| 型別 | baseline (us) | optimized (us) | speedup (avg) | 最佳 block (mask9) |
|---|---:|---:|---:|---:|
| double | 2759.0 | 2741.8 | 1.01x | 1.00x（近 DRAM roofline，如預期無改善）|
| float  | 1802.1 | 1782.8 | 1.01x | 1.07x |
| int16  | 1752.2 | 1504.0 | **1.17x** | **1.39x**（bs512: 1668.8→1197.6us）|

結論：與 §3 判定一致——`double` 已近記憶體 roofline、headroom 極小；`int16`（原本 29% peak、
受邊界分支等固定指令開銷拖累）在移除內部 boundary branch 後得到 **1.17x 平均 / 1.39x 最佳**；
`float` 介於兩者之間、改善有限。`conv1d_tiled` / `conv1d_tiled_caching` 未修改（profiling
已證實對小 mask 是反效果）。

剩餘 headroom：int16 優化後仍約 40% peak，後續可嘗試「每 thread 以 32-bit 載入 2 個 int16」
進一步提高 memory-level parallelism（尚未做）。

### Rejected experiment — 128-bit 向量化 + 純量 halo

第一版嘗試讓每 thread 處理 `E=16/sizeof(T)` 個輸出，中心以 `double2/float4/short8` 單一
128-bit 交易載入、左右 halo 以純量讀取。實測**大幅退步**（compare job 952606 run1）：

| 型別 | baseline | vectorized | speedup |
|---|---:|---:|---:|
| double/conv1d | 2767us | 9719us | **0.28x** |
| float/conv1d | 1852us | 3362us | 0.55x |
| int16/conv1d | 1752us | 1687us | 1.04x |

原因：baseline 的純量 `conv1d` 本來就**完全 coalesced**（固定 tap j 時 warp 內位址連續），
重疊讀取由 L2 吸收；向量化版本每 thread 多出的 `2h` 個純量 halo 載入跨 warp 是 **stride-E
的非 coalesced 存取**，double（E=2）時 halo 還遠多於 center，導致記憶體效率崩潰。
→ 教訓：降低「表面上的重複讀取」不等於更快；破壞 coalescing 的代價遠大於省下的重疊讀取。
