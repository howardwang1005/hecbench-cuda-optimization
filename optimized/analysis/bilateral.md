# bilateral bottleneck 分析

執行參數：`./main 2960 1440 0.5 0.5 1000`（image 2960×1440，σ_I=0.5、σ_spatial=0.5、repeat=1000）。
比較指標：三個 radius 的 `Average kernel execution time (RxR) (ms)`。
baseline：3×3 = 0.920 ms、6×6 = 3.405 ms、9×9 = 7.276 ms（job 951758）。

## 1. Baseline 行為

`bilateralFilter<R>`（template on radius，main 跑 R=3/6/9）：每 thread 一個 output pixel，
掃 (2R+1)² 視窗，對每個鄰點：
- mirror-edge 邊界處理（branch）
- `range = -(I-I_w)²/(2σ_I²)`，`spatial = -((idk-idx)²+(idl-idy)²)/(2σ_s²)`
- `weight = a² * expf(spatial + range)` ← **每鄰點一個 expf（transcendental）**
- 累加 weight 與 I_w*weight，最後 `out = res/normalization`

工作量：9×9 → 每 pixel 361 個 expf，× 2960×1440 ≈ 4.26M pixel → ~1.5G expf/kernel。
時間隨 R² 成長（0.92→3.4→7.3ms ≈ (R+0.5)² 比例）→ 與「每鄰點固定成本」一致。

觀察：`spatial` 對**內部 pixel** 只與相對位移 (i,j) 有關（idk-idx=i, idl-idy=j），與 pixel 值無關
→ exp(spatial) 可預算成 (2R+1)² 的常數表；但 expf(range) 仍需每鄰點算（range 依 I_w）。
預期 **compute-bound（特殊函數單元 expf 主導）**，而非 memory-bound。

正確性：只比對 9×9 的 GPU 結果 vs CPU `reference<9>`（用 expf），容差 **1e-3 絕對值**
→ 若改 `__expf` 快速內建需驗證是否仍 PASS。

## 2. Profiling 證據（job 952714，nvprof + ncu）

**nvprof**：bilateralFilter<9> 8.65ms、<6> 4.08ms、<3> 1.12ms；memcpy 為輔。

**ncu（三個 radius）**
- **DRAM Throughput 0.9–3.3%**、Memory Throughput ~10% → **完全不是 memory-bound**。
- **Compute (SM) Throughput 75.6–78.4%**、**Issued IPC 3.0–3.2**（高）→ **compute-bound**。
- Achieved Occupancy 59–72%。

## 3. Bottleneck 判定

**compute-bound（transcendental `expf` 主導）**：DRAM ~1% 排除 memory-bound；SM 78% + 高 IPC
顯示指令吞吐受限，而每鄰點一個 `expf`（標準函式庫的多項式展開 = 數十條指令、走 MUFU/XU
pipe）是主成本。判斷依據：ncu DRAM% 極低 + SM% 高 + 工作量隨 (2R+1)² 的 expf 次數成長。

## 4. 優化決策

**`expf` → `__expf`（快速內建，單一 MUFU 指令）**（attempt 1）：對 transcendental-bound kernel
直接大幅減少指令數。風險：精度 vs verify 的 1e-3 絕對容差 → 以 smoke 驗證是否仍 PASS。
（備案：若 __expf 失準，改「預算 exp(spatial) 常數表 + 保留 expf(range)」的 exact 版本。）

## 5. 結果與驗證

正確性：smoke（job 952718）PASS；compare（job 952719）PASS（只比對 9×9，tol 1e-3）。
baseline = `baseline/results/951758/bilateral-cuda/run-{1..5}.log`，optimized = job 952719，
皆取 5 次中位數。

**優化步驟與效果（逐步累加）**：

| 步驟 | 3×3 | 6×6 | 9×9 |
|---|---:|---:|---:|
| baseline | 0.920 ms | 3.414 ms | 7.276 ms |
| attempt 1：`expf`→`__expf`（job 952717） | 0.843 ms (1.09x) | 3.041 ms (1.12x) | 6.603 ms (1.10x) |
| **attempt 2：+ constant spatial 表 + interior fast-path + reciprocal（job 952719）** | **0.327 ms (2.81x)** | **1.144 ms (2.98x)** | **2.519 ms (2.89x)** |

確認改善：**~2.8–3.0×**。attempt 1（`__expf` 單獨）只有 1.1×，顯示 expf 雖走 MUFU 但
**並非唯一瓶頸**；真正關鍵是 attempt 2 把內層每鄰點的 **spatial 計算（平方/加/除/轉型）與一個
transcendental 折成 constant 表查值**、並對內部像素**移除 4 個 mirror-edge 分支**，大幅降低
SM 78% 中的非必要指令。三個 radius 改善幅度相近（~2.9×），符合「每鄰點固定成本下降」的預期。

**推測為何不是更高**：剩餘成本是每鄰點仍需 1 次 global load（in[I_w]，雖多被 L1/L2 吸收）、
1 次 `__expf(range)`（range 依 I_w 無法預算）、與累加 FMA；這些是 bilateral 的本質下限。
進一步可試 shared-mem input tiling 降 load，但 profiling 顯示 memory 佔比低，預期幫助有限。
