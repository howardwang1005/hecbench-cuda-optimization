# lavaMD bottleneck 分析

執行參數：`./main -boxes1d 30`。`fp = float`。比較指標：自印 `Kernel execution time`
（baseline ≈ 0.0343 s）。無內建 PASS/FAIL；可用 `-DOUTPUT` 產生 `result.txt`（所有 force）
對 baseline diff 驗證（`fv_cpu` 初始為 0）。

## 1. Baseline 行為

單一 `md` kernel：Lennard-Jones 力計算。每個 box 一個 block（NUMBER_THREADS=128），把 home
box 的粒子載入 `rA_shared`，對 (1+nn≈27) 個鄰居 box，把鄰居粒子載入 `rB_shared/qB_shared`，
內層對 100 個鄰居粒子算 `r2`(DOT) → `exp(-u2)` → 4 個 force 分量。已用 shared tiling。
NUMBER_PAR_PER_BOX=100 ≤ 128 → 每 thread 剛好一個粒子。
注意：baseline 在內層**每次迭代對 global `d_fv_gpu` 做 4 個 +=**，且呼叫 **double `exp`**（fp 是 float）。

## 2. Profiling 證據（job 952743，ncu）

- **DRAM 0.63%**、**L2 Hit 98%**（資料全在 shared/cache）、**Compute (SM) 81.9%**、IPC 2.09、
  occupancy 56% → **compute-bound**（非 memory）。

## 3. Bottleneck 判定

**compute-bound**（SM 82%、DRAM 0.6%）。瓶頸是內層的 LJ 力 compute：DOT + transcendental +
多個 FMA × 100 鄰粒 × 27 box。判斷依據：ncu DRAM%/SM%/L2 hit。

## 4. 優化決策與三次嘗試

1. **register 累加**（消除內層 global RMW，最後寫一次）→ **~1.0×（noise）**。推測：`d_fv_gpu`
   位址在內層固定（wtx=tx），編譯器本就大致暫存；且 kernel 非 memory-bound，故無感。保留（更乾淨）。
2. **`exp`→`expf`**（避免 float→double 提升）→ **~1.01×**，且 result 與 baseline **完全相同
   （max diff 0 / 2.7M）**。推測：expf 與 double exp 都是多指令多項式，指令數相近。
3. **`expf`→`__expf`**（單一 MUFU 指令）→ **1.15×**。確認：transcendental 確實占可觀指令，
   換成硬體 MUFU 才真正減少。代價：max force 偏差 **3.7e-4**（float 精度級；lavaMD 無正式容差）。

最終採用 register 累加 + `__expf`。

## 5. 結果與驗證

正確性：`-DOUTPUT` 的 `result.txt` 對 baseline diff，**max abs diff = 3.67e-4 / 2,700,000 rows**
（float 級偏差，無正式容差故視為通過）。

| | baseline | optimized | speedup |
|---|---:|---:|---:|
| Kernel execution time（3 次） | ~0.0343 s | **~0.0299 s** | **1.15×** |

**推測為何僅 1.15×**：compute-bound 且力計算的 DOT/FMA 是演算法本質、無法縮減；只有 exp 這段
可用 MUFU 加速，故上限有限。再快需減少 box-pair 工作量（演算法層級）或 mixed precision。
baseline = `baseline/results/951758/lavaMD-cuda/run-{1..5}.log`（Kernel execution time）。
