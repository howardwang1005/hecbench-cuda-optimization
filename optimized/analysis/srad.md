# srad bottleneck 分析

執行參數：`./main 1000 0.5 502 458`（Rodinia SRAD，真實 `../data/srad/image.pgm`）。
比較指標：自印分段計時的 **`COMPUTE (1000 iterations)`** stage（baseline ≈ 0.0560 s）。
無內建 PASS/FAIL → 以輸出 `image_out.pgm` 對 baseline diff 驗證。

## 1. Baseline 行為

各向異性擴散去噪，每 iteration：`prepare` → 多階 `reduce`（算整張影像的 sum、sum2）→
**D2H 拷回 sum/sum2，host 算 mean/variance(q0sqr)** → `srad` → `srad2`。1000 iters。
`reduce` kernel 用**最慢的 interleaved-addressing + `(tx+1)%i==0` modulo 規約**。

## 2. Profiling 證據（job 952757，nvprof）

GPU activity 佔比：**`reduce` 39.1%（13.3 ms，3000 launches＝每 iter 3 階）**、srad 23.8%、
srad2 17.6%、prepare 9.6%、**DtoH memcpy 9.6%（2001 次＝每 iter 把 sum 拷回 host）**。

## 3. Bottleneck 判定

兩層：
- **GPU 端**：`reduce` 是最大 kernel（39%），且用了 divergent 的 modulo 規約 → 可優化。
- **Stage 端（COMPUTE 56 ms）**：受**每 iteration 的 host 同步**主導——reduce 結果 D2H 拷回、
  host 算 mean/variance、再進下一輪 kernel；這個 per-iter round-trip 是關鍵路徑。
  判斷依據：nvprof 的 reduce 佔比 + 每 iter 的 DtoH memcpy + stage 時間(56ms) 遠大於 GPU
  kernel 總和(~34ms)。

## 4. 優化決策與結果

**attempt 1：把 `reduce` 改成 sequential-addressing 規約**（連續 active lane、無 modulo、
無 bank conflict、inactive lane 補 0），保留多階 load/store 索引 → host 驅動迴圈不動。
→ COMPUTE stage **~1.06×**、**輸出影像位元相同（IDENTICAL）**。

**為何只有 1.06×（推測，profiling 支持）**：reduce 的 GPU 時間雖降，但 COMPUTE stage 被
**每 iteration 的 D2H 同步 + host mean/variance 計算**卡住（stage 56ms ≫ GPU kernel 34ms）。
即使 reduce 13.3ms→~6ms，對 56ms 的 stage 也只 ~1.1× 上限。
**下一步方向（未做）**：把 mean/variance 計算搬上 device（消除 per-iter D2H round-trip），
讓 srad kernel 直接吃 device 上的 q0sqr → 可解 stage 端的 host-sync 瓶頸。

## 5. 結果與驗證

正確性：`image_out.pgm` 對 baseline **完全相同（cmp IDENTICAL）**。

| | baseline | optimized | speedup |
|---|---:|---:|---:|
| COMPUTE (1000 iters) stage（3 次中位數） | 0.0560 s | **0.0530 s** | **1.06×** |

reduce kernel 本身已換成高效規約（正確、輸出不變）；stage 端受 host 同步限制，誠實記為
小幅改善 + 已指出後續可解的 host-sync 瓶頸。
baseline = `baseline/results/951758/srad-cuda/run-{1..5}.log`。
