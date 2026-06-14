# heat bottleneck 分析

執行參數：`./main 4096 1000`（4096² grid、1000 步）。FP64。比較指標：自印
`Solve time (s)`（solve kernel 迴圈）。baseline ≈ **0.341 s**（job 951758，~787 GB/s）。
內建驗證：`Error (L2norm)` ≈ 4.6e-10（MMS 解析解比對）。

## 1. Baseline 行為

`solve` kernel：2D 5-point 熱傳 stencil，**naive**（1D launch，`i=idx%n, j=idx/n`，
無 shared，每 cell 讀 center + 4 鄰點、寫 1 個，含邊界 ternary）。記憶體流量主導。

## 2. Profiling 證據（job 952740，ncu）

- nvprof：solve 340 µs ×100（佔扣除 init memcpy 後幾乎全部 GPU 時間）。
- ncu：**DRAM Throughput 88.06%**、Compute(SM) 24.2%、**L2 Hit 74.9%**。

## 3. Bottleneck 判定

**memory-bandwidth-bound、已近 roofline（88% DRAM）**：SM 24% 排除 compute-bound；
L2 hit 74.9% 表示水平/垂直鄰點重用已被 cache 吸收，剩餘為必要的 read+write 流量。
`i=idx%n,j=idx/n` 的整數除法/取模**不是瓶頸**（DRAM 已飽和、被掩蓋）。
判斷依據：ncu DRAM% 88% + SM% 24% + L2 hit。

## 4. 優化決策與結果

**不改動 kernel**（已近 memory roofline，headroom 極小）。
- 即使把 1D launch + div/mod 改成 2D grid（省整數運算），DRAM 已 88% → 預期無顯著改善；
  shared-mem tiling 對 2D 5-point 在大 L2 的 V100 上通常也不會贏 naive（cache 已抓住重用，
  反增 sync），與 convolution1D 的教訓一致。
- 誠實記為 memory-roofline-bound 案例。

## 5. 結果與驗證

未修改 → 與 baseline 等價（內建 L2norm 驗證仍成立）。
baseline = `baseline/results/951758/heat-cuda/run-{1..5}.log`（Solve time ≈ 0.341 s）。
