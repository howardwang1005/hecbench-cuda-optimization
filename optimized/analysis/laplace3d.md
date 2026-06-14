# laplace3d bottleneck 分析

執行參數（baseline 951758 跑兩個 config）：`128 128 128 100 1`（verify on，0.000114 s）與
**`512 512 512 100 0`（verify off，0.002154 s，真正目標）**。比較指標：自印
`Average kernel execution time (s)`。FP32。內建 verify（128³ config：RMS error + PASS）。

## 1. Baseline 行為

經典 Mike Giles shared-memory z-marching 7-point Laplace stencil（kernel.h）：block 32×8，
每 thread 一個 (i,j) 節點，沿 k 方向 march，shared `u1[3*KOFF]` 同時保留 3 個 k-plane（
k-1/k/k+1），含 halo 載入（y-halo coalesced）。已 tiling 過。

## 2. Profiling 證據（job 952739，512³，ncu）

- nvprof：laplace3d kernel 2.21 ms（其餘為 init memcpy）。
- ncu：**DRAM Throughput 61.5%**、Compute(SM) 32.4%、**L2 Hit 68.7%**、Registers 23、
  Theoretical Occupancy 100%（shared 限 8 blocks/SM）、**Waves Per SM = 1.60**
  （1 full + 385-block partial wave）。

## 3. Bottleneck 判定

**memory-bound 但未飽和（61.5% DRAM），主要受 partial-wave tail 限制**：
- occupancy 理論 100%（非 occupancy 問題）；但 grid 僅 16×64 = 1024 個 block、每個 block
  march 全部 512 個 k-plane → 只有 **1.6 waves**，0.6 的 partial wave 造成 ~19% 尾端浪費。
- L2 hit 68.7% 顯示 shared marching 已抓到 stencil 重用；剩餘 DRAM 流量為必要的 plane 讀寫。
- 判斷依據：ncu DRAM% / Waves Per SM / occupancy。

## 4. 優化決策與結果

**不改動 kernel**（保持與 baseline 等價、正確性自然成立），記錄為近優化案例：
- 唯一明確槓桿是 **z-tiling**（把 k-march 切成多段 → 更多 block → 更多 waves → 降 tail），
  但需在每個 chunk 邊界**重新載入 k0-1 的 halo plane**（原 kernel 靠 k=0 為 Dirichlet 邊界
  才容許未初始化的 plane −1；chunk 起點 k0>0 時 plane k0-1 是內部點、必須正確載入），
  否則結果錯誤。此改動對這支緊湊的 marching kernel 索引邏輯風險高、預期僅 ~1.15×
  （消 19% tail，但增 chunk 邊界 ~1-2% 重載）→ **列為分析過的後續方向，本次不實作**。

## 5. 結果與驗證

未修改 → optimized 與 baseline 等價（128³ config 的內建 RMS/PASS 仍成立）。
記錄為 memory-bound（61.5% DRAM）、partial-wave-tail 限制的誠實案例。
baseline = `baseline/results/951758/laplace3d-cuda/run-{1..5}.log`（512³ ≈ 0.002154 s）。
