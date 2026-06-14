# fdtd3d bottleneck 分析

執行參數：`--dimx=192 --dimy=184 --timesteps=90`（radius-4 FDTD，90 步）。
比較指標：自印 `Average kernel execution time`（baseline ≈ 0.000190 s）。內建 PASS。

## 1. Baseline 行為

NVIDIA SDK FDTD3d 範例：`finite_difference` kernel，已用 shared-mem tile
（`tile[blockY+2R][blockX+2R]`）+ register z-marching（`infront[R]`/`behind[R]`，R=4）。
block 32×8。每步 launch 一次（90 步）。

## 2. Profiling 證據（job 952750，ncu）

- nvprof：finite_difference 195 µs（其餘為 init memcpy）。
- ncu：DRAM 32.1%、Compute(SM) 30.3%、L2 72.6%、**Achieved Occupancy 21.58%**、
  Registers 48（Block Limit Registers = 5）、**Waves Per SM = 0.35**。

## 3. Bottleneck 判定

**問題規模太小、GPU 嚴重 under-utilized（0.35 waves）**：
- grid 僅 (192/32)×(184/8) = 6×23 = **138 個 block**，但 V100 可容 ~80×5 = 400 個 block slot
  → 連 0.35 個 wave 都填不滿 → DRAM(32%) 與 SM(30%) **皆未飽和**、occupancy 21.58%。
- 這是**空間平行度不足**（問題小、timesteps 又是序列相依無法平行）造成，**非 kernel 缺陷**。
- 判斷依據：ncu Waves Per SM 0.35 + block 數 138 + DRAM/SM 皆低 + occupancy。

## 4. 優化決策

**不改動 kernel**（NVIDIA 範例已 shared-tile + z-march 優化）。
- 減 register 以提高每 SM block 數**無效**：binding constraint 是「總 block 數只有 138」，
  本就填不滿，不是每 SM 容量不足。
- 無法在固定問題規模下創造更多空間平行度；timesteps 序列相依不可平行。
- 誠實記為 problem-size-limited / under-utilized 案例。

## 5. 結果與驗證

未修改 → 與 baseline 等價（內建 PASS 仍成立）。
baseline = `baseline/results/951758/fdtd3d-cuda/run-{1..5}.log`（≈ 0.000190 s）。
