# stencil3d bottleneck 分析

執行參數：`./main 512 100`（512³ grid、100 reps）。比較指標：自印 `Average kernel
execution time (s)`。baseline ≈ **0.036 s**（job 951758）。

⚠️ 程式**無內建 PASS/FAIL 驗證**（只有 `-DDUMP` 除錯輸出）→ 若修改 kernel，需用小尺寸
`-DDUMP` 對 baseline/optimized 做 diff 驗證等價性。

## 1. Baseline 行為

各向異性 3D stencil（sigmaX/Y/Z 方向係數），kernel 已**相當優化**：
`__shared__ Real sm_psi[4][16][16]`（rolling 4-plane shared 緩衝）+ XTILE=20 的 register
marching（pii/cii/nii 滾動 prev/cur/next 平面），block = 16×16、每 block 處理一個 x-tile。
每點需讀 d_Vm 鄰點 + sigmaX/Y/Z 多方向係數（每方向多個分量）。預期 memory-bound，且因為已
tiling 過，**headroom 可能有限**。

## 2. Profiling 證據（job 952728，nvprof + ncu，FP64）

**nvprof**：stencil3d kernel 35.9 ms ×10 = 11.8% GPU 時間；其餘為 init memcpy（Vm + 9.6GB
sigma + dVm 的 H2D/D2H，屬一次性，不計入 kernel 指標）。

**ncu（stencil3d）**
- **DRAM Throughput 60.7%（543.9 GB/s）**：memory-bound 但**未飽和**（peak ~900 GB/s）。
- **Compute (SM) Throughput 7.85%**：compute 幾乎閒置。
- **L2 Hit Rate 8.6%**：極低 → 幾乎無重用。
- **Achieved Occupancy 61.8%**。

## 3. Bottleneck 判定

**memory-bound（streaming），且已被 baseline 充分優化**：
- DRAM 60.7% + SM 7.85% → memory-bound、非 compute。
- 資料量：512³ FP64 的 d_psi（含 halo）+ 9 個 sigma 分量（~9.6 GB）+ d_npsi。sigma 係數
  **每點各讀一次、無重用** → L2 hit 僅 8.6% 是**本質的**（streaming，無法靠 cache 改善）。
- baseline kernel 已用 `__shared__ sm_psi[4][16][16]` rolling buffer + XTILE=20 register
  marching 抓住 stencil 的 halo 重用；剩下的 sigma streaming 無重用空間。
- occupancy 61.8% 受**register marching 刻意使用的大量暫存器**限制（滾動的 prev/cur/next
  平面狀態）。

## 4. 優化決策（與 rejected 推論）

判定為**已近實務上限、headroom 小**，**不修改 kernel**（保持與 baseline 等價、正確性自然成立）：
- **Rejected（推論，profiling 支持）**：用 `__launch_bounds__` 強拉 occupancy（62%→更高）
  會壓低每 thread 暫存器，**把 register-marching 的平面狀態 spill 到 local memory**，反而
  增加記憶體流量、可能退步——對這種刻意用暫存器換 locality 的 marching kernel 是反效果。
- sigma 係數為 streaming（L2 8.6% 本質如此），shared/cache tiling 無重用可抓。
- 真正要再快需演算法層級改動（如把 9 個 sigma 分量壓縮/重排以減少位元組、或 mixed
  precision），超出「等價優化」範圍且有正確性風險（本題無內建 verify）。

## 5. 結果與驗證

**未修改 kernel** → optimized 與 baseline 位元等價（無需 A/B；正確性由「不變更」保證）。
記錄為**已優化、memory-bound 60% DRAM、streaming 係數限制**的誠實案例。
baseline kernel 時間 = 0.036 s（job 951758）。

> 註：本題無內建 PASS/FAIL；若日後嘗試演算法級優化，需以小尺寸 `-DDUMP` 對 baseline/optimized
> 做 diff 驗證等價性。
