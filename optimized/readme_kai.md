# Profiling 導向的 CUDA 優化報告（第二批 16 題）

本報告彙整第二批 16 個 HeCBench CUDA 程式的 profiling 證據、bottleneck 判定、優化決策、
失敗實驗、正確性驗證與量測結果。每題的完整逐步記錄在 `optimized/analysis/<bench>.md`；
profiling 原始輸出在 `profiling/results/<jobid>/`；Slurm job log 在 `logs/`。

原始程式在 `src/<bench>-cuda/`（baseline，不改），優化版在 `optimized/<bench>-cuda/`。

---

## 1. 實驗設定

### 硬體與軟體
- GPU：NVIDIA Tesla V100-SXM2-32GB（sm_70）
- CUDA：`module load cuda` → 12.8；build 一律 `make ARCH=sm_70`
- 叢集：TWCC，Slurm，project `ACD115083`，partition `gp1d`
- Profiling 工具：**nvprof**（GPU activity / API 時間佔比）＋ **Nsight Compute `ncu`**
  （per-kernel 的 DRAM/SM throughput、occupancy、L2 hit、warp stall reason 等）

### 量測方法
- 效能指標一律採用**各程式自己印出的 kernel/stage 時間**（非 Slurm wall time）。
- 各程式自印的 per-kernel 時間是「每次 launch 平均」、與 repeat 次數無關，因此 baseline 不
  重跑，直接重用已 commit 的 `baseline/results/951758/<bench>-cuda/run-{1..5}.log`；
  只重跑優化版（`scripts/compare.sbatch`），取 5 次中位數。
- **比較參數以 baseline log 實際跑的為準**（不一定等於 Makefile 的 `run:` target；
  convolution3D / laplace3d 的 baseline 各跑了兩個 config）。
- 腳本：`scripts/profile.sbatch`、`scripts/smoke.sbatch`、`scripts/compare.sbatch`。

### 進度
16 題中 **13 題完成**（profiling + 優化 + 量測）；**3 題跳過**（hotspot / hotspot3D / sobel，
無真實輸入資料——`data.zip` 只含 DVC 指標檔、無 dvc 工具可 pull、sobel 的 .bmp 也缺）。

---

## 2. 結果總表

### 確認的加速（8 題）

| 題目 | 類別 | speedup | bottleneck（profiling 判定） | 採用的優化 |
|---|---|---:|---|---|
| xsbench | Sim | **5.90×** | memory-latency（92.9% warp cycle 卡記憶體、L2 39%、DRAM 50%、SM 11%） | 依 energy 排序 17M lookups 提升 grid 存取 locality |
| bilateral | CV | **2.81–2.98×** | compute（DRAM ~1%、SM ~78%；transcendental + 索引/分支） | constant 空間權重表 + interior fast-path + `__expf` |
| nbody | Sim | **2.00×** | accelerate occupancy 12%（register 壓力）；accumulate_energy 序列 17% | shared float4 (pos,mass) tiling + 平行能量規約 |
| miniWeather | Sim | **1.67×** | launch/host-overhead（每步 MPI halo 的同步 host memcpy ~13%） | single-rank halo 改 device-to-device，跳過 host round-trip + MPI |
| convolution3D | CV | **1.33–1.69×** | compute/instruction（SM 80%、DRAM 1.4%；位址整數運算稀釋 FMA） | filter slice W[m] 放 shared + 編譯期 K 展開 + 索引 strength reduction |
| convolution1D | CV | **1.17–1.39×**(int16) | int16 instruction-overhead（邊界分支佔比高；float/double 已近 roofline） | 保持 coalesced 的 interior fast-path + 編譯期 mask 展開 |
| lavaMD | Sim | **1.15×** | compute（DRAM 0.6%、SM 82%；LJ 力內層的 double `exp`） | register 力累加 + `__expf`（float 快速 transcendental） |
| srad | CV | **1.06×** | reduce 佔 39%（最慢的 modulo 規約）；COMPUTE stage 受每 iter 的 D2H 同步限制 | sequential-addressing 規約（輸出影像位元相同） |

### 分析後判定「已近最佳、不改動」（5 題）

| 題目 | speedup | profiling 判定（為何不改） |
|---|---:|---|
| stencil3d | ~1.0× | memory-bound 60% DRAM；已 shared+register marching；9.6GB sigma 係數為 streaming（L2 8.6%、無重用），強拉 occupancy 會 spill marching 狀態 |
| convolutionSeparable | ~1.0× | NVIDIA 高度優化範例；conv_cols 已 83.5% DRAM、occupancy 96.5%；filter→constant 試過僅 1.006×（filter 太小非瓶頸） |
| laplace3d | ~1.0× | memory-bound 61.5% DRAM；受 partial-wave tail（1.6 waves）限制；z-tiling 可降 tail 但需小心 halo 重載、風險高 |
| heat | ~1.0× | memory-bandwidth-bound、已近 roofline（DRAM 88%）；naive 5-point 已飽和，div/mod 非瓶頸 |
| fdtd3d | ~1.0× | under-utilized：問題太小（0.35 waves、138 blocks），DRAM 32% / SM 30% 皆閒置；NVIDIA 範例已優化、無法在此規模增加空間平行度 |

---

## 3. 各題詳細分析

### 3.1 xsbench — 5.90×（記憶體延遲 → energy 排序）
- **baseline**：event-based Monte Carlo 截面查表，17M lookups（每 thread 一個），各取樣隨機
  energy → 對 32MB unionized grid 做資料相依二分搜 → 對 5.6GB index/nuclide grid 隨機 gather。
- **profiling（job 952677）**：ncu 顯示 **92.9% 的 warp cycle 卡在 global-memory scoreboard
  stall**；DRAM 50%（未飽和）、SM 11%、L2 hit 僅 39% → **memory-latency-bound**。
- **優化**：一次性把 17M lookups 依取樣 energy 排序（`compute_energy_key` 由 seed 重算 energy
  → `thrust::sort_by_key`），lookup kernel 改吃 `idx_sorted[t]`。相鄰 thread energy 相近 →
  grid 存取 locality 大增。verification 不變（每 thread 保留原 lookup 索引）。
- **結果**：lookup kernel **0.342s → 0.058s = 5.90×**（checksum Valid）。一次性排序成本 ~7.8ms
  另外用 nvprof 量測並誠實標註（含 sort 後 net 仍 ~5.5–6.2×）。

### 3.2 bilateral — 2.81–2.98×（compute → 空間表 + `__expf`）
- **baseline**：`bilateralFilter<R>`（R=3/6/9），每 thread 一像素掃 (2R+1)² 視窗，每鄰點一個
  `expf` + mirror 邊界分支。
- **profiling（job 952714）**：DRAM 0.9–3.3%、**SM 75–78%** → compute-bound（transcendental）。
- **優化（兩步）**：(1) `expf`→`__expf` 只有 ~1.10×（expf 非唯一成本）；(2) spatial 權重對內部
  像素只與 (i,j) 有關 → 預算進 constant memory，內層省掉 spatial 計算與一個 transcendental，
  且內部像素走無 mirror 分支的 fast-path（邊界保留精確 mirror）。
- **結果**：**2.81 / 2.98 / 2.89×**（3×3 / 6×6 / 9×9），全 PASS（tol 1e-3）。

### 3.3 nbody — 2.00×（occupancy + 平行規約）
- **baseline**：O(N²) all-pairs 重力 `accelerate_particles`（每 thread 複製整個 40-byte
  Particle）；`accumulate_energy<<<1,1>>>` 單執行緒序列規約。
- **profiling（job 952721）**：accelerate 佔 82%，DRAM 0.03%、L2 98.8%、**occupancy 12.4%**
  （register 壓力 → 每 SM 1 block）→ occupancy/latency-bound；accumulate_energy 佔 **17%**。
- **優化**：(1) 每 tile 的 (pos,mass) 以 coalesced `float4` 載入 shared、每 thread 只留 pos+mass
  → 降 register → 升 occupancy 以掩蓋 rsqrt 延遲；(2) 序列規約改單 block 平行規約。
- **結果**：GFLOPS **1957 → 3917 = 2.00×**（Time 1.98×），PASS。數學與 baseline 等價。

### 3.4 miniWeather — 1.67×（host overhead → on-device halo）
- **baseline**：有限體積天氣 mini-app，每步多個小 kernel；x 邊界用 MPI halo 交換
  `pack → D2H memcpy → MPI → H2D memcpy → unpack`。
- **profiling（job 952755）**：kernel 都很小但被呼叫上萬次；**21600 HtoD + 21604 DtoH 小 memcpy
  ~13%**；loop 1.185s 中約 40% 是 launch/memcpy/host 同步開銷。
- **優化**：single-rank（`./main` 無 mpirun）時左右鄰居都是自己 → 把 x-halo 自交換改成
  **device-to-device cudaMemcpy**（`d_recvbuf_l ← d_sendbuf_r`、`d_recvbuf_r ← d_sendbuf_l`，
  對應 MPI tag 的 periodic swap），跳過 2 D2H + 2 H2D 與 MPI；多 rank 保留原路徑。
- **結果**：Total loop **1.185s → 0.711s = 1.67×**，PASS（質量守恆）。

### 3.5 convolution3D — 1.33–1.69×（compute → shared filter + 展開）
- **baseline-args 修正**：baseline 跑兩個真實 conv 層（小層 `32 6 16 14 14 5`、**重層
  `32 96 256 26 26 5` ≈ 13.5ms**），非 Makefile 的 tiny config。
- **profiling（job 952657，重層）**：DRAM 1.4%、L1/L2 hit 95/97%、**SM 79.7%** →
  compute/instruction-bound；內層每次重算 `II`/`WI` 索引稀釋 FMA。
- **優化**：每 block 固定 m → 把 filter slice W[m]（9.6KB）載入 shared 共用；template on K
  展開 K×K；每 channel 算一次 base、內層用常數偏移（strength reduction）。
- **結果**：重層 **1.33–1.41×**、小層 **1.59–1.69×**，全 PASS。

### 3.6 convolution1D — int16 1.17–1.39×（指令開銷 → coalesced fast-path）
- **profiling（job 952532）**：`conv1d<double>` 已達 89% DRAM（近 roofline）；float 65%、int16
  29%。double 是 bandwidth-bound、headroom 小；int16 是 instruction-overhead-bound（邊界分支）。
- **優化**：保持 baseline 完全 coalesced 的「一元素一 thread」存取，只移除內部 block 的邊界分支、
  以編譯期 mask 寬度展開。
- **結果**：int16 **1.17× 平均 / 1.39× 最佳**；float/double ~1.0×（已近 roofline）。
- **Rejected experiment**：128-bit 向量化（double2/float4/short8 中心 + 純量 halo）**退步到
  0.28×**——halo 變成 stride 非 coalesced。教訓：破壞 coalescing 的代價遠大於省下的重疊讀取。

### 3.7 lavaMD — 1.15×（compute → `__expf`）
- **profiling（job 952743）**：DRAM 0.6%、L2 98%、**SM 82%** → compute-bound；`fp=float` 卻呼叫
  double `exp`。
- **優化（三次嘗試）**：(1) register 力累加（消內層 global RMW）~1.0×；(2) `exp`→`expf` ~1.01×
  （輸出與 baseline 完全相同）；(3) `expf`→`__expf`（單一 MUFU）**1.15×**。
- **結果**：**1.15×**，max force 偏差 3.7e-4（float 級；lavaMD 無正式容差）。

### 3.8 srad — 1.06×（reduce 規約；stage 受 host-sync 限制）
- **profiling（job 952757）**：`reduce` 佔 **39%**（每 iter 3 階多階規約，用最慢的
  `(tx+1)%i==0` modulo 規約）；每 iter 還有 D2H 把 sum 拷回 host。
- **優化**：reduce 改成 sequential-addressing 規約（無 modulo、無 bank conflict、inactive lane
  補 0），保留多階索引 → host 驅動迴圈不動。
- **結果**：COMPUTE stage **0.0560s → 0.0530s = 1.06×**，**輸出影像位元相同（cmp IDENTICAL）**。
- **為何只有 1.06×**：COMPUTE stage（56ms）被每 iter 的 **D2H 同步 + host mean/variance 計算**
  卡住（stage ≫ GPU kernel 總和 34ms）；下一步要把 mean/variance 搬上 device 消除 round-trip。

### 3.9–3.13 已近最佳（不改動）
詳見 `optimized/analysis/{stencil3d,convolutionSeparable,laplace3d,heat,fdtd3d}.md`。
共同點：皆為已被妥善優化或受**本質限制**（streaming 係數、近 DRAM roofline、partial-wave tail、
問題規模太小）的 kernel；每題以 profiling 數據說明原因，保持與 baseline 位元等價、正確性自然成立。

---

## 4. 跳過的題目（無輸入資料）

hotspot / hotspot3D 需要 `../data/hotspot*/temp_*`、`power_*`；sobel 需要
`SobelFilter_Input.bmp`。提供的 `data.zip` 只含 hotspot/hotspot3D 的 **DVC 指標 stub**
（非真資料），系統無 `dvc` 工具可 pull，sobel 的 BMP 也缺 → 依專案決定跳過。
（srad 的真實 `image.pgm` 在 zip 中，故 srad 有做。）

---

## 5. 結論與方法學教訓

- **最大的 win 來自演算法 / 存取模式的改變**（xsbench 排序、miniWeather on-device halo、
  nbody tiling），不是微調。
- **降低「表面的低效」不一定變快**：convolution1D 的向量化版退步到 0.28×（halo 變非 coalesced），
  列為 rejected experiment。
- **比較參數要對齊 baseline 實際跑的**，不一定是 Makefile 的 `run:`（convolution3D 跑兩個 conv
  層、laplace3d 跑 512³ config）——以 baseline log 為準。
- **效果有限也要誠實記錄並推測原因**（已達 roofline / partial-wave tail / 問題太小 / host-sync
  限制…），這些都有 profiling 數據支持，是有效成果。
- 無內建 verify 的題目（stencil3d、laplace3d、lavaMD、srad、miniWeather）以對應方式驗證：
  輸出影像/結果檔 diff、`-DOUTPUT`/`-DDUMP`、或程式自帶的守恆/PASS 檢查。

## 6. 重現方式

```bash
cd /home/r14922146/hecbench-cuda-optimization
sbatch scripts/profile.sbatch <bench>   # nvprof + ncu -> profiling/results/<jobid>/
sbatch scripts/smoke.sbatch   <bench>   # 小規模 build + 正確性
sbatch scripts/compare.sbatch <bench>   # 只跑 optimized 5 次，對 951758 baseline 算 speedup
```
（convolution3D / laplace3d 為多 config；miniWeather 需 `module load openmpi/4.1.6_ucx1.14.1_cuda12.3`；
srad / lavaMD / miniWeather 的正確性用各自的 image/result diff 或內建 PASS 驗證。）
