# nbody bottleneck 分析

執行參數：`./main 16000 10`（16000 particles、10 steps）。FP32。
比較指標：程式自印 `Average Performance (GFLOPS)`（越高越好）與 `Total Time (s)`。
baseline：Total Time 0.0235 s、**2309 GFLOPS**（job 951758）。正確性：`Verify()` 印 PASS/FAIL。

## 1. Baseline 行為

三個 kernel（GSimulationKernels.hpp），每 step：
1. **`accelerate_particles`**：O(N²) all-pairs 重力。每 thread i 對所有 j 讀 `p[j]`
   （AoS `Particle` ≈ pos[3]+vel[3]+acc[3]+mass ≈ 40 bytes），算 dx/dy/dz、`rsqrtf`、累加 acc。
   - **無 shared-memory tiling**：所有 thread 都從 global 讀同一串 p[j]（靠 L1/L2 吸收）。
   - **AoS 浪費頻寬**：每次讀整個 40-byte Particle，實際只需 pos+mass = 16 bytes。
2. `update_particles`：O(N) 更新 vel/pos、寫 energy e[i]。
3. **`accumulate_energy<<<1,1>>>`**：**單一執行緒**序列加總 e[0..n-1] → 完全不平行（紅旗）。

工作量：accelerate O(N²)=2.56e8 interactions/step ×~20 flops ×10 ≈ 51 GFLOP；2309 GFLOPS
約 V100 FP32 peak(~15.7T) 的 15% → 有 headroom。

## 2. Profiling 證據（job 952721，nvprof + ncu）

**nvprof（3 steps）**
- `accelerate_particles` 2.28 ms ×3 = **81.9%** GPU 時間。
- **`accumulate_energy<<<1,1>>>` 472 µs ×3 = 17.0%**！單執行緒序列規約竟佔第二大。
- `update_particles` 7 µs（可忽略）。

**ncu（accelerate_particles）**
- **DRAM Throughput 0.03%**、**L2 Hit 98.8%** → 完全非 memory-bound（p[j] 全被 L2 吸收）。
- **Compute (SM) Throughput 27%**、**Issued IPC 1.05**（低）、**Achieved Occupancy 12.4%**。
- 12.4% ≈ 每 SM 僅 1 個 256-thread block → **register-limited**（baseline `auto pi=p[i]`、
  `auto pj=p[j]` 各複製整個 40-byte Particle → 暫存器爆量）。

## 3. Bottleneck 判定

兩個獨立瓶頸：
1. **`accelerate_particles`：occupancy-limited（latency-bound）**，非 memory 也非 compute
   飽和。證據：DRAM 0.03% + SM 27% + occupancy 12.4% + IPC 1.05 → warp 太少、掩蓋不了
   rsqrt/相依鏈延遲。根因是整個 Particle struct 複製造成的高 register 壓力。
2. **`accumulate_energy<<<1,1>>>`：序列化**，單 thread 加總 16000 個 → 占 17%。

## 4. 優化決策

1. **accelerate_particles → shared-memory float4 tiling**：每個 block 協同把一個 tile 的
   `(pos.x,pos.y,pos.z,mass)` 載入 shared（16 bytes/粒子，coalesced），block 內所有 thread
   重用；只保留 pos+mass、不複製整個 struct → **降 register → 升 occupancy → 掩蓋延遲**，
   同時把 p[j] 的 global 讀取轉成 shared。數學與 baseline 等價（含 self/padding 貢獻 0）。
2. **accumulate_energy → 平行規約**（單 block 256 threads：grid-stride 載入 + shared tree
   reduce，thread 0 寫 e[0]），把 472µs 降到 ~µs 級。

## 5. 結果與驗證

正確性：smoke（job 952725）PASS；compare（job 952726）PASS（`Verify()` 比對 CPU 參考）。
baseline = `baseline/results/951758/nbody-cuda/run-{1..5}.log`，optimized = job 952726，
皆取 5 次中位數。

| 指標 | baseline | optimized | speedup |
|---|---:|---:|---:|
| Average Performance (GFLOPS) | 1956.8 | **3916.5** | **2.00x** |
| Total Time (s) | 0.02771 | 0.01396 | **1.98x** |

確認改善 **~2.0×**。兩個對策皆生效：
- accelerate_particles：shared float4 tiling + 只保留 pos+mass → 暫存器大幅下降、
  occupancy 從 12.4% 提升，掩蓋 rsqrt 延遲（這是主要 ~82% 工作量的提速來源）。
- accumulate_energy：序列 `<<<1,1>>>`（472µs、17%）→ 單 block 平行規約，降到 µs 級。

**推測為何約 2×（非更高）**：optimized 後 accelerate 仍是 O(N²) 的本質計算量，且 V100
FP32 ~3.9 TFLOPS 約 peak 的 25%——進一步要 register blocking（每 thread 算多個 i 的
micro-tile 提高資料重用與 ILP），但已達穩定 2×、且 N=16000 規模下 occupancy 提升的邊際
效益遞減。屬確認改善。
