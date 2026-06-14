# convolution3D bottleneck 分析

⚠️ **比較基準修正**：committed baseline（job 951758）跑的**不是** Makefile 的
`32 1 6 32 32 5 100`（tiny），而是兩個真實 conv 層：
- config1（small）：`32 6 16 14 14 5`（C=6,M=16,14×14 → out 10×10）→ ~17.6 us
- config2（heavy）：`32 96 256 26 26 5`（C=96,M=256,26×26 → out 22×22）→ **~13554 us**

優化與比較以 baseline 實際參數為準（heavy config2 是主要目標）。FP32 only。
比較指標：各程式自印的 `conv3d_s{1,2,3}` 平均時間（取 warmup 後那組）。

## 1. Baseline 行為

直接 2D 卷積，三個 kernel `conv3d_s1/s2/s3` 演算法相同，只差 block/grid 維度到
(n, m, tile) 的映射。block = 16×16，每 thread 算 1 個 output：
```
for c in C: for p in K: for q in K: s += X[II(n,c,h+p,w+q)] * W[WI(m,c,p,q)]
```
filter `W` 在 **global memory**（heavy config 大小 M·C·K·K = 256·96·25 = 614,400 floats
= **2.4 MB**，遠超 constant memory 64KB → 不能用 constant）。

heavy config2 規模：output N·M·Hout·Wout = 32·256·22·22 ≈ 3.96M、每 output C·K·K = 2400 MAC
→ 每個 kernel 約 19 GFLOP-MAC。grid s1 = (32,256,4) = 32768 blocks（51 waves，無 tail）。

## 2. Profiling 證據（job 952657，heavy config2，nvprof + ncu）

**nvprof**：conv3d_s1/s2/s3 各約 13.2 / 13.2 / 14.1 ms（1002 launches 平均），三者接近；
memcpy 僅 0.06%。

**ncu（conv3d_s1，grid (32,256,4)×(16,16)）**
- **DRAM Throughput 1.4%**、**L1/TEX Hit 94.7%**、**L2 Hit 97.1%** → X/W 重複讀幾乎全被
  cache 吸收，**不是 memory-bound**。
- **Compute (SM) Throughput 79.7%**、Achieved Occupancy 94.9%、Waves Per SM 51 →
  **compute / instruction-bound、occupancy 與 wave 充足**。
- 注意：~13.5ms 做 19 GFLOP 只有 ~1.4 TFLOP/s（peak FP32 ~15.7T 的 ~9%）；SM 80% 是
  **整體管線（含位址整數運算與 load）忙碌**，並非 FMA 飽和 → 內層每次重算 `II`/`WI`
  的整數位址指令稀釋了 FMA 比例。

## 3. Bottleneck 判定

**compute / instruction-bound**（非 memory-bound）：
- DRAM 1.4% + cache hit 94–97% → 排除 memory-bandwidth-bound。
- SM 79.7%、occupancy 95%、51 waves → 受指令吞吐限制，主因是**每個 FMA 伴隨大量
  位址整數運算 + 從 L1 讀 W**。
- 判斷依據：ncu DRAM% / cache hit / SM% / Waves / occupancy。

（與先前對 tiny config 的判定不同：tiny config 是 1.2-wave tail-bound；heavy config 才是
baseline 真正量測、且 compute-bound 的工作量。）

## 4. 優化決策

1. **filter slice W[m] → shared memory**：每個 block 的 m 固定（s1: blockIdx.y、s2: .x、
   s3: .z），block 內 256 threads 共用同一份 W[m]（C·K·K floats；config2=9.6KB、
   config1=600B，皆 < 48KB）→ 一次協同載入 shared，內層改讀 shared，消除 W 的 L1 流量
   與 tag 開銷。
2. **編譯期 K 展開（template on KK）+ 索引 strength reduction**：每個 c 先算 `xbase`、
   `wbase` 指標，內層 KK×KK 只用編譯期常數偏移 `p*Win+q` / `p*KK+q` → 大幅減少位址整數
   指令，提高 FMA 佔比。
3. KK=0 runtime fallback 保留給非常見 K。
4. 不用 constant memory（W 2.4MB 超過 64KB）。X shared tiling 暫不做（跨 96 channel 的
   tile 放不進 shared，且 L1 已 94% 命中）。

## 5. 結果與驗證

正確性：smoke（job 952668）12/12 PASS、compare（job 952670）兩 config 全 PASS。
baseline = `baseline/results/951758/convolution3D-cuda/run-{1..5}.log`（各 config warmup 後數字）；
optimized = job 952670，皆取 5 次中位數。

| config | kernel | baseline | optimized | speedup |
|---|---|---:|---:|---:|
| small (C=6, 14×14) | conv3d_s1 | 17.63 us | 11.06 us | **1.59x** |
| small | conv3d_s2 | 19.23 us | 11.40 us | **1.69x** |
| small | conv3d_s3 | 17.58 us | 10.97 us | **1.60x** |
| heavy (C=96, 26×26) | conv3d_s1 | 13.61 ms | 9.98 ms | **1.36x** |
| heavy | conv3d_s2 | 13.13 ms | 9.89 ms | **1.33x** |
| heavy | conv3d_s3 | 14.10 ms | 10.03 ms | **1.41x** |

確認改善：shared-W + 編譯期 K 展開 + 索引 strength reduction，**small 1.6–1.7×、heavy 1.33–1.41×**。

**為何 heavy 改善幅度小於 small（推測）**：heavy config 的 baseline SM throughput 已達 ~80%、
FMA 工作量大（每 output 2400 MAC），屬「已相當 compute-飽和」——優化主要省下位址整數指令
與 W 的 L1 流量，但無法縮減本質的 FMA 量，故改善約 1.35×。small config 的固定/位址開銷
相對 FMA 比例更高，移除後收益更大（~1.65×）。要再進一步（heavy）需 register tiling
（每 thread 算多個 output 重用暫存器中的 X/W）把 FMA:load 比例再拉高——列為後續方向。

剩餘 headroom（heavy）：優化後仍 ~1.9 TFLOP/s（peak 的 ~12%），距 compute roofline 仍遠，
代表仍是 instruction/issue-bound；register tiling 應可再有斬獲（未做）。
