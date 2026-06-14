# miniWeather bottleneck 分析

執行參數：`./main`（NX=400, NZ=200, 編譯期固定；single-rank MPI）。需 `module load
openmpi/4.1.6_ucx1.14.1_cuda12.3`（用到 `mpi.h`/libmpi）。比較指標：自印
`Total main time step loop (sec)`。baseline ≈ **1.185 s**。內建 PASS（`d_mass` 守恆）。

## 1. Baseline 行為

有限體積天氣 mini-app。每個 timestep 跑多個小 kernel（compute_flux_x/z、compute_tend_x/z、
update_fluid_state、update_state_x/z、pack/unpack_send_buf…）。x 方向邊界用 **MPI halo 交換**：
`pack → cudaMemcpy D2H(sendbuf) → MPI_Isend/Irecv → cudaMemcpy H2D(recvbuf) → unpack`。

## 2. Profiling 證據（job 952755，nvprof）

- 所有 kernel 都很小（2–13 µs），但被呼叫上萬次（10800–21600）：update_fluid_state 26.6%、
  compute_flux_z 19.8%、compute_flux_x 17.8%、compute_tend_z 8.6% …
- **21600 HtoD + 21604 DtoH 小 memcpy（~13% GPU 時間）**＝每步的 MPI halo host round-trip。
- 「Total loop 1.185 s」中，GPU kernel 總和約 0.7 s → 約 40% 是 launch / memcpy / host 同步開銷。
- grid 400×200 很小 → 每個 kernel under-utilized（launch/overhead-bound）。

## 3. Bottleneck 判定

**launch / host-overhead-bound（小網格 + 每步大量小 kernel 與同步 host memcpy）**：
- 每步的 4 個同步 `cudaMemcpy`（2 D2H + 2 H2D）做 MPI halo 交換 → 卡住 pipeline。
- 對 single-rank（`./main` 無 mpirun → nranks=1），左右鄰居都是自己、週期性自交換，
  這個 host round-trip + MPI **完全是多餘的**。判斷依據：nvprof 的 memcpy 次數/佔比 + kernel 皆小。

## 4. 優化決策

**single-rank 的 x-halo 交換改走 device-to-device**：`pack` 後若 `left_rank==myrank &&
right_rank==myrank`，直接 `cudaMemcpy(d_recvbuf_l, d_sendbuf_r, D2D)` 與
`cudaMemcpy(d_recvbuf_r, d_sendbuf_l, D2D)`（對應 MPI tag 的 periodic swap），跳過 2 D2H +
2 H2D host copy 與 MPI/Waitall。多 rank 時保留原 MPI 路徑。數學等價（內建 PASS 驗證）。

## 5. 結果與驗證

正確性：optimized 3 次皆 **PASS**（質量守恆 `d_mass`）。

| | baseline | optimized | speedup |
|---|---:|---:|---:|
| Total main time step loop（3 次中位數） | 1.185 s | **0.711 s** | **1.67×** |

確認改善 ~1.67×：消除每步的同步 host memcpy 與 MPI 開銷，讓 timestep loop 不再被 host
round-trip 卡住。**推測剩餘成本**：上萬次小 kernel 的 launch overhead 與小網格 under-utilization
仍在（要再快需 kernel fusion 減 launch 數）。
baseline = `baseline/results/951758/miniWeather-cuda/run-{1..5}.log`。
