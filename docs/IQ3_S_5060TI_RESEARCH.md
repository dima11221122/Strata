# IQ3_S on a 16 GB RTX 5060 Ti

This fork investigates 90 tokens/s for one request using the original Qwen3.8-Flash-Next IQ3_S weights. The highest observed three-prompt median so far is 70.2 tokens/s, from an admission-overlap candidate whose gain did not repeat. The target has not been reached. This document records hypotheses and measurements; it does not promise that this hardware can reach the target.

## Measurement contract

- Hardware: RTX 5060 Ti 16 GB, PCIe Gen4 x8, EPYC 7K62, 23 vCPUs, 98 GiB RAM, Linux, CUDA 13.
- Upstream engine: `6f32ec070f23ced9f50e704d854d775da52591ab`.
- GGUF revision: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_S, both shards including PLE.
- One request at a time. Three distinct prompts: Python LRU cache, database recovery, TypeScript task queue. Each generates 512 tokens, greedy, reasoning off. Use the median of the engine's decode rates, with each individual rate recorded. Repeat promising candidates and use held-out prompts.
- Keep authentication, 65,536-token capacity, tool calls, and both model shards. Report separate numbers for reduced context or changed weight quantization; those do not satisfy the original target.
- Profiles and credentials stay outside the checkout. GPU event profiling changes scheduling, so its rate is diagnostic only.
- The [sanitized result ledger](IQ3_S_5060TI_RESULTS.json) records valid timestamped runs, exact benchmark prompts, executable hashes, output hashes, draft acceptance, and observed clocks. It omits credentials and generated answers.

## Results before engine changes

| Candidate | Decode rates, tokens/s | Median |
|---|---|---|
| Installer defaults | 53.3, 52.6, 61.0 | 53.3 |
| Native calibration: PCIe fraction 0.20, draft probability floor 0.70 | 67.7, 62.4, 75.9 | 67.7 |
| Cache reordered from six separate coding prompts | 65.6, 63.1, 75.0 | 65.6 |

The workload cache was reverted. The calibrated CPU pool uses 22 workers; reducing it to 15 or 11 was slower in the native calibration.

## Research and applicability

1. [Fiddler, ICLR 2025](https://arxiv.org/abs/2402.07033): model the latency of expert computation on CPU versus copying its weights and computing on GPU. Its decision depends on the number of inputs per expert. Strata already splits CPU/GPU experts, but its PCIe share chooses missed experts in routing order. Test whether choosing by reuse and per-layer cost improves the critical path. Fiddler's reported speedups use different, uncompressed models and are not predictions for this server.
2. [SpecMoEOff](https://arxiv.org/abs/2508.21706): reuse weights across speculative tokens and choose speculation length using measured draft acceptance and execution costs. Its throughput experiments also use large request batches, which are outside this target. Applicable here: tune single-request window depth and CPU/GPU balance together; retain target-model verification. Its CPU attention strategy is less compelling for this Qwen model at short context, where attention is a small measured stage.
3. [MoE-SpeQ](https://arxiv.org/abs/2511.14102): lookahead-aware expert scheduling and a governor trade additional speculative work against memory and transfer costs. Its draft is a quantized full MoE model. Strata's MTP is one MoE draft block and does not produce expert routes for the target's 48 layers. Replacing it with a second full MoE draft would add substantial memory and complexity. Applicable here: measure the draft precision, acceptance, and VRAM tradeoff while retaining target verification. The paper's A100 40 GB and PCIe x16 results are not predictions for this server.
4. [PowerInfer](https://arxiv.org/abs/2312.12456): exploit hot activation locality and heterogeneous placement. Its sparse-neuron predictor is model-specific; it cannot simply be applied to this MoE checkpoint. Applicable here: measure expert popularity and per-layer cost, and test placement under the actual VRAM budget.
5. [T-MAC](https://arxiv.org/abs/2407.00088): arrange low-bit operands for register lookup instructions and control lookup-table storage. Its bit-plane product algorithm is not a direct replacement for these IQ codebooks. Applicable idea: retain exact codebook coefficients but encode each signed coefficient as a nibble, then use AVX2 byte shuffles instead of scattered scalar table loads.
6. [LUT-GEMM](https://arxiv.org/abs/2206.09557) and [FLUTE](https://arxiv.org/abs/2407.10960): reduce dequantization work and restructure lookup-quantized weights. Their GPU formats and arithmetic differ from Strata's native blocks. The CPU trial below takes the layout idea without changing checkpoint precision; the earlier CUDA shared-codebook trial was slower here.
7. [DALI](https://arxiv.org/abs/2602.03495): use workload-dependent CPU/GPU timings for expert assignment, residual-corrected prefetching, and cache updates based on workloads accumulated over a token window. Its transfer/compute pipeline model assumes useful overlap. The mapped fetch and resident-kernel delays measured here must be incorporated before reusing that model; this fork's earlier uncalibrated greedy selector failed. Its EPYC 7532, RTX 3090, PCIe x16 and multi-request results are not speed predictions for this IQ3_S single-request workload.
8. [Efficient CPU-GPU Collaborative Inference](https://arxiv.org/abs/2512.16473): compute cache misses on CPU while asynchronously loading experts into GPU cache for future tokens. This targets single requests, but Strata already has asynchronous adaptive admission. Disabling adaptation measured 56.4 tokens/s in the earlier sweep. Any new admission policy needs measured savings beyond the existing mechanism; copying the paper's cache primitive alone is not a new optimization.

## Experimental sequence

- [x] Fork upstream and preserve a calibrated baseline.
- [x] Compare a workload cache profile on separate benchmark prompts.
- [x] Enable CPU/GPU decode stage timings and record wait, pool, draft and dense-kernel costs.
- [ ] Compare speculative window depths 2, 4, 6, 8 and confidence floors, without changing target verification.
- [ ] Compare INT8 and Q4 KV, and 32K versus 20K resident KV while keeping 64K total capacity. Check long-context retrieval before retaining lower precision.
- [x] Compare transfer modes and CPU codebook gather/prefetch settings. Each run changes one factor from baseline.
- [x] Test expert input-reuse placement against routing-order placement with a pure scheduler test and native expert parity checks.
- [x] Profile a dominant CUDA kernel, test an opt-in projection implementation with independent numerical checks and microbenchmarks, then measure complete decode.
- [ ] Repeat the best configuration, run held-out prompts, long-context retrieval, and real Pi tool execution. Publish every result, including regressions.

## Initial profiling

Instrumented windows averaged 2.12–2.89 accepted tokens. Aggregate CPU expert work was 9.84–11.51 ms/window, GPU-reach waits 17.34–18.81 ms/window, and drafting 2.66–3.23 ms/window. VRAM expert hits were about 73%, and PCIe experts about 3% of all routed entries. These categories overlap and must not be added to infer a hardware ceiling. Event profiling disables Strata's separate shared-expert stream, so final speed runs must disable profiling.

## Fork experiments

All new engine paths are opt-in. The calibrated default keeps the original arithmetic and placement policy. Config capacity remains 65,536 tokens; the following runs use short prompts.

| Candidate | Decode rates, tokens/s | Median |
|---|---|---|
| Fork control, first sweep start | 61.9, 60.5, 75.3 | 61.9 |
| Reuse-based PCIe selection | 67.6, 62.1, 74.8 | 67.6 |
| BF16 tensor-core projections | 63.4, 64.0, 74.6 | 64.0 |
| Q8_0 draft experts | 62.8, 59.0, 70.4 | 62.8 |
| Q8_0 draft experts, speculation 6 | 57.5, 55.4, 66.6 | 57.5 |
| Fork control, first sweep end | 64.1, 60.6, 74.1 | 64.1 |
| Fork control, second sweep start | 52.3, 60.1, 74.4 | 60.1 |
| Reuse-based PCIe selection, repeat | 69.1, 61.3, 78.6 | 69.1 |
| Reuse selection, 450 MiB reserve, 20K resident KV | 58.5, 62.1, 74.7 | 62.1 |
| CPU multi-token kernels for single-token groups | 65.6, 60.7, 72.9 | 65.6 |
| CPU prefetch disabled | 64.7, 59.7, 74.4 | 64.7 |
| CPU prefetch 4096 bytes | 64.1, 59.0, 74.7 | 64.1 |
| Equal per-layer expert cache | 57.1, 63.1, 71.4 | 63.1 |
| Fork control, second sweep end | 66.0, 61.1, 74.9 | 66.0 |

These are separate completions, not identical token replays. Greedy output can change with expert placement and group arithmetic, as upstream's `STRATA_IQ_MT_MIN` documentation explains. Repeat controls and held-out checks are required before a final deployment. No speedup percentage is inferred from one unusually slow control.

### Further comparisons

The next sweep used the rebuilt executable with SHA-256 `5752617309eb869a45ad4e32cce3d3f84557d689a6aec8f1021faed91df00ef4`. It completed at 09:40 UTC on 2026-10-05.

| Candidate | Decode rates, tokens/s | Median |
|---|---|---|
| Control, sweep start | 63.4, 56.0, 67.2 | 63.4 |
| Requested clock locks, reuse selection | 55.7, 54.2, 66.3 | 55.7 |
| Requested clock locks, PCIe fraction 0 | 55.8, 58.1, 70.6 | 58.1 |
| Control after clock reset | 64.9, 60.8, 74.2 | 64.9 |
| Q5_0 draft experts | 47.9, 44.9, 61.6 | 47.9 |
| Q5_0 draft experts, speculation 6 | 58.0, 58.3, 69.1 | 58.3 |
| Reuse selection, PCIe fraction 0.10 | 53.4, 64.1, 74.3 | 64.1 |
| Reuse selection, PCIe fraction 0.35 | 51.9, 50.0, 60.3 | 51.9 |
| GPU codebook staging disabled | 62.4, 57.7, 72.3 | 62.4 |
| Draft probability floor 0.85 | 64.3, 60.8, 74.2 | 64.3 |
| Control, sweep end | 49.4, 52.6, 71.9 | 52.6 |

The higher floor increased draft acceptance to 89.1%, but did not improve throughput. Q5 acceptance was 80.7% at depth 4 and 74.3% at depth 6. Its extra memory reduced the target expert cache to about 4099 entries. Neither Q5 draft experiment improved the best median.

Clock commands requested 3090 MHz graphics and 14001 MHz memory after model readiness, but the recorded loaded clocks were 2730/13801 MHz. These numbers therefore describe the observed driver behavior, not a verified run at the requested maximum clocks. Power limits stayed at the stock 180 W, and locks were reset after every experiment and on sweep exit. Earlier clock experiments applied locks before service restart and were invalidated.

### AVX2 single-input experiment

A CPU-only probe used 16 real IQ3_S expert gate/up pairs with Q8_K inputs. A fused gate/up loop preserved all tested output bits but was 11–15% slower for two to four inputs, so it was rejected. Disabling loop unrolling did not produce a useful gain either.

On the measured Zen 2 host, the existing single-input IQ256 specialization took about 1350 microseconds per expert, versus about 660 microseconds for GGML. Calling the existing two-input specialization with the same input twice took about 550 microseconds and reproduced the single-input IQ256 output bits. This microbenchmark uses one CPU thread and does not predict a model speedup.

`STRATA_IQ256_PAIR_SINGLE=1` tests that scheduling for single-input IQ3_S gate/up groups on the active AVX2 path. It discards the second output, uses bounded scratch through row 640, and retains the previous specialization for larger row ranges. Other group sizes and formats stay on their existing paths. The default remains disabled. Enabling it switches single-input gate/up from GGML to the existing multi-token reduction order, so final answer quality must be checked.

| Candidate | Decode rates, tokens/s | Median |
|---|---|---|
| Control, sweep start | 55.1, 59.7, 75.4 | 59.7 |
| Paired single-input IQ3_S | 65.6, 59.7, 76.3 | 65.6 |
| Reuse selection, repeat | 61.1, 57.1, 73.6 | 61.1 |
| Paired single input with reuse selection | 63.0, 61.1, 73.1 | 63.0 |
| Control, sweep end | 63.4, 52.0, 63.1 | 63.1 |

This sweep used executable SHA-256 `912558effb2c2848ad805e6eedf8fa59090f31f9acd824dde828a2e60071f223`. Baseline variability remains substantial. These results do not establish the 90 tokens/s target or a stable speedup percentage.

### Lossless CPU lookup-code trial

A throwaway probe read 16 real gate/up expert pairs for each native GU format in the IQ3_S checkpoint: IQ3_XXS (18), IQ3_S (21), and IQ2_S (22). Each 256-value block keeps the original FP16 scale, its exact sub-scales, and 128 bytes of nibble codes. The 16-entry signed lookup alphabet contains the original integer coefficients. The derived block is 146 bytes, larger than the original 98/110/82 bytes, and is never sent to the GPU.

Seven alternating timing rounds on one pinned CPU core gave these median kernel speedups over the actual default dispatch (GGML at NT1, IQ256 at NT2–4):

| GU format | NT1 | NT2 | NT3 | NT4 |
|---|---:|---:|---:|---:|
| IQ3_XXS | 2.15× | 1.86× | 1.64× | 1.57× |
| IQ3_S | 3.24× | 2.46× | 2.29× | 2.14× |
| IQ2_S | 1.24× | 1.57× | 1.51× | 1.36× |

All sampled outputs matched the existing IQ256 kernel bit for bit. Single-input GGML uses another floating-point reduction order; the maximum observed relative difference was 2.96e-4, including near-zero outputs. This is not an answer-quality result or an end-to-end speedup.

`STRATA_IQ_PACKED_CACHE_GB=6` enables an experimental bounded LRU cache of those derived GU rows on non-AVX512 AVX2 CPUs. Entries use source/layer/expert identity, with the active batch retained until all row tasks finish. Workers pack a missing row range before computing it. Original blobs, GPU copies and down weights remain unchanged. The default is disabled. Regression is deferred until a repeatable complete-decode improvement, as requested.

| Cache implementation | Capacity | Decode rates, tokens/s | Median |
|---|---:|---|---:|
| Scalar packing, value-initialized new buffers | 6 GiB | 33.7, 30.6, 33.3 | 33.3 |
| Scalar packing, value-initialized new buffers | 12 GiB | 31.9, 38.8, 46.6 | 38.8 |
| SIMD packing, value-initialized new buffers | 6 GiB | 42.8, 42.4, 48.5 | 42.8 |
| SIMD packing, value-initialized new buffers | 12 GiB | 37.2, 50.5, 57.6 | 50.5 |
| SIMD packing, recycled uninitialized buffers | 12 GiB | 52.2, 64.1, 71.5 | 64.1 |
| Recycled buffers, native GU type21 only | 12 GiB | 61.8, 58.9, 75.7 | 61.8 |
| Recycled buffers, native GU types18+21 | 12 GiB | 52.6, 60.6, 73.0 | 60.6 |

The baseline medians in the initial sweeps were 63.8 and60.3. Buffer recycling removed host zeroing and reused physical pages; its64.1 median still fell below the66.2/65.1 controls. Restricting the cache to selected native GU formats also failed to improve complete generation. The optimization remains disabled and work on it is stopped. Increased hit traffic (146-byte derived blocks versus82–110 original bytes) and approximately23% misses are plausible costs, not a proven wall-time attribution. The161.4ms preparation counter excludes packing inside worker tasks. A read-only review found no defect in buffer lifetime or complete row initialization. Incomplete follow-up combinations were stopped and do not count as measured medians.

### Verifier critical-path trials

The stock `--spec4` configuration automatically uses verification width6 and MTP cap4 when suffix drafting is enabled. Explicit width8/MTP4 measured61.4 tokens/s; legacy `--spec6` (width8/MTP6) measured52.2. Zero PCIe expert offloading measured63.4. Controls bracketing that sweep measured60.1 and65.5. No setting was retained and regression remains deferred.

Two default-off probes target the existing verifier path. `STRATA_Q6_COMPACT_MULTI=1` chooses the existing two-row multi-column layout only for Q6_K projections, preserving checkpoint bytes and activation quantization while changing floating-point reduction order. It measured55.5 tokens/s. `STRATA_FETCH_128=1` changes mapped-fetch launch size from384 to128 CTAs; its grid-stride loop still copies the same bytes. It measured60.0. The controls were59.5 and65.2. Neither option was retained.

`STRATA_Q6_ROW_WARP=1` is a default-off option retained on this server after measurement and focused regression. Each warp owns a complete output row and uses the existing Q6_K/Q8_1 dot; this eliminates the cross-warp shared-memory reduction. Weight bytes, quantized activations and single-column calls remain unchanged. Floating-point accumulation order changes. Medians68.5,68.2 and67.0 repeated against controls65.1,63.9 and63.6. All candidate runs produced the same output hashes and aggregate draft acceptance81.5%; candidate answers can differ from the controls. This is a measured gain against these controls, not90 tokens/s. It takes priority over the compact flag when both are enabled.

After the gain repeated, the existing MMVQ contract check and both96-case numerical suites passed. The new suites compare independent scalar Q6_K dequantization and FP64 dots over the same Q8_1 bytes, with widths256/512/2560/8192, ragged row tails, inactive-output sentinels and captured launches. Worst relative L2 errors were2.164e-7 for the default and2.169e-7 for rowwarp (limit3e-6). Three generated Python functions passed their assertions, and retrieval returned the exact code from a20,534-token prompt. Missing/invalid/correct credentials returned401/401/200. Both remote and local Pi completed a real bash printf tool call and final response. Context capacity remains65,536; the retrieval check did not fill that entire capacity.

### CPU phase and sign-decoding probes

`STRATA_DECODE_TIMING=1` now prints request-local deltas of the existing CPU pool gate/up, activation quantization and down timers. It does not enable GPU event profiling or change model arithmetic. One512-token diagnostic with the retained Q6 setting measured49.9 tokens/s and16.32/0.22/5.26ms per window for these phases. This is not a three-prompt benchmark or a hardware ceiling.

A throwaway sign-decoding change replaced two128-bit shuffles and an insert with one256-bit broadcast/shuffle, preserving the sign bytes without expanding weight memory. Its basic diagnostic measured68.4 tokens/s and6.38/0.20/3.00ms/window. Both GU and down changed, so that isolated observation cannot attribute the improvement to the sign change. Saved binaries were compared under the same Q6 setting:

| Run | Rates, tokens/s | Median |
|---|---|---:|
| Control |60.3,63.9,76.0|63.9|
| Sign shuffle |66.0,62.8,74.1|66.0|
| Control |66.9,63.5,76.0|66.9|
| Sign shuffle |65.3,63.2,61.4|63.2|
| Control |55.4,52.8,74.8|55.4|

No repeatable gain was established; the sign source change is reverted. Several identical-output runs differ considerably in speed, including the candidate's TypeScript result, so answer changes alone do not explain the variance. No regression suite ran for this rejected probe. Valid three-prompt records remain separate from single diagnostic records.

Shortening the existing worker spin-before-sleep threshold from20,000us to100/500/1000us measured medians52.3/57.1/59.2 against58.5/63.8 controls. The100us rates were52.3/49.4/59.9,500us41.3/57.1/70.4,1000us55.4/59.2/67.5. The retained Q6 flag stayed enabled. No improvement was established, so the default worker threshold is restored; no new source or regression suite was needed.

### Two-row Q6 and comparable-work diagnostics

`STRATA_Q6_ROWS_PER_WARP=2`, with the retained rowwarp flag enabled, gives each warp two guarded output rows. Each row preserves the original block order and Q6_K/Q8_1 dot, with separate accumulators. It halves row-block count and may expose instruction overlap, but activation reuse is not explicit and additional registers may hurt. The default remains one row. Basic generation measured56.4 for one row and57.7 for two; alternating three-prompt medians were63.5/60.2/67.1/61.1/64.4. Both two-row repetitions lost to their nearby controls, so the option stays disabled. Regression was not run for this rejected candidate.

`STRATA_POOL_PHASE_TIMING=1`, together with decode timing, separates native pool pre-park, drain-and-completion and re-park durations. Existing GU/down phase counters include synchronization; neither those counters nor drain are pure kernel time. Request-end summaries also report verification-width counts, an ordered dispatch digest covering layer, width, routes, placement and formats, CPU input-group counts, and logical CPU expert-blob bytes. The digest does not compare activation values; logical bytes are not measured hardware memory traffic. Per-task clocks, CPU samples, arrays and summaries additionally require `STRATA_POOL_TASK_TIMING=1`; this separates their overhead from lighter phase comparisons. Earlier diagnostic builds below collected both under the phase flag alone. Diagnostics remain disabled by default and do not change placement or model arithmetic.

The next optimization depends on the exposed tail. Unequal worker completion correlated with format/input count supports smaller or cost-weighted CPU tasks; uniformly costly tasks do not establish imbalance. Per-expert GU/quant/down pipelining is justified only if early expert completion can fill otherwise idle worker time. Latency-based PCIe placement needs evidence that CPU finishes later and the GPU can absorb transfer plus compute. Aggregate CPU timers alone cannot select among these changes.

The first three identical-prompt basic replays used executable `d4bd47b6ae5ab41db1fee41698bbd4869726ed971fafcf880fb2e13f53c96be3`. They measured59.7/65.6/67.3 tokens/s, with different answers, widths and dispatch digests. Drain-and-completion was14.28/9.49/9.77ms/window; pre-/re-park combined0.46/0.07/0.07ms/window. This localizes most pool time beyond parking but does not prove unequal tasks. These records are diagnostics, excluded from benchmark medians. Subsequent opt-in task sampling records all task wall durations and active worker final completion, plus execution thread CPU time every16th task. Sampling excludes parked spinning but adds overhead; sampled totals and finish tails are diagnostic, not a recoverable-time guarantee. CPU sampling reports zero samples on platforms without a thread CPU clock.

The task diagnostic (`3310fd5f6f5d9b04a5c03460ec12cbb8dabeac50c86a2805b5af502ef311bd25`) measured51.3/59.1/59.9 tokens/s. Warm replay GU/down completion tails were1.38/0.76 and1.53/0.73ms/window; sampled execution CPU was3721.86/3864.30 and3723.62/3866.45ms wall. Mean task duration increased from roughly34–35us at maximum group1 to62–63us at group4. These means mix layers and GU/down, and sampling can miss delays. This build used the upper middle finish for even active-worker counts; the corrected source averages the middle pair, so those tails are lower estimates. `STRATA_POOL_TASKS_PER_THREAD=6` is the resulting bounded balancing probe: double the native phase's default task count, keeping original row arithmetic and the fused quantization's32-row cap. Invalid/unset values retain3; accepted range1..12. First measure basic and complete decode without diagnostics; require repeats before regression.

The first uninstrumented basic comparison measured52.6 tokens/s for three tasks/thread and51.4 for six. A bracketed identical-prompt replay comparison then measured control61.8/50.0/57.2, six tasks62.1/59.4/60.3, closing control66.3/62.3/62.5. The candidate's warm pair lost to the closing control, so six tasks/thread was rejected and three remains retained. These basic records are excluded from three-distinct-prompt benchmark medians. No numerical regression ran for this rejected probe.

### Expert readiness and further bounded probes

`STRATA_POOL_GU_BALANCE=1` optionally divides native gate/up rows in measured-cost space while retaining three tasks/thread, the original row kernels and the down schedule. A reusable vector of monotonic boundaries weights each expert by its format and input count. Costs came from the earlier original-kernel AVX2 probe; type21 NT2 was slightly cheaper than NT1, so linear input-count weighting would be inappropriate. Eligibility requires AVX2 withoutAVX512, GU types18/21/22, NT1–4 and default kernel/cache settings; fusion, dependency pipelines and unsupported cases retain equal-row scheduling. Execution and readiness diagnostics share the boundaries. These server-specific estimates exclude multithread contention. Review found no Critical/Important issues; missing gather/prefetch override guards were added before building.

Engine `fcfedd229750fafb78a3e210bab4941dbdba40528cdcfcfb905f125662f731a5` built engine-only and completed basic control65.6/63.9/65.1, candidate67.1/64.4/65.8 and closing66.5/64.8/65.6. All corresponding answer hashes and aggregate draft counts match. Candidate warm65.1 lost closing65.2, so a repeatable gain remains unproven. The option stays disabled; no full benchmark or regression ran. These identical-prompt replays remain outside three-prompt medians.

The readiness diagnostic used `9bed4314d14513eabe046e4f751b2aaa984dd2a6c6a9541bf007bc15f7342495`, with three basic rates61.2/60.0/61.0. All three dispatch digests and output hashes match the earlier task diagnostic, although rates differ. Warm expert-finish headroom summed per window was2.82/2.78ms mean and5.27/5.25ms maximum. These values estimate readiness from row-task end times; they do not prove available overlap or identical activation values. Warm worker tails remained1.81/.78 and1.79/.83ms/window. Per-expert dependencies are therefore a measured candidate, not an established speedup.

`STRATA_POOL_EXPERT_PIPELINE=1` is an experimental, default-off prototype for multiple native experts with fused HQ disabled. A fixed epoch-checked batch claims all GU tasks before down tasks. Each expert's disjoint GU segments join through an acquire/release remaining-row counter; the final segment runs the same complete-expert activation quantizer and release-publishes readiness. Down segments acquire readiness before reading the intermediate. Waits use the existing stall deadline and release GPU waits before abort. This preserves row kernels and original blobs, but quantization moves onto a worker and execution order changes. A down task can still wait for a late expert instead of selecting another ready expert. Combined pipeline elapsed time is reported separately; GU/down task diagnostics cover the original two phases only. Read-only review found no issues. Engine `b0c959542c31805b4f4783163e8c636d9a0a7d319f077b535da62ec7d3f9cb7d` completed basic control/candidate at58.0/62.0 tokens/s with identical output hashes, but different drafted counts377/381. Alternating three-prompt medians were66.3/61.8/61.5/68.3/66.2. The candidate gain did not repeat, so the option remains disabled. A lighter phase comparison is the next diagnostic; no numerical or held-out answer-quality validation is claimed and regression has not run for this prototype.

`STRATA_Q6_ROW_WARP_SINGLE=1` separately opts single-column Q6_K into the existing one-row warp kernel. It preserves Q6_K/Q8_1 operands but changes floating-point summation order; the multi-column setting remains independent. Basic control/candidate rates52.2/53.2 prompted full alternating medians55.4/63.4/68.2/65.6/59.0. Candidate rates were55.6/63.4/75.9 and65.6/58.2/74.1; controls49.8/55.4/77.2,68.2/63.7/76.3,53.6/59.0/64.6. The variability and stronger middle control leave a repeatable gain unproven. This option remains off; no regression suite ran for it.

Light phase comparisons used engine `391767935c7ea8d9b6c4df8a5af00cb00de67bb27c6e7c44c990ada30bac0825`. Basic control/pipeline/control rates were51.3/56.4/60.6,60.7/64.2/67.4,66.9/66.3/67.8. The two controls match all three dispatch digests, output hashes and width histograms. Their CPU execution remained about10ms/window and GPU-reach waits about16.4–16.9ms/window; host staging changed11.07/5.43/3.63 to0.05/0.06/0.06ms/window. The pipeline changed dispatched work, so its rates do not isolate a scheduling improvement. Main PLE I/O was confirmed as RAM. These are diagnostic replays, excluded from benchmark medians.

The older timing log presented host staging as an additive component, although PLE collection during layer0 is also included in per-layer host time. The label now states that overlap. `STRATA_HOST_STAGE_TIMING=1` separates main-window input/hash/prefetch preparation from PLE collection, reporting wall time and Linux thread CPU time plus minor/major fault deltas. Resource-sampling failures and non-Linux platforms report no resource samples. Resource CPU accounting includes probe calls and can exceed the very short wall interval; tiny differences are not precise kernel CPU timings. The slot/batch path is not instrumented. The probe leaves arithmetic, placement and flag publication unchanged and remains disabled by default.

Engine `f9e8ad7641655bb217bb5959507a4235f5e6aec18d014791ffebc9827c4f92f7` completed three diagnostic replays54.0/60.1/66.9. Dispatch digests, outputs and widths match both light controls. Input staging averaged0.002ms/window; PLE collection0.048/0.145/0.047, with0/1/0 minor faults and zero major faults. This run did not reproduce the staging spike. Native CPU execution instead varied19.58/13.29/9.97ms/window, while GPU-reach waiting stayed16.3–16.8. Thus variance can move between intervals; no single scheduling or memory cause is established. A bounded20-versus-default22-worker probe is next, leaving two vCPUs for host/driver work while keeping the host worker and original row arithmetic. It requires repeated uninstrumented gains before regression.

The20-worker warm basic comparison was control55.4/63.8/65.7, candidate66.0/65.4/66.9, closing control67.2/64.9/66.3. Candidate warm mean66.15 exceeded both controls64.75/65.60 and prompted full alternating medians61.9/65.4/60.8/66.3/54.4. The gain repeated against these controls, but the candidate remains below older validated Q6 medians. Original checkpoint bytes and row dots are unchanged. Only after those repeated gains, focused coding/context/auth/Pi regression ran on engine `f9e8ad7641655bb217bb5959507a4235f5e6aec18d014791ffebc9827c4f92f7`: all three generated Python functions passed assertions, exact retrieval passed from20,534 prompt tokens, credentials returned401/401/200, and both remote and local Pi completed a real bash tool call and final response. This CLI-only worker change did not trigger another numerical kernel suite. The20-worker configuration is retained as a measured candidate; the calibrated22-worker control is preserved. This does not establish90 tokens/s or a new best validated median.

`STRATA_VERIFY_TAIL_PROFILE=1` is a default-off diagnostic of the contiguous GPU MoE slice: plan wait, resident groups, PCIe preparation/fetch, PCIe groups, CPU-result wait, and copy/shared-stream-join/combine. It retains normal shared-expert stream overlap; the existing full `STRATA_VERIFY_PROFILE` disables that overlap and takes precedence if both flags are set. Tail mode omits earlier/later and fused-GR stamps, skips missing/out-of-order neighboring timestamps, and labels its partial total. Existing split-group collection restrictions remain. With device planning, stamp23 is absent and CPU-wait/final-copy intervals are unavailable, explicitly reported. It adds timestamp overhead, so its rates are diagnostic rather than benchmark results. Read-only review found no Critical/Important issues; both minor labeling/coverage findings were addressed. Engine-only builds and basic validation are recorded below.

Tail profiler engine `6b8fa7e056930514fd1c4e9e21e21944b4bc71ddbd2ee53dc064c62a73cdfabf` completed basic rates64.8/63.9/65.9 on20 workers, with all three work digests, output hashes and width histograms matching the earlier controls. Native CPU execution was11.53/11.17/10.81ms/window. The GPU summaries measured CPU-result waits about3.28/2.59/2.34ms/window, PCIe preparation/fetch6.48/5.96/5.76, and resident groups4.80/4.40/4.61. These first summaries included219/222/217 profiled windows versus210/221/216 decode windows because prompt/replay stamps were not cleared. They are approximate diagnostics, not request-exact placement costs or a speed gain. The decode reporter now discards prior GPU stamps before its snapshots. A corrected control/no-PCIe/control basic comparison is running to see whether fetch removal merely exposes CPU time. No regression runs for these diagnostic changes.

Corrected engine `351f0e26edd493cc2a632f5b9ba20292b66b46ec013ca519e5a30d37074e59d4` completed control64.2/63.8/65.4, no-PCIe63.2/64.6/69.0, closing control64.5/63.8/65.2 on20 workers. Control GPU counts exactly match210/221/216 decode windows and earlier work/output/width digests. GPU CPU-result waits were2.18–3.01ms/window; PCIe preparation/fetch5.85–6.30. Removing PCIe exposed7.19–9.50ms/window CPU-result waits and changed work/routes/widths. It did not show a consistent gain. These instrumented rates are excluded from the benchmark ledger; calibrated PCIe remains retained.

`STRATA_LAYER_TIMING=1` adds request-local per-layer CPU time, distinct groups, input entries, resident groups/entries and fetch counts. Together with tail profiling it reports six GPU MoE intervals per layer. Engine `aa955802bbc07a35079712d3378300b25d846c4d5b211345ee6307d64e67fc64` completed basic53.3/58.0/61.2, with all48 layer/profile counts matching decode windows and earlier dispatch/output/width digests. CPU execution stayed10.52/10.31/10.16ms/window, while staging8.85/4.29/3.04 contributed a large layer0 GPU wait. CPU costs exclude that staging. IQ3_S gate/up layers averaged about49us per weighted group, versus32–40us for several other native formats; these are workload-local wall measurements, not pure kernel coefficients.

`STRATA_VERIFY_FETCH_OVERLAP=1` is a default-off native mode2 scheduling option. After the published GPU plan is copied, a nonblocking stream waits for flagB, stages mapped expert blobs and rebases only PCIe pointers while the main stream computes resident groups. The main stream joins before computing PCIe groups; shared overlap and original weight bytes remain unchanged. Full profiling retains sequential fetch. This follows [NVIDIA's capture fork/join requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html). Tail profiling with this option measures resident work under concurrent fetch and only the remaining fetch join, not separate fetch execution. Review found no issues. Engine `baab79777561438e12aa3d6276815e62a74f27234c6ecd4848b97df225fd1baa` completed basic candidate67.8/66.4/67.4 versus controls52.5/58.7/62.3 and66.5/65.2/66.1, then full candidate medians66.0/67.0 against64.3/64.4/62.1. Candidate rates were66.0/62.8/74.6 and67.0/63.4/75.8. This repeated a gain against the current controls but remains below older validated Q6 medians.

Only after that repeat, focused regression passed on the same executable: all three generated Python functions, exact retrieval from20,534 prompt tokens, auth401/401/200, and remote/local Pi bash tool calls plus final responses. No numerical kernel suite reran for this scheduling-only change. The20-worker Q6+fetch configuration is retained; the cost-based placement trial below uses it as its control. The65,536 capacity remains configured; the retrieval check did not fill it. G=2 resource ownership was reviewed but these generations used the normal batch-window path.

`STRATA_PCIE_COST_PROFILE` optionally supplies measured CPU/resident/fetch/PCIe affine costs, bound to worker count, host participation, native format and blob bytes. A greedy selector removes one CPU group/input set and adds transfer plus GPU computation only if the predicted later completion decreases by more than5us. It respects pinned eligibility, staging caps and the64-entry tables. Positive PCIe fraction enables the cost policy without a fixed fractional quota; unsupported formats, invalid profiles, worker mismatch, remote/peer and other modes retain configured placement. Fetch overlap is accounted for when enabled. A local nonnegative fit uses48-layer diagnostics; two format buckets with only three samples are excluded. This applies Fiddler's completion-cost idea to unchanged native IQ3_S bytes. Unprofiled improvement and regression remain unproven.

The [reference cost profile](benchmarks/2026-10-05-5060ti-pcie-cost.txt) is specific to this server's20-worker setup and diagnostic workload; it is not a transferable hardware calibration. Header fields are version, worker count and host-work boolean. Each row contains GU type, down type, blob bytes, then eleven nonnegative microsecond coefficients: CPU constant/group/input, resident constant/group/input, fetch constant/blob, and PCIe constant/group/input. CPU constant includes measured activation/job preparation. CPU/resident/fetch/PCIe RMS fit residuals were approximately2–7/<0.4/0.8–6.6/<0.3us across included buckets. These are training residuals, not held-out accuracy. Set the environment variable to the absolute profile path to enable the experiment; keep it unset for the configured quota.

Review found no Critical/Important issues. A minor malformed unsigned-byte parsing case was fixed with `from_chars`, including full-token and overflow checks; follow-up review found no remaining issues. Engine `8895af2f71a226ffa41080eb8115b963db03beadecdf66e631d980a876da52be` built and completed unprofiled basic control56.3/59.3/62.3, combined fetch+cost63.7/64.2/63.8, closing control63.9/64.3/66.3. Candidate warm mean64.00 lost closing65.30, so the cost selector remains disabled. No full benchmark or regression ran for this rejected trial. These basic replays are excluded from benchmark medians. Fetch quality evidence above belongs to the earlier `baab797...` executable.

The same executable then compared `--spec8 --mtp-max-t4 --suffix-draft6` with the retained fetch configuration. The suffix option sets minimum matching length, not requested draft count. Wider strong-match lookup basic rates were60.3/64.0/67.0 versus controls66.1/65.7/67.0 and65.5/65.3/67.1. Warm candidate mean65.50 lost both controls66.35/66.20. This trial is rejected, with no full benchmark or regression. Original IQ3_S and configured65,536 capacity were preserved.

Four-token MTP windows with `--spec-min-p0.50` were then compared with the retained0.70 cutoff on the same `fcfedd229750fafb78a3e210bab4941dbdba40528cdcfcfb905f125662f731a5` executable, Q2 draft pack and20-worker fetch configuration. Basic warm66.4 beat64.5/65.6, prompting full alternation. Full control medians64.7/64.9/67.5 did not establish a repeated gain for candidates65.2/64.6. Candidate rates were65.2/63.8/78.2 and64.6/63.7/75.3. The cutoff was restored to0.70, with no regression for the rejected option. Earlier six-token/lower-cutoff and suffix-only trials remain separate experiments.

### Placement

`STRATA_DMA_CALLBACK_PROFILE=1` reports Linux callback CPU/affinity once per callback thread. In engine `4d8a69d58000d09d2e99f021e4a5fe08512ff93434bdc9748c43b5cdf20643cc`, the DMA completion callback was constrained toCPU0 with the inference host. Diagnostic rates62.2/61.6/70.4 are excluded from benchmarks. This established thread placement, not scheduling cost. `STRATA_DMA_CALLBACK_CPU=21` optionally validates CPU21 against launch affinity, then moves only the callback thread there once. Invalid selection or OS failure leaves the original flagB handoff intact; non-Linux does not change affinity. The callback makes no CUDA API calls, following [NVIDIA's host-function constraints](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html). Source review found no Critical/Important issues.

Engine `d870e605abefe2a90a1193113d8d57455b81ad3ceb84d23fdfd3ebbe28e6e855` successfully moved the callback toCPU21, but basic rates54.5/57.4/58.2 lost retained kernel-fetch controls61.1/61.7/62.7 and56.1/60.4/63.7. Warm candidate57.8 lost62.2/62.05. The DMA affinity option remains off; no full benchmark or regression ran. The first two answer hashes and aggregate draft counts match across all three rows, but complete routed work was not profiled. This rejects the combined DMA+CPU21 candidate without isolating the affinity move's marginal cost against unmodified DMA.

`STRATA_STAGING_REUSE=1` is a default-off prototype that retains original native expert blobs in the verifier's existing16 maximum-sized GPU staging slots. A host LRU table keys source identity, layer, expert and byte length. All current-layer cached hits are protected before new admissions; cached experts join resident groups. Indexed copy destinations are appended to the plan without changing earlier offsets. Split groups, non-native and other transfer modes, peer/remote execution and disabled PCIe keep the old path and invalidate tags before sequential staging writes. Device-generated and empty plans disable indexing. The old six-argument copy APIs remain available to SYCL; CUDA/HIP adds indexed overloads. Expert staging VRAM and main weight precision remain unchanged. The shared-header API compatibility finding was fixed; a later capacity finding is described below.

Engine `72fdf32a8ffe4ca003c32d21fc08110463ae5ca50bb41f800fd6925be56d94ce` built engine-only and completed basic candidate64.5/64.2/66.1 versus controls64.8/64.3/66.8 and68.8/66.8/68.6. Warm candidate65.15 lost controls65.55/67.70. This trial mistakenly required64 slots even though this checkout allocates16, leaving reuse disabled. It cannot evaluate reuse performance. No full benchmark or regression ran. A diagnostic-only hit/active counter under `STRATA_LAYER_TIMING` distinguishes actual activation. Corrected16-slot engine `dc524af757625a7120343c0ca4fff125bedb2bcbe92f73cf394b4c6b6e9d0b5e` built and completed diagnostic basic64.0/64.6/65.4, active in all48 layers. It served0.386–0.517 staging-hit groups/window while transferring33.073–38.115 groups. Diagnostic rates are excluded from benchmark medians; different work/output hashes prevent attributing rate changes solely to reuse.

`STRATA_STAGING_REUSE=64` explicitly expands native staging to64 slots, adding121.875MiB for this pack. Default allocation remains16; value1 reuses those16. Metadata accepts the actual allocated count and bounds lookup/reservation to it. Source/base/stride/capacity changes invalidate tags. All captured fetch offsets and split-group halves use actual allocation. Review found no Critical/Important issues; the stale group-count comment was corrected. Engine `eba4b8ea57a65f45ad078bb8850787a28a0ef8c034931d58714c5ed61474b3fd` built engine-only. Basic candidate63.7/66.0/69.4 gave warm67.7 versus controls66.6/63.65, prompting full alternation. Full control medians66.2/66.8/64.7 exceeded or bracketed candidate63.6/65.1; the gain did not repeat. Candidate individual rates were63.6/63.0/76.7 and65.1/63.5/77.3. Permanent cache stayed4,601 slots. Expansion remains disabled, with no regression suite. Ninety tokens/s remains unproven.

The same-binary diagnostic completed with staging active in all48 layers. Hits were6.444/5.978/4.901 per window, about1.86–2.11% ofnonpermanent groups, while transfers remained39.136/32.443/29.847. CPU-result waits totaled3.11–4.82ms/window after PCIe work. Resident GPU work under concurrent fetch totaled9.58–10.66ms/window; remaining fetch joins were about0.05ms/window. Those intervals cannot isolate fetch contention against older sequential profiles because routed work differs. Diagnostic rates59.9/63.4/66.9 are excluded from benchmark medians. The retained20-worker Q6+fetch configuration was restored.

Selected primary-paper sections also informed this cache probe: [MoE-Infinity](https://arxiv.org/abs/2401.14361) uses request activation traces and predicted priorities rather than plain LRU; [MoE-Lightning](https://arxiv.org/abs/2411.11217) develops paged transfers and microbatch CPU/GPU scheduling; [the caching analysis](https://arxiv.org/abs/2511.05814) examines LRU traces and LFU experiments. These suggest measuring route reuse and admission quality. Their throughput claims do not establish a gain on this single-request checkpoint, and no paper speedup is included in our ledger.

`STRATA_PCIE_REUSE=1` selects the missed, pinned experts serving the most input rows, subject to the same transfer quota and staging limit. Ties retain the old last-in-routing-order choice. Resident and peer-GPU experts are excluded. This transfers the same expert blob once for multiple inputs and leaves CPU/GPU result merging unchanged.

### Projection arithmetic

`STRATA_BF16_DECODE_TC=1` rounds eligible FP32 activations to BF16 and uses WMMA with FP32 accumulation. It only applies to 2–8 token windows, input dimensions divisible by 16 and no larger than 4096, and output dimensions divisible by 16 and at least 1024. Other shapes use the original kernel. Eight warps split the reduction; microbenchmarks alone do not establish an end-to-end gain. This changes arithmetic and requires answer-quality checks. It remains disabled by default.

`STRATA_Q6_INT8_MMA=1` is a separate CUDA prototype for native Q6_K projections with 2–8 columns. It unpacks the original Q6 integer values directly into `mma.sync.m16n8k16` operands, with the existing Q8_1 activation bytes. Signed Q6 scales are applied to each 16-element INT32 dot, followed by the original FP16 block scales and FP32 accumulation. There is no expanded weight cache or new model quantization. Floating-point summation order changes, so bit identity and answer quality are not assumed. [NVIDIA's integer-fragment specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) defines the lane mapping used here.

The opt-in path requires a CUDA device of compute capability 8.0 or newer; a lower-architecture PTX image executes a rowwarp fallback. One-time diagnostics report the loaded PTX/binary versions and whether the compiled MMA path is active. HIP and single-column calls retain their existing dispatch. Source review found no major defect, and the engine-only build passed.

The first basic comparison used executable `5d183f31cd3b79a5c66b3478175c2a7a805870e4c6f87ca90020b24694a9f1d2`: controls68.7/66.6/66.6 and64.5/64.6/66.7, candidate56.9/62.8/59.8. Both controls had matching output hashes and draft counts; the candidate produced different completions. Loaded PTX/binary versions were120, confirming MMA rather than the fallback, and every observed specialization used208 registers/thread. The candidate lost the basic comparison and remains disabled. No full benchmark, numerical regression or answer-quality regression ran for it; these repeated-prompt measurements are excluded from the three-prompt ledger. High register use motivates a bounded loop-unroll experiment, but does not establish the cause of the slowdown. The90 tokens/s target remains unproven.

Suppressing the outer eight-iteration loop's unroll reduced loaded register use to56/thread in executable `d1fe650fafe3862d0773738f556032d9e3893cf4a714aa87819fa1816e3ba10d`. Its basic rates46.7/47.2/47.5 lost to controls59.0/57.5/59.8 and64.0/63.0/64.8. Both controls retained the prior hashes and draft counts; their speed variation still does not establish identical internal work. Lower register use did not yield a gain, and loop unrolling was restored. This second failed candidate received no full benchmark or regression. The MMA override remains off.

Reducing the existing mapped-copy grid from384 to128 blocks was also retested with20 workers and fetch overlap enabled, while MMA stayed off. The rebuilt executable `fdf4a298d57e798afbd9908d3d0efd8f41471d22c0908a12bf4668a16c13e881` produced basic controls61.6/59.8/60.6 and66.8/66.4/69.1, with candidate63.9/63.3/68.5. Corresponding output hashes and aggregate draft counts matched across all three arms. The candidate's warm pair65.9 lost to the closing control67.75, so the setting remains off and no full benchmark or regression ran. The basic measurements are excluded from the public ledger; neither a GPU-contention cause nor a gain is established.

The fixed PCIe fraction was then retuned on the same executable and retained20-worker/fetch-overlap setup, changing one option at a time. At0.35, basic controls67.7/65.6/67.0 and67.8/65.3/66.7 bracketed candidate54.7/54.8/58.5. Aggregate draft acceptance increased, but speed dropped. At0.10, controls68.3/67.8/68.9 and66.6/66.5/67.8 bracketed candidate66.6/69.6/67.5; its warm pair68.55 was only0.20 above the first control68.35. Corresponding controls kept matching hashes and aggregate draft counts, while changed placement produced different completions. Neither candidate earned a full benchmark or regression, and0.20 remains the retained fraction. The quota rounds down per layer: `floor(nmiss * pcie_num / 256)`. These basic results establish no90 tokens/s gain or placement root cause.

### Draft precision

`tools/mtp_pack.py --experts q5_0` and `--experts q8_0`, followed by `tools/mtp_rt.py`, produce optional draft packs. These change the MTP routed experts only. Dense draft weights and the vocabulary subset must remain identical between arms. The main IQ3_S model is unchanged.

The runtime validates `experts.txt` and the exact expert file size before loading. Missing manifests preserve the legacy canonical Q2_0 path. Native Q5_0/Q8_0 packs use original GGUF block order and Q8_1 activations in the existing native expert kernel. Q8_0 experts occupy 2.49 GiB versus 675 MiB for Q2_0, reducing target-cache slots from 4601 to 3629. In the first sweep, draft acceptance was about 81% for both formats; deeper Q8 speculation reduced it to about 73% and was slower.

### Deployment provenance

Initial engine experiments accidentally launched the installer binary because `exe` still pointed at `engine/strata`. Those engine-variant reports are invalid and excluded above. At 08:34 UTC on 2026-10-05, the config was changed to `build/strata`. Each subsequent run checks the configured and live executable hashes; engine variants also require the built hash to match. The table above used SHA-256 `30439454c75f8ad50b4dcf14dbe5bcd91c3e6d340fb02f50ae6a3290935f30f3`.

## Verification limits

### Signed IQ3_S codebook basic trial

`STRATA_IQ_SIGNED_GRID=1` selects a default-off AVX2 GU decoder for native type 21. A 32 KiB table combines the original nine-bit grid index with four sign bits, removing vector sign expansion without expanding expert rows or changing checkpoint bytes. The shared row loop retains INT32 lane accumulation and FP32 FMA/reduction order. Normal Q8_K activations in `[-127,127]` preserve the integer products; manually supplied `-128` computes the true signed product while the original sign-on-activation path wraps negation. The original-decoder GU balancing costs exclude this option. Pair sums remain safely within [Intel's VPMADDUBSW limits](https://www.intel.com/content/dam/develop/external/us/en/documents/cpp_compiler_classic.pdf).

An engine-only build and control/candidate/control basic generation trial completed at 19:00 UTC on 2026-10-05 with executable SHA-256 `8ef5657f88ce0b00d56dbb61df8b5ddd44649b161c93d315338ccefb03872e36`. Controls returned 67.3/65.0/66.7 and 65.3/65.1/66.4 tokens/s; the candidate returned 64.1/65.0/66.3. Its warm mean 65.65 did not beat controls 65.85/65.75. All corresponding output hashes and aggregate draft counts matched, the table activation log was present, and the permanent cache stayed at 4601 slots. These three replays of one prompt are basic checks, not the three-distinct-prompt benchmark; internal execution equivalence is not established by the answer hashes alone. No full benchmark, numerical suite or quality regression followed. The option remains disabled, with 20 workers, Q6 rowwarp, GPU fetch overlap and PCIe fraction 0.20 restored. The 90 tokens/s target remains unmet.

### Bounded SwiGLU grid basic trial

A fresh Nsight hardware trace of the retained configuration captured one 256-token generation with executable `8ef5657f88ce0b00d56dbb61df8b5ddd44649b161c93d315338ccefb03872e36`. The fused expert SwiGLU/Q8_1 kernel averaged 26.60 µs, compared with 1.02 µs in the older trace. In the fresh trace, calls overlapping mapped fetch averaged 69.57 µs, while isolated calls averaged 0.91 µs. Active groups and inputs differ between these samples; temporal overlap does not prove contention or exposed critical-path cost. Inspection of two- and four-column Q6 rowwarp machine code found shared weight-decode operations, so a duplicate predecode implementation was not pursued.

Those whole-trace means include prefill. Ordered expert chains separate nine prefill windows (49 inputs) from 113 decode windows (301 inputs). All six-input resident groups belong to prefill. Decode-only resident and combined GPU expert spans averaged 10.56 and 10.97 ms/window; mapped fetch averaged 6.44 ms/window and overlapped resident work. The wait after PCIe expert computation averaged 2.44 ms/window; its CPU-result identity is inferred from launch order. PCIe expert computation itself averaged 0.37 ms/window. These overlapping intervals cannot be added or treated as recoverable savings. Earlier whole-trace totals overstate the decode work available to a fetch/compute pipeline.

`STRATA_IQ_SWIGLU_32=1` tests a CUDA-only cap of 32 CTAs for the existing fused expert activation kernel. Its existing grid-stride loop, 256 threads, arithmetic, scratch and checkpoint bytes remain unchanged. Whole-warp boundaries preserve the existing reduction and output ownership. Default, smaller-grid, two-pass and HIP paths retain their original launch geometry.

The engine-only build and control/candidate/control basic trial completed at 19:21 UTC on 2026-10-05 with SHA-256 `09ce20f0f9c999695af43183ab4e3942433b460bd66f37905ffca89fd599bc48`. Controls returned 65.0/64.9/66.3 and 66.1/64.8/66.4 tokens/s; the candidate returned 66.6/64.9/65.8. Its warm mean 65.35 lost both controls at 65.60. Corresponding output hashes and aggregate draft counts matched, the cap activation log was present, and the permanent cache stayed at 4601 slots. These single-prompt replays establish no complete benchmark gain or internal-work equivalence. No full benchmark or regression followed. The cap remains disabled. The 90 tokens/s target remains unmet.

A capped-grid Nsight diagnostic completed at 19:27 UTC. Recorded launch dimensions confirmed that 8736 calls used 32 blocks instead of the retained trace's 50–150 blocks; the remaining 2976 calls kept 25 blocks. SwiGLU still averaged 26.57 µs across 11712 calls (311.24 ms cumulative), versus 26.60 µs (311.50 ms) without the cap. Overlapping-fetch calls averaged 69.54 µs and isolated calls 0.91 µs. Both diagnostic generations produced the same 256-token output hash and aggregate draft counts (188 drafted, 143 accepted), but used different executables and do not establish identical routes or placement. Reducing the grid did not improve this observed cost or basic throughput. Profiling rates remain outside the benchmark ledger. Cleanup restored the retained options, verified on the live executable at 19:33 UTC; no new quality validation is claimed for this build.

### Pending cache-admission timing

Three retained-configuration basic replays with executable `09ce20f0f9c999695af43183ab4e3942433b460bd66f37905ffca89fd599bc48` left 2.98/3.02/2.74 ms/window outside the existing verify, commit and draft timers. The native pool's pre-park/re-park intervals were only about 0.03/0.04 ms/window, so pool synchronization did not account for that gap.

Decode timing now separately samples `apply_pending` and the adaptation thread's join. These request-local clocks and summaries require the existing `STRATA_DECODE_TIMING` flag. They preserve blocking admission, synchronization, placement, checkpoint bytes and arithmetic; they add no GPU events. `apply_pending` includes waiting for pending copies, committing admissions and uploading residency tables, so its total is not a pure transfer timer.

Engine-only build `a518cc6f180e3208fe7970c948a37e8b49f4be9cc040da0efdecddf95c05e600` completed three basic replays at 69.3/67.5/69.0 tokens/s. Applying pending admissions cost 2.968/2.850/2.763 ms/window; thread joins cost 0.027/0.015/0.009 and the remaining residual was 0.001/0.001/0.002. Maximum apply times were 14.519/13.763/13.808 ms. Output hashes, ordered work digests, window counts and width histograms matched the preceding diagnostic. Lower CPU execution times accompanied the higher rates, so these rates do not establish a gain from instrumentation. They are excluded from benchmark medians. Scoped review found no issues; no regression suite or new quality validation ran.

This localizes a separate admission cost before each verification window. Smaller admission batches can reduce that cost while hurting future cache hits; complete unprofiled decode measurements must decide the tradeoff. The existing non-blocking admission option is not enabled because its timing-dependent CPU/GPU placement changes rounding.

The existing `--adapt-swaps 32` setting was bracketed against retained 96 using the same `a518cc6...` executable without profiling. Controls returned 65.6/63.2/63.1 and 60.5/64.7/67.6 tokens/s; the candidate returned 60.9/67.3/64.1. Candidate warm mean 65.70 lost closing control 66.15. The controls matched corresponding output hashes and draft counts; changing admission changed the candidate's placement and outputs. The smaller batch was rejected, 96 restored, and no full benchmark or regression followed.

A finer diagnostic with executable `621bb7487499660318b92a771186df5d09e87a8c0418e566e66784d74685d2b5` completed at 20:20 UTC: three basic rates 65.5/65.1/66.4, matching previous immediate-admission work/output digests. Readiness cost 2.994/2.818/3.025 ms/window; RAM commits rounded to zero and residency uploads cost 0.010/0.008/0.008. Each completed admission round waited 11.32–12.10 ms on average. The runs admitted 4991/5280/5177 experts in 52/55/54 rounds, totaling 10.02/10.58/10.37 GB of logical admission blobs. These byte counts describe scheduled admissions, not measured link traffic or proof of PCIe saturation. The component counters cover completed local/stage applications; peer work and unsuccessful non-blocking queries remain only in the outer apply timer. No GPU events or regression ran.

`STRATA_ADAPT_VERIFY_OVERLAP=1` is a default-off experiment for speculative serving with one local cache, arena expert source and CPU pool; multi-GPU stages, peer and remote caches retain original admission. At a window with pending copies, checked residency-table publication marks evictions before replay. The incoming experts stay on CPU throughout that complete verifier window. Existing asynchronous refills run alongside verification, then the existing blocking admission publishes them after the verifier has finished its graph, callbacks and CPU work. Publication follows a fixed window boundary rather than a copy-readiness query. The explicit default-stream synchronization makes the eviction table ready before the nonblocking verifier stream reads it.

This preserves original checkpoint bytes and native arithmetic, but changes CPU/GPU placement and may change outputs through their reduction differences. Extra CPU misses, PCIe contention, or queueing the small table upload behind large refills can erase the exposed-wait saving. Decode timing reports delayed-apply and eviction-publication costs separately without double counting. The scoped review found no material issues; two reporting labels/accounting details were corrected. Engine-only builds passed.

The first basic bracket completed at 20:40 UTC with executable `2b9d09d77c98ade8995139f9d457eb65049916093902989757ef9ee9fcb381b9`. Controls returned 68.5/67.2/68.4 and 67.0/66.5/68.1; candidate 64.0/69.6/66.5 tokens/s. Warm means 67.80/68.05/67.30 did not clear the gate. No full benchmark or regression followed. A CPU-clock diagnostic completed at 20:44: delayed readiness was 0.002 ms/window, but the eviction-table H2D copy plus synchronization cost 2.979/2.670/2.807 ms/window. The exposed wait moved to publication; this does not establish PCIe saturation.

The bounded follow-up publishes a request-local pinned mapped snapshot with the existing `copy_i32_from_mapped` kernel instead of an H2D copy-engine operation. Synchronization finishes the snapshot read before replay or reuse. The fixed admission boundary and eligibility guards remain unchanged. Its engine-only build passed and independent review found no issues.

Executable `ed5736b65f2c365b0f72d5d07f52a12e033305b959605f5035bd0413817b6148` completed the basic bracket at 20:56 UTC. Controls returned 67.3/66.1/67.4 and 66.2/65.9/67.3; candidate 67.7/73.3/68.9 tokens/s. Warm means 66.75/71.10/66.60 cleared the gate, prompting five alternating three-prompt runs. These basic replays do not count as benchmark medians. Repeatable gains and new quality validation remain unproven; regression has not run.

The full alternation completed at 21:04 UTC with control/candidate/control/candidate/control medians 67.3/60.8/62.9/70.2/67.2. Candidate rates were 60.8/59.8/71.3 and 69.3/70.2/80.1. One candidate lost to both neighboring controls; the other won. The gain did not repeat, so fixed-window overlap remains disabled and no regression ran. The five valid full reports are recorded in the public ledger; basic replays and diagnostics remain excluded. A CPU-clock-only component comparison is underway to check publication, readiness and verifier costs without GPU events.

That comparison completed at 21:10 UTC on the same executable. Mapped publication cost 0.296/0.295/0.298 ms/window and delayed readiness 0.001 in all three requests; immediate-admission controls still waited 2.849–3.042 ms/window for readiness. Candidate verifier time was 31.35/29.17/30.16 ms/window. Placement, routed work and speculative widths changed, so these counters do not isolate PCIe contention or explain the full-run variance. They confirm that mapped publication removed most of the earlier table-copy wait. In the older retained hardware trace, ordinary mapped int32 plan copies total only 0.290 ms/decode window, including draft/commit copies, so changing those globally has little measured upside.

### Sixteen-block fetch probe

`STRATA_FETCH_16=1` is a default-off diagnostic alternative to the retained 384-block mapped-fetch grid. It selects 16 blocks of 256 threads; the existing grid-stride loop preserves complete, exclusive coverage of the original 16-byte vectors. Unlike the failed 128-block probe, this grid has fewer blocks than the GPU's 36 SMs. That bounds the launch footprint but does not prove occupancy, contention, or an exposed speed benefit.

The engine-only build completed at 21:16 UTC with executable SHA256 `4094725d3f4f14b68a45bb508c0f17f21aaa46a018fb0f107c73d5769e20bd7a`. Basic control/candidate/control replays completed at 21:21 UTC with rates 68.5/66.4/68.9, 68.7/66.8/67.4, and 67.4/65.7/67.3 tokens/s. Their last-two warm means were 67.65/67.10/66.50. Corresponding answer hashes and aggregate draft counts matched, and the candidate activation log confirmed the 16-block grid. The candidate failed the gain threshold, so no full benchmark or regression ran; 384 blocks remain selected. These basic replays are excluded from the public full-run ledger.

### Measured MTP prefix candidate

`STRATA_MTP_COST_POLICY=1` opts into a bounded MTP prefix selector. The existing confidence cutoff remains an upper bound. Actual committed tokens and complete round costs are tracked separately for each original confidence cap and executed width; no acceptance beyond a truncated prefix is inferred. Three initial observations per width alternate by sample count. Every sixteenth eligible offer rotates a width probe; other offers shorten only when the observed tokens/ms ratio beats the full cap by the existing 3% margin. Paired token/cost EMAs use weight 0.05. First, lookup, EOS and terminal incomplete rounds do not train this selector.

The measured round includes admission waiting, verification, commit, next drafting and adaptation joins. The original Q2 MTP chain and original IQ3_S target verification remain in use. Width selection changes catch-up work, expert grouping and residency evolution, so answer identity and speed are unproven. Confidence caps are coarse context groups and the small bootstrap can be noisy. The refined engine-only build completed at 21:35 UTC, SHA256 `c97ba62ddcad39e16aea4f08046550f5fa97dbfc69c9b4f50b78ffc78d92a64a`; scoped review found no remaining issues. A basic control/candidate/control bracket is running. No regression has run for this candidate.

[SpecDec++](https://arxiv.org/abs/2405.19715) motivates adapting candidate length to acceptance and cost, using a trained acceptance head. Selected primary excerpts describe experiments with Llama-2 and two A100-80G GPUs. This empirical local controller does not implement that head or inherit the paper's guarantees or gains. [SpecMoEOff](https://arxiv.org/abs/2508.21706) likewise motivates tuning speculation for offloading hardware and workload; its reported gains do not predict this server's result.

### Earlier validation

The initial CUDA 13, SM 120 build completed. Of 75 registered CTest cases, 73 passed. `ple_parity` lacked its relative Q2_0 GGUF fixture; `expert_multi_test` requires AVX-512 VNNI/VBMI unavailable on this EPYC. These two failures are reported rather than hidden. The IQ3_S runtime uses the supported AVX2 path.

Targeted checks cover five PCIe-selection fixtures, manifest rejection, native grouped experts, native Q8_0 expert parity, and 70 graph-captured projection cases per arithmetic mode. Projection outputs are compared with an independent double-precision CPU reference, with BF16 rounding only for eligible cases, relative L2 error at most 3e-6, finite outputs, and untouched stride padding. Literal Python fixtures check runtime role boundaries and the Q5_0 scale, high-bit plane, low-nibble plane, zero blocks, and complete rows.

The CUDA dispatch also checks the current device and the loaded kernel's PTX target. A compute-75 PTX image running on SM120 failed 20 of 70 cases before this guard, then passed all 70 by falling back to FP32. That regression is registered as `bf16_decode_tc_ptx75_fallback`. Mixed-device switching is covered by the dispatch logic but has not been tested on multiple physical GPUs here. The existing Q5_0/Q8_0 and Q4_K/Q5_0 expert parity checks exercise Q5 in each role separately; they do not establish combined Q5_0/Q5_0 draft numerical parity.

After the portability fixes, a fresh incremental build and all nine targeted CTest cases passed. All 15 Python MTP tests passed. This does not replace the full-suite limitations reported above.

The AVX2 trial adds two 48-case CPU tests. The enabled test failed 12 single-input cases before implementation; both tests passed afterward, along with the nine earlier targeted checks. They cover the existing GGML default, the existing IQ256 reduction when enabled, partial row writes, inactive output buffers, finite values, input widths 512/2560, and row counts 640/768. Both tests explicitly clear the inherited `STRATA_NO_IQ256` switch; they also passed with that switch set in the parent environment.

Trial builds compile only the engine and use short generation/throughput checks. Regression followed the repeatable rowwarp improvement, as requested; its focused results are above. The earlier test results do not validate the newer expanded cache, CPU fusion or shared GPU codebook. Those failed options remain disabled. A fresh full-suite pass and physical HIP validation are not claimed. This branch records a validated small gain and failed experiments, not a completed90 tokens/s optimization.
