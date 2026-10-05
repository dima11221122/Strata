# IQ3_S on a 16 GB RTX 5060 Ti

This fork investigates 90 tokens/s for one request using the original Qwen3.8-Flash-Next IQ3_S weights. The best observed three-prompt median so far is 69.1 tokens/s. The target has not been reached. This document records hypotheses and measurements; it does not promise that this hardware can reach the target.

## Measurement contract

- Hardware: RTX 5060 Ti 16 GB, PCIe Gen4 x8, EPYC 7K62, 23 vCPUs, 98 GiB RAM, Linux, CUDA 13.
- Upstream engine: `6f32ec070f23ced9f50e704d854d775da52591ab`.
- GGUF revision: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_S, both shards including PLE.
- One request at a time. Three distinct prompts: Python LRU cache, database recovery, TypeScript task queue. Each generates 512 tokens, greedy, reasoning off. Use the median of the engine's decode rates, with each individual rate recorded. Repeat promising candidates and use held-out prompts.
- Keep authentication, 65,536-token capacity, tool calls, and both model shards. Report separate numbers for reduced context or changed weight quantization; those do not satisfy the original target.
- Profiles and credentials stay outside the checkout. GPU event profiling changes scheduling, so its rate is diagnostic only.

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

## Experimental sequence

- [x] Fork upstream and preserve a calibrated baseline.
- [x] Compare a workload cache profile on separate benchmark prompts.
- [x] Enable CPU/GPU decode stage timings and record wait, pool, draft and dense-kernel costs.
- [ ] Compare speculative window depths 2, 4, 6, 8 and confidence floors, without changing target verification.
- [ ] Compare INT8 and Q4 KV, and 32K versus 20K resident KV while keeping 64K total capacity. Check long-context retrieval before retaining lower precision.
- [ ] Compare transfer modes and CPU codebook gather/prefetch settings. Each run changes one factor from baseline.
- [ ] Test measured-cost expert placement against routing-order placement with a pure scheduler test and native expert parity checks.
- [ ] Profile and optimize a dominant CUDA kernel; use exact arithmetic order where possible, microbenchmarks, then complete decode benchmarks.
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

### Placement

`STRATA_PCIE_REUSE=1` selects the missed, pinned experts serving the most input rows, subject to the same transfer quota and staging limit. Ties retain the old last-in-routing-order choice. Resident and peer-GPU experts are excluded. This transfers the same expert blob once for multiple inputs and leaves CPU/GPU result merging unchanged.

### Projection arithmetic

`STRATA_BF16_DECODE_TC=1` rounds eligible FP32 activations to BF16 and uses WMMA with FP32 accumulation. It only applies to 2–8 token windows, input dimensions divisible by 16 and no larger than 4096, and output dimensions divisible by 16 and at least 1024. Other shapes use the original kernel. Eight warps split the reduction; microbenchmarks alone do not establish an end-to-end gain. This changes arithmetic and requires answer-quality checks. It remains disabled by default.

### Draft precision

`tools/mtp_pack.py --experts q5_0` and `--experts q8_0`, followed by `tools/mtp_rt.py`, produce optional draft packs. These change the MTP routed experts only. Dense draft weights and the vocabulary subset must remain identical between arms. The main IQ3_S model is unchanged.

The runtime validates `experts.txt` and the exact expert file size before loading. Missing manifests preserve the legacy canonical Q2_0 path. Native Q5_0/Q8_0 packs use original GGUF block order and Q8_1 activations in the existing native expert kernel. Q8_0 experts occupy 2.49 GiB versus 675 MiB for Q2_0, reducing target-cache slots from 4601 to 3629. In the first sweep, draft acceptance was about 81% for both formats; deeper Q8 speculation reduced it to about 73% and was slower.

### Deployment provenance

Initial engine experiments accidentally launched the installer binary because `exe` still pointed at `engine/strata`. Those engine-variant reports are invalid and excluded above. At 08:34 UTC on 2026-10-05, the config was changed to `build/strata`. Each subsequent run checks the configured and live executable hashes; engine variants also require the built hash to match. The table above used SHA-256 `30439454c75f8ad50b4dcf14dbe5bcd91c3e6d340fb02f50ae6a3290935f30f3`.

## Verification limits

The initial CUDA 13, SM 120 build completed. Of 75 registered CTest cases, 73 passed. `ple_parity` lacked its relative Q2_0 GGUF fixture; `expert_multi_test` requires AVX-512 VNNI/VBMI unavailable on this EPYC. These two failures are reported rather than hidden. The IQ3_S runtime uses the supported AVX2 path.

Targeted checks cover five PCIe-selection fixtures, manifest rejection, native grouped experts, native Q8_0 expert parity, and 70 graph-captured projection cases per arithmetic mode. Projection outputs are compared with an independent double-precision CPU reference, with BF16 rounding only for eligible cases, relative L2 error at most 3e-6, finite outputs, and untouched stride padding. Literal Python fixtures check runtime role boundaries and the Q5_0 scale, high-bit plane, low-nibble plane, zero blocks, and complete rows.

The CUDA dispatch also checks the current device and the loaded kernel's PTX target. A compute-75 PTX image running on SM120 failed 20 of70 cases before this guard, then passed all70 by falling back to FP32. That regression is registered as `bf16_decode_tc_ptx75_fallback`. Mixed-device switching is covered by the dispatch logic but has not been tested on multiple physical GPUs here. The existing Q5_0/Q8_0 and Q4_K/Q5_0 expert parity checks exercise Q5 in each role separately; they do not establish combined Q5_0/Q5_0 draft numerical parity.
