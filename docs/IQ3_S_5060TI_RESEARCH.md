# IQ3_S on a 16 GB RTX 5060 Ti

This fork investigates 90 tokens/s for one request using the original Qwen3.8-Flash-Next IQ3_S weights. The current measured median is 67.7 tokens/s. This document records hypotheses and measurements; it does not promise that this hardware can reach the target.

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
3. [MoE-SpeQ](https://arxiv.org/abs/2511.14102): lookahead-aware expert scheduling and a governor trade additional speculative work against memory and transfer costs. Strata's MTP is one dense draft layer and does not produce the full target's expert routes. Replacing it with a second full MoE draft would add substantial memory and complexity. Start with the existing router lookahead and window-level reuse, not another model.
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
