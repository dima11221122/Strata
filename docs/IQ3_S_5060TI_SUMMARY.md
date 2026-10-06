# IQ3_S optimization and deployment status

October 6, 2026. Original Qwen3.8-Flash-Next IQ3_S on RTX 5060 Ti 16 GB, EPYC 7K62 and 98 GiB host RAM. Results concern one active request.

## Completed

- Forked upstream to [dima11221122/Strata](https://github.com/dima11221122/Strata), branch `perf/qwen-iq3s-5060ti`, with rollback binaries and configurations.
- Calibrated CPU/GPU placement, speculation and memory. Installer defaults measured **53.3 tokens/s**; calibration reached **67.7**.
- Retained Q6 rowwarp and the English/code draft vocabulary. Repeated English-head results were **70.3 and 69.8 tokens/s**, about **4%** above their bracket controls. These used 65,536 capacity and short prompts.
- Investigated CPU scheduling, quantized kernels, tensor cores, fusion, offloading, fetch/admission overlap, draft precision, vocabulary storage and alignment training. Recorded primary research, measurements and rejected candidates.
- Ran focused checks after retained gains: Python assertions, 20,534-token retrieval, short multilingual replies, authentication and Pi integration. Published **152 valid full benchmark reports**; basic and allocation-only attempts are separate.

## Best validated setup

Original IQ3_S target; canonical 675 MiB Q2 MTP drafter; 40,525 English/code draft rows; 20 CPU workers; Q6 rowwarp; overlapped mapped fetch; PCIe fraction 0.20; draft cap 4 and confidence 0.70; verifier capacity 6 through suffix drafting. INT8 KV with 32,768 resident cells and host-RAM PLE. Configured reserve: 700 MiB. Actual expert cache: **4,601 slots / 8.77 GiB**. CLI `--expert-cache 3538` represents a converted byte budget, not the actual slot count. Multilingual drafting can be less effective with this subset.

Validated executable SHA256: `3378aa4130f6a3fa4e5ff446f5c43809f9d434b0fed33fb652a41103b625c3a7`.

## Not achieved

**90 tokens/s remains unmet.** No hardware ceiling was proved. Most experimental gains disappeared on repetition and remain disabled. The latest TF32 HC-up full bracket measured 67.5 / 65.6 / 67.0 / 64.6 / 67.5 and failed.

Mixed-Q3 drafting missed the fixed cache allocation requirement. The trained final mixer lost held-out yield. PDL lacked sufficient measured opportunity. Lossless BF16 compression has a size estimate, but no implemented codec or runtime gain. A fresh complete regression suite and broad long-context quality evaluation were not run.

## Running at 128k from the laptop

The validated engine is **running at 131,072 capacity**, preserving the setup above. Pi 1.0.2 is configured for this provider, thinking off, and 8,192 output tokens. Normal replies and an actual local `read` tool call passed.

The unchanged **121,551-input-token** retrieval prompt returned all three facts correctly in **90.5 seconds**: engine prefill 82.8 seconds, decode **54.5 tokens/s**. A separate short benchmark measured **66.5 tokens/s median** at 128k capacity. These checks do not establish repeated long-context performance.

Pi initially estimated too many input tokens and clamped output to one. A Strata-only `samplingParams.max_tokens=8192` setting fixed it; the server still enforces its actual tokenizer limit. Post-check GPU headroom was **255 MiB**, 1 MiB below the earlier trial floor: no OOM occurred, but memory is tight.

With the existing SSH tunnel, start `pi`; Strata is the default. Explicit selection: `pi --model strata/Qwen3.8-Flash-Next-IQ3_S`.

Evidence: [128k checks](IQ3_S_128K_CHECK.json), [research](IQ3_S_5060TI_RESEARCH.md), [full benchmarks](IQ3_S_5060TI_RESULTS.json).
