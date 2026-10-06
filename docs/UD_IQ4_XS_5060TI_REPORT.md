# UD-IQ4_XS deployment and performance

October 6, 2026. **Qwen3.8-Flash-Next-UD-IQ4_XS is deployed at 131,072 context capacity, with MTP enabled and working from laptop Pi.** It is slower than the earlier IQ3_S deployment on this RTX 5060 Ti 16 GB / EPYC 7K62 server.

## Performance

The same three short prompts requested 512 output tokens each, with thinking off and one active request. Clean runs after reboot measured:

| Run | Python LRU | Crash recovery | TypeScript queue | Median tokens/s |
| --- | ---: | ---: | ---: | ---: |
| First | 42.4 | 39.5 | 42.2 | 42.2 |
| Repeat | 37.9 | 38.0 | 51.6 | 38.0 |
| Fresh IQ3_S baseline before rollout | 64.5 | 64.7 | 76.2 | 64.7 |

Combined UD median: **40.85 tokens/s**, about **37% below** the fresh IQ3_S baseline taken before this rollout. That baseline is a separate run from the historical 66.5 median in the IQ3_S summary. This is an unpaired deployment comparison: cache allocation, reserve and driver also changed. It does not isolate quantization cost or establish a hardware ceiling. MTP accepted 1,772 of 2,179 offered drafts across the six clean short requests.

Actual Pi reply and local `read` tool checks passed. A **121,553-input-token** retrieval returned all three facts correctly in **116.0 seconds**: engine prefill 109.418 seconds at 1,110.9 tokens/s; decode 36.4 tokens/s for 122 outputs. MTP accepted 80/94 drafts. This verifies one long request at 128k capacity, not sustained full-context performance.

## Active setup

- Validated engine 0.1.39; original executable SHA recorded in the evidence.
- Canonical Q2 MTP: 675 MiB expert weights, 40,525 English/code head rows, cap 4, confidence 0.70; verifier capacity 6.
- 20 CPU workers; Q6 rowwarp; overlapped mapped fetch; PCIe fraction 0.20.
- INT8 KV, 32,768 resident GPU cells, host-RAM PLE.
- Automatic native cache: **3,451 actual slots / 7.77 GiB**. Host expert complement: **47.67 GiB**, within a 56 GiB budget.
- **1,024 MiB configured reserve**; 476 MiB GPU memory free after the long check. The initial 700 MiB setting fell below the 256 MiB floor and was replaced before clean benchmarking.

Pi defaults to `strata/Qwen3.8-Flash-Next-UD-IQ4_XS`, thinking off, with 8,192 output tokens. Reply/tool tests, authenticated access and tunnel reconnection after reboot passed. Start a new `pi` session to use the default.

## Limits and recovery

Concurrent Ubuntu maintenance caused a driver/library mismatch; its affected runs were excluded. Package configuration completed, `dpkg --audit` was clean, and reboot aligned driver 580.178.04 with kernel 6.8.0-138. Automatic update timers remain enabled.

All model weights stay on the server. IQ3_S weights and the partial laptop backup were deleted on request; switching back requires a server download. UD remains active. Broad quality improvement and 90 tokens/s are unproven; no full regression suite was run after this slower trial.

Evidence: [sanitized measurements](UD_IQ4_XS_5060TI_CHECK.json), [historical IQ3_S campaign](IQ3_S_5060TI_SUMMARY.md).
