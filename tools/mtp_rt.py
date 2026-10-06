"""tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.

    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt

Writes
  experts.bin   512 routed experts in the engine's blob layout (`include/strata/kernels/cpu/expert.hpp`): gate/up
                rows interleaved (2r = gate r, 2r+1 = up r), then down rows; the Q2_0 codes in one plane and the fp16
                scales in another.  A lossless relayout of the GGUF's Q2_0 blocks (same bytes, `cpu_expert_fixture.py`).
                Native Q5_0/Q8_0 and mixed Q3_K GU/Q2_0 down use original GGUF blocks:
                gate rows, up rows, then down rows.
  experts.txt   expert format and geometry; missing files retain the legacy Q2_0 path.
  dense.bin     every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding) so the
                engine's multi-column MMVQ can run them; the hyper-connection and router weights kept BF16; the
                RMSNorm weights as F32 with the Gemma "+1" applied (vLLM GemmaRMSNorm scales by 1 + w).
  dense.txt     one line per tensor: name kind rows cols offset bytes   (kind = q8_0 | bf16 | f32)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

H, FF, NE = 2560, 640, 512
BLOB = 3 * (H * FF * 18 // 64)
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """gu: (1280, 720) Q2_0 rows (gate rows 0..639, up rows 640..1279); dn: (2560, 180)."""
    gub = gu.reshape(2 * FF, H // 64, 18)
    inter = np.empty_like(gub)
    inter[0::2] = gub[:FF]
    inter[1::2] = gub[FF:]
    dnb = dn.reshape(H, FF // 64, 18)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, len(out)
    return out


def native_blob_of(gu: np.ndarray, dn: np.ndarray, block_bytes: int) -> bytes:
    """Raw GGUF roles: matching Q5/Q8, or 110-byte Q3_K GU with Q2_0 down."""
    if block_bytes == 110:
        gu_row_bytes, dn_row_bytes = H // 256 * 110, FF // 64 * 18
    elif block_bytes in (22, 34):
        gu_row_bytes, dn_row_bytes = H // 32 * block_bytes, FF // 32 * block_bytes
    else:
        raise ValueError("Unsupported MTP native gate/up block size")
    if gu.dtype != np.uint8 or dn.dtype != np.uint8 or gu.shape != (2 * FF, gu_row_bytes) or dn.shape != (H, dn_row_bytes):
        raise ValueError("MTP native experts need byte arrays with the original gate/up/down dimensions")
    return gu.tobytes() + dn.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    gu_tensor = tens["mtp.layers.0.mlp.experts.gate_up_proj"]
    dn_tensor = tens["mtp.layers.0.mlp.experts.down_proj"]
    gu, dn = np.asarray(gu_tensor.data), np.asarray(dn_tensor.data)
    formats = {gguf.GGMLQuantizationType.Q2_0: ("canonical-q2_0", 64, 18),
               gguf.GGMLQuantizationType.Q5_0: ("native-q5_0", 32, 22),
               gguf.GGMLQuantizationType.Q8_0: ("native-q8_0", 32, 34)}
    if gu_tensor.tensor_type == gguf.GGMLQuantizationType.Q3_K and dn_tensor.tensor_type == gguf.GGMLQuantizationType.Q2_0:
        expert_kind, native = "native-q3_k-q2_0", True
        block_bytes = 110
        gu_bytes, dn_bytes = H // 256 * 110, FF // 64 * 18
    else:
        if gu_tensor.tensor_type not in formats or dn_tensor.tensor_type != gu_tensor.tensor_type:
            raise ValueError("MTP runtime supports matching Q2_0/Q5_0/Q8_0, or Q3_K gate/up with Q2_0 down")
        expert_kind, block, block_bytes = formats[gu_tensor.tensor_type]
        native = block == 32
        gu_bytes, dn_bytes = H // block * block_bytes, FF // block * block_bytes
    if gu.shape != (NE, 2 * FF, gu_bytes) or dn.shape != (NE, H, dn_bytes):
        raise ValueError(f"Incompatible expert shapes: {gu.shape}, {dn.shape}")
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            f.write(native_blob_of(gu[e], dn[e], block_bytes) if native else blob_of(gu[e], dn[e]))
    (out / "experts.txt").write_text(f"{expert_kind} {H} {FF} {NE}\n", encoding="utf-8")
    lines = []
    off = 0
    with open(out / "dense.bin", "wb") as f:
        for name, t in tens.items():
            if "experts." in name:
                continue
            short = name[len("mtp."):]
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            data = np.asarray(t.data)
            if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
                arr = (data.astype(np.float32) + 1.0)
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:                              # BF16
                u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
                u16 = u16.reshape(data.shape[0], -1)
                rows, cols = u16.shape
                if short in Q8:
                    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                    raw, kind = q8_0(f32), "q8_0"
                else:
                    raw, kind = u16.tobytes(), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {(out / 'experts.bin').stat().st_size} B ({expert_kind}), dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
