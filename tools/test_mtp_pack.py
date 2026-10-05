"""Literal GGML block fixtures catch wrong scales, bit planes, and row layout."""
import unittest
import numpy as np
import mtp_pack


class Q5DraftPackTest(unittest.TestCase):
    def test_q5_block_uses_ggml_scale_and_bit_planes(self):
        self.assertIn("q5_0", mtp_pack.QUANT)
        quantize, qtype, block, size = mtp_pack.QUANT["q5_0"]
        weights = np.arange(-16, 16, dtype=np.float32).reshape(1, 32)
        blob = quantize(weights)
        self.assertEqual(blob.tobytes().hex(), "003c0000ffff00112233445566778899aabbccddeeff")
        np.testing.assert_array_equal(mtp_pack.dequant("q5_0", blob, 32), weights.ravel())

    def test_q5_zero_and_multiple_rows_reconstruct(self):
        self.assertIn("q5_0", mtp_pack.QUANT)
        quantize = mtp_pack.QUANT["q5_0"][0]
        weights = np.stack([np.zeros(32), np.arange(-16, 16), -np.arange(-16, 16)]).astype(np.float32)
        blob = quantize(weights)
        self.assertEqual(blob.size, 66)
        reconstructed = mtp_pack.dequant("q5_0", blob, 96)
        self.assertTrue(np.isfinite(reconstructed).all())
        np.testing.assert_array_equal(reconstructed, weights.ravel())

    def test_q5_partial_rows_are_rejected(self):
        self.assertIn("q5_0", mtp_pack.QUANT)
        with self.assertRaises(ValueError):
            mtp_pack.QUANT["q5_0"][0](np.ones((2, 31), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
