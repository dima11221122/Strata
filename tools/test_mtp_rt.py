"""Lossless runtime expert layout and input guards; no model or GPU needed."""
import unittest
import numpy as np
from mtp_rt import native_blob_of


class NativeRuntimeTest(unittest.TestCase):
    def test_q5_gate_up_down_remain_contiguous(self):
        gu = np.empty((1280, 1760), dtype=np.uint8)
        gu[:640] = 17
        gu[640:] = 34
        dn = np.full((2560, 440), 51, dtype=np.uint8)
        blob = native_blob_of(gu, dn, 22)
        self.assertEqual(len(blob), 3379200)
        for offset, value in [(0, 17), (1760, 17), (1126399, 17), (1126400, 34),
                              (2252799, 34), (2252800, 51), (3379199, 51)]:
            self.assertEqual(blob[offset], value)

    def test_gate_up_down_remain_contiguous(self):
        gu = np.empty((1280, 2720), dtype=np.uint8)
        gu[:640] = 17
        gu[640:] = 34
        dn = np.full((2560, 680), 51, dtype=np.uint8)
        blob = native_blob_of(gu, dn, 34)
        self.assertEqual(len(blob), 5222400)
        for offset, value in [(0, 17), (640, 17), (2720, 17), (1740799, 17),
                              (1740800, 34), (3481599, 34), (3481600, 51), (5222399, 51)]:
            self.assertEqual(blob[offset], value)

    def test_wrong_expert_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            native_blob_of(np.zeros((1280, 720), dtype=np.uint8), np.zeros((2560, 680), dtype=np.uint8), 34)
        with self.assertRaises(ValueError):
            native_blob_of(np.zeros((1280, 2720), dtype=np.uint8), np.zeros((2560, 180), dtype=np.uint8), 34)

    def test_non_byte_storage_is_rejected(self):
        with self.assertRaises(ValueError):
            native_blob_of(np.zeros((1280, 2720), dtype=np.uint16), np.zeros((2560, 680), dtype=np.uint8), 34)


if __name__ == "__main__":
    unittest.main()
