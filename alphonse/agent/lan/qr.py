"""Minimal QR generator for short ASCII codes (Version 1-L only).

This is intentionally tiny and supports only byte mode payloads up to 17 bytes,
which is enough for pairing codes. It avoids external dependencies.
"""

from __future__ import annotations

from typing import Iterable


def render_svg_qr(text: str, scale: int = 4, border: int = 2) -> str:
    qr = _QrV1L.encode(text)
    size = qr.size
    dim = (size + border * 2) * scale
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{dim}" height="{dim}"',
        f' viewBox="0 0 {dim} {dim}" shape-rendering="crispEdges">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for y in range(size):
        for x in range(size):
            if qr.modules[y][x]:
                parts.append(
                    f'<rect x="{(x + border) * scale}" y="{(y + border) * scale}"'
                    f' width="{scale}" height="{scale}" fill="#000000"/>'
                )
    parts.append("</svg>")
    return "".join(parts)


def render_ascii_qr(text: str) -> str:
    qr = _QrV1L.encode(text)
    size = qr.size
    border = 1
    lines = []
    for y in range(-border, size + border):
        row = []
        for x in range(-border, size + border):
            module = 0 <= x < size and 0 <= y < size and qr.modules[y][x]
            row.append("██" if module else "  ")
        lines.append("".join(row))
    return "\n".join(lines)


class _QrV1L:
    # Version 1, Error Correction L
    size = 21
    data_codewords = 19
    ecc_codewords = 7

    def __init__(self) -> None:
        self.modules = [[False] * self.size for _ in range(self.size)]
        self.is_function = [[False] * self.size for _ in range(self.size)]

    @classmethod
    def encode(cls, text: str) -> "_QrV1L":
        data = text.encode("utf-8")
        if len(data) > 17:
            raise ValueError("QR payload too long for version 1-L")
        qr = cls()
        qr._draw_function_patterns()
        codewords = qr._build_codewords(data)
        qr._draw_codewords(codewords)
        qr._apply_mask_0()
        qr._draw_format_bits()
        return qr

    def _draw_function_patterns(self) -> None:
        self._draw_finder(3, 3)
        self._draw_finder(self.size - 4, 3)
        self._draw_finder(3, self.size - 4)
        for i in range(self.size):
            self._set_function(6, i, i % 2 == 0)
            self._set_function(i, 6, i % 2 == 0)
        self._set_function(8, self.size - 8, True)
        self._reserve_format_bits()

    def _draw_finder(self, x: int, y: int) -> None:
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                dist = max(abs(dx), abs(dy))
                self._set_function(x + dx, y + dy, dist != 2 and dist != 4)

    def _set_function(self, x: int, y: int, value: bool) -> None:
        if 0 <= x < self.size and 0 <= y < self.size:
            self.modules[y][x] = value
            self.is_function[y][x] = True

    def _reserve_format_bits(self) -> None:
        # Reserve format info modules (set as function without overwriting values).
        coords = []
        for i in range(9):
            if i != 6:
                coords.append((8, i))
                coords.append((i, 8))
        for i in range(8):
            coords.append((self.size - 1 - i, 8))
            coords.append((8, self.size - 1 - i))
        for x, y in coords:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.is_function[y][x] = True

    def _build_codewords(self, data: bytes) -> bytes:
        bb = _BitBuffer()
        bb.append_bits(0b0100, 4)  # byte mode
        bb.append_bits(len(data), 8)
        for b in data:
            bb.append_bits(b, 8)
        # Terminator
        bb.append_bits(0, 4)
        # Pad to byte boundary
        bb.append_bits(0, (8 - len(bb) % 8) % 8)
        # Pad to data codewords
        pad = 0xEC
        while len(bb) < self.data_codewords * 8:
            bb.append_bits(pad, 8)
            pad ^= 0xEC ^ 0x11
        data_codewords = bb.get_bytes()
        ecc = _reed_solomon_remainder(data_codewords, self.ecc_codewords)
        return data_codewords + ecc

    def _draw_codewords(self, data: bytes) -> None:
        i = 0
        total_bits = len(data) * 8
        for right in range(self.size - 1, 0, -2):
            if right == 6:
                right = 5
            for vert in range(self.size):
                y = self.size - 1 - vert if (right + 1) % 4 == 0 else vert
                for x in (right, right - 1):
                    if not self.is_function[y][x]:
                        if i >= total_bits:
                            return
                        bit = (data[i >> 3] >> (7 - (i & 7))) & 1
                        self.modules[y][x] = bit == 1
                        i += 1

    def _apply_mask_0(self) -> None:
        for y in range(self.size):
            for x in range(self.size):
                if not self.is_function[y][x] and (x + y) % 2 == 0:
                    self.modules[y][x] = not self.modules[y][x]

    def _draw_format_bits(self) -> None:
        # EC Level L (01) + mask 0 -> format bits after BCH
        format_bits = 0b111011111000100
        for i in range(15):
            bit = ((format_bits >> i) & 1) != 0
            if i < 6:
                self._set_function(8, i, bit)
            elif i < 8:
                self._set_function(8, i + 1, bit)
            else:
                self._set_function(8, self.size - 15 + i, bit)
            if i < 8:
                self._set_function(self.size - 1 - i, 8, bit)
            else:
                self._set_function(14 - i, 8, bit)


class _BitBuffer(list[int]):
    def append_bits(self, val: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            self.append((val >> i) & 1)

    def get_bytes(self) -> bytes:
        out = bytearray()
        val = 0
        for i, bit in enumerate(self):
            val = (val << 1) | bit
            if i % 8 == 7:
                out.append(val)
                val = 0
        return bytes(out)


def _reed_solomon_remainder(data: bytes, degree: int) -> bytes:
    result = [0] * degree
    poly = _reed_solomon_divisor(degree)
    for b in data:
        factor = b ^ result[0]
        result = result[1:] + [0]
        for i in range(degree):
            result[i] ^= _rs_multiply(factor, poly[i])
    return bytes(result)


def _reed_solomon_divisor(degree: int) -> list[int]:
    result = [1]
    root = 1
    for _ in range(degree):
        result = _poly_multiply(result, [1, root])
        root = _rs_multiply(root, 0x02)
    return result


def _poly_multiply(p: Iterable[int], q: Iterable[int]) -> list[int]:
    p_list = list(p)
    q_list = list(q)
    out = [0] * (len(p_list) + len(q_list) - 1)
    for i, a in enumerate(p_list):
        for j, b in enumerate(q_list):
            out[i + j] ^= _rs_multiply(a, b)
    return out


def _rs_multiply(x: int, y: int) -> int:
    z = 0
    for _ in range(8):
        if y & 1:
            z ^= x
        carry = x & 0x80
        x = (x << 1) & 0xFF
        if carry:
            x ^= 0x1D
        y >>= 1
    return z
