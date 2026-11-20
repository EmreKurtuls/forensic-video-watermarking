M64 = (1 << 64) - 1


def siphash24(key: bytes, msg: bytes) -> int:
    if len(key) != 16:
        raise ValueError("SipHash key must be 16 bytes")
    k0 = int.from_bytes(key[:8], "little")
    k1 = int.from_bytes(key[8:], "little")
    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    def rounds(n, v0, v1, v2, v3):
        for _ in range(n):
            v0 = (v0 + v1) & M64; v1 = ((v1 << 13) | (v1 >> 51)) & M64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & M64
            v2 = (v2 + v3) & M64; v3 = ((v3 << 16) | (v3 >> 48)) & M64; v3 ^= v2
            v0 = (v0 + v3) & M64; v3 = ((v3 << 21) | (v3 >> 43)) & M64; v3 ^= v0
            v2 = (v2 + v1) & M64; v1 = ((v1 << 17) | (v1 >> 47)) & M64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & M64
        return v0, v1, v2, v3

    end = len(msg) - len(msg) % 8
    for i in range(0, end, 8):
        m = int.from_bytes(msg[i:i + 8], "little")
        v3 ^= m
        v0, v1, v2, v3 = rounds(2, v0, v1, v2, v3)
        v0 ^= m
    b = ((len(msg) & 0xFF) << 56) | int.from_bytes(msg[end:], "little")
    v3 ^= b
    v0, v1, v2, v3 = rounds(2, v0, v1, v2, v3)
    v0 ^= b
    v2 ^= 0xFF
    v0, v1, v2, v3 = rounds(4, v0, v1, v2, v3)
    return v0 ^ v1 ^ v2 ^ v3


def siphash24_bytes(key: bytes, msg: bytes) -> bytes:
    return siphash24(key, msg).to_bytes(8, "little")
