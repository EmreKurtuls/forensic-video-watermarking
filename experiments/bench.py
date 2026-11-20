import os, sys, time, tracemalloc, secrets
sys.path.insert(0, os.path.dirname(__file__))
from ciphers import present128_round_keys, present_encrypt_block, present_ctr, FF1, siphash24
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

rk0 = present128_round_keys(0)
print("PRESENT-128 KAT k=0,p=0:", hex(present_encrypt_block(0, rk0)), "expect 0x96db702a2e6900af")
ff = FF1(bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C"), 10)
print("FF1 KAT sample1:", ff.encrypt_digits("0123456789"), "expect 2433477484")
print("FF1 KAT sample2:", ff.encrypt_digits("0123456789", bytes.fromhex("39383736353433323130")), "expect 6124200773")
print("SipHash KAT:", hex(siphash24(bytes(range(16)), b"")), "expect 0x726fdb47dd0e0e31")

def bench(fn, reps):
    tracemalloc.start()
    t0 = time.perf_counter(); c0 = time.process_time()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter(); c1 = time.process_time()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (t1 - t0) / reps, (c1 - c0) / reps, peak

key = secrets.token_bytes(16)
rks = present128_round_keys(int.from_bytes(key, "big"))
ff2 = FF1(key, 2)
aes = AESGCM(secrets.token_bytes(32)); cha = ChaCha20Poly1305(secrets.token_bytes(32))
pt64 = secrets.randbits(64)

print("\n== Small payload: one 64-bit forensic ID ==")
rows = [
    ("PRESENT-128 (1 block)", lambda: present_encrypt_block(pt64, rks), 64),
    ("FF1 radix-2, n=64, tweak=film_id", lambda: ff2.encrypt_bits(pt64, 64, b"film0042"), 64),
    ("AES-256-GCM (ref)", lambda: aes.encrypt(os.urandom(12), pt64.to_bytes(8, "big"), None), 64 + 96 + 128),
    ("ChaCha20-Poly1305 (ref)", lambda: cha.encrypt(os.urandom(12), pt64.to_bytes(8, "big"), None), 64 + 96 + 128),
]
print(f"{'cipher':36s} {'us/op':>9s} {'bits embedded':>14s}")
for name, fn, bits in rows:
    w, c, p = bench(fn, 2000)
    print(f"{name:36s} {w*1e6:9.1f} {bits:14d}")
w, c, p = bench(lambda: siphash24(key, pt64.to_bytes(8, 'big')), 2000)
print(f"{'+ SipHash-2-4 MAC (32-bit trunc)':36s} {w*1e6:9.1f} {'+32':>14s}")

print("\n== Large data ==")
print(f"{'cipher':36s} {'size':>7s} {'MB/s':>9s} {'cpu s':>8s} {'peak KB':>9s}")
for size in (1024, 64 * 1024, 1024 * 1024):
    data = os.urandom(size)
    x = int.from_bytes(data, "big")
    cases = [
        ("PRESENT-128 CTR", lambda: present_ctr(data, rks, 7)),
        ("FF1 whole message (radix 2)", lambda: ff2.encrypt_bits(x, size * 8, b"film0042")),
    ]
    if size <= 64 * 1024:
        def ff1_chunked():
            for i in range(0, size, 8):
                ff2.encrypt_bits(int.from_bytes(data[i:i+8], "big"), 64, i.to_bytes(8, "big"))
        cases.append(("FF1 chunked 64-bit, tweak=index", ff1_chunked))
    cases += [
        ("AES-256-GCM (ref)", lambda: aes.encrypt(os.urandom(12), data, None)),
        ("ChaCha20-Poly1305 (ref)", lambda: cha.encrypt(os.urandom(12), data, None)),
    ]
    for name, fn in cases:
        reps = 3 if size >= 64 * 1024 else 20
        w, c, p = bench(fn, reps)
        print(f"{name:36s} {size//1024:5d}KB {size/w/1e6:9.3f} {c:8.3f} {p/1024:9.0f}")
