import random, ext
random.seed(1); L = []
for _ in range(20000):
    x = random.choice([random.randint(-8 << 16, 8 << 16), random.randint(-2**31, 2**31 - 1), random.randint(-14 << 16, 14 << 16)]) & 0xFFFFFFFF
    m = random.choice([random.randint(1, 2**32 - 1), random.randint(1, 1 << 20), 1 << random.randint(0, 31)])
    a, b = random.getrandbits(32), random.getrandbits(32)
    L.append((x, ext.exp2q(x), ext.sigmq(x) & 0xFFFFFFFF, m, ext.rsqq(m), a, b, ext.rot16(a, b)))
open("ext_vectors.txt", "w").write("\n".join(" ".join(str(v) for v in r) for r in L))
