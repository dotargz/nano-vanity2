// GPU search kernel (CUDA). Compiled to PTX by build.rs when nvcc is
// available; the checked-in kernel.ptx is used otherwise.
//
// Same algorithm as the CPU engines: per key one or two BLAKE2b
// compressions, a table-only fixed-base multiplication without doublings,
// and one field inversion shared by a batch of keys.
//
// Field elements are 8 little-endian 32-bit limbs holding any value below
// 2^256, taken modulo p = 2^255 - 19 (so 2^256 = 38). Multi-limb arithmetic
// uses PTX carry chains; each chain lives in a single asm statement.

typedef unsigned int u32;
typedef unsigned long long u64;

#define INL __device__ __forceinline__
#define BMAX 64
#define MAXFIX 60
#define MAXSEG 8
#define MAXFLOAT 64

struct Params {
    u64 key[4];  // candidate c is the secret (key[0] + c, key[1], key[2], key[3])
    u64 off[4];  // recoding offset: sum of 2^(w-1) * 2^(w*i) for i < n-1
    u64 m0, v0, mc, vc;
    u32 index_be;  // seed index, byte swapped
    u32 w, n;
    u32 batch;  // keys per thread
    u32 max_hits;
    u32 dump;  // debugging: write every key's result to `out`
    u32 nfix, nseg, float_lo, float_hi;
    u32 fix_pos[MAXFIX];
    u32 fix_mask[MAXFIX];
    u32 seg_len[MAXSEG];
    u32 fl_mask[MAXFLOAT];
};

// ---------------------------------------------------------------- field

// acc[0..8] += (m0, m1, m2, m3) * bi, where product k lands on limbs
// (2k, 2k+1); the carry out goes to `co`.
#define MADROW(A0, A1, A2, A3, A4, A5, A6, A7, CO, M0, M1, M2, M3, BI)            \
    asm("mad.lo.cc.u32 %0,%9,%13,%0; madc.hi.cc.u32 %1,%9,%13,%1;"                \
        "madc.lo.cc.u32 %2,%10,%13,%2; madc.hi.cc.u32 %3,%10,%13,%3;"             \
        "madc.lo.cc.u32 %4,%11,%13,%4; madc.hi.cc.u32 %5,%11,%13,%5;"             \
        "madc.lo.cc.u32 %6,%12,%13,%6; madc.hi.cc.u32 %7,%12,%13,%7;"             \
        "addc.u32 %8,0,0;"                                                        \
        : "+r"(A0), "+r"(A1), "+r"(A2), "+r"(A3), "+r"(A4), "+r"(A5), "+r"(A6),   \
          "+r"(A7), "=r"(CO)                                                      \
        : "r"(M0), "r"(M1), "r"(M2), "r"(M3), "r"(BI))

#define MULROW(R0, R1, R2, R3, R4, R5, R6, R7, M0, M1, M2, M3, BI)                \
    asm("mul.lo.u32 %0,%8,%12; mul.hi.u32 %1,%8,%12;"                             \
        "mul.lo.u32 %2,%9,%12; mul.hi.u32 %3,%9,%12;"                             \
        "mul.lo.u32 %4,%10,%12; mul.hi.u32 %5,%10,%12;"                           \
        "mul.lo.u32 %6,%11,%12; mul.hi.u32 %7,%11,%12;"                           \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3), "=r"(R4), "=r"(R5), "=r"(R6),   \
          "=r"(R7)                                                                \
        : "r"(M0), "r"(M1), "r"(M2), "r"(M3), "r"(BI))

// Reduce the 16-limb value p + (q << 32) (q given from limb 1) to 8 limbs.
INL void fe_reduce(u32 *r, u32 *p, const u32 *q) {
    u32 c0, c1, c2, c3;
    const u32 k38 = 38;
    asm("add.cc.u32 %0,%0,%8; addc.cc.u32 %1,%1,%9; addc.cc.u32 %2,%2,%10;"
        "addc.cc.u32 %3,%3,%11; addc.cc.u32 %4,%4,%12; addc.cc.u32 %5,%5,%13;"
        "addc.cc.u32 %6,%6,%14; addc.u32 %7,0,0;"
        : "+r"(p[1]), "+r"(p[2]), "+r"(p[3]), "+r"(p[4]), "+r"(p[5]), "+r"(p[6]), "+r"(p[7]),
          "=r"(c0)
        : "r"(q[1]), "r"(q[2]), "r"(q[3]), "r"(q[4]), "r"(q[5]), "r"(q[6]), "r"(q[7]));
    asm("add.cc.u32 %0,%0,%8; addc.cc.u32 %1,%1,%9; addc.cc.u32 %2,%2,%10;"
        "addc.cc.u32 %3,%3,%11; addc.cc.u32 %4,%4,%12; addc.cc.u32 %5,%5,%13;"
        "addc.cc.u32 %6,%6,%14; addc.u32 %7,%7,%15;"
        : "+r"(p[8]), "+r"(p[9]), "+r"(p[10]), "+r"(p[11]), "+r"(p[12]), "+r"(p[13]),
          "+r"(p[14]), "+r"(p[15])
        : "r"(q[8]), "r"(q[9]), "r"(q[10]), "r"(q[11]), "r"(q[12]), "r"(q[13]), "r"(q[14]),
          "r"(q[15]));
    // low += 38 * high
    MADROW(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], c1, p[8], p[10], p[12], p[14], k38);
    asm("mad.lo.cc.u32 %0,%8,%12,%0; madc.hi.cc.u32 %1,%8,%12,%1;"
        "madc.lo.cc.u32 %2,%9,%12,%2; madc.hi.cc.u32 %3,%9,%12,%3;"
        "madc.lo.cc.u32 %4,%10,%12,%4; madc.hi.cc.u32 %5,%10,%12,%5;"
        "madc.lo.cc.u32 %6,%11,%12,%6; madc.hi.u32 %7,%11,%12,0;"
        : "+r"(p[1]), "+r"(p[2]), "+r"(p[3]), "+r"(p[4]), "+r"(p[5]), "+r"(p[6]), "+r"(p[7]),
          "=r"(c2)
        : "r"(p[9]), "r"(p[11]), "r"(p[13]), "r"(p[15]), "r"(k38));
    u32 t = (c0 + c1 + c2) * 38;
    asm("add.cc.u32 %0,%0,%9; addc.cc.u32 %1,%1,0; addc.cc.u32 %2,%2,0;"
        "addc.cc.u32 %3,%3,0; addc.cc.u32 %4,%4,0; addc.cc.u32 %5,%5,0;"
        "addc.cc.u32 %6,%6,0; addc.cc.u32 %7,%7,0; addc.u32 %8,0,0;"
        : "+r"(p[0]), "+r"(p[1]), "+r"(p[2]), "+r"(p[3]), "+r"(p[4]), "+r"(p[5]), "+r"(p[6]),
          "+r"(p[7]), "=r"(c3)
        : "r"(t));
    r[0] = p[0] + ((0u - c3) & 38);
    r[1] = p[1];
    r[2] = p[2];
    r[3] = p[3];
    r[4] = p[4];
    r[5] = p[5];
    r[6] = p[6];
    r[7] = p[7];
}

INL void fe_mul(u32 *r, const u32 *a, const u32 *b) {
    u32 p[16], q[17];
    MULROW(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], a[0], a[2], a[4], a[6], b[0]);
    MULROW(q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], a[1], a[3], a[5], a[7], b[0]);
    p[8] = 0;
    q[9] = 0;
#pragma unroll
    for (int i = 1; i < 8; i++) {
        MADROW(p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4], p[i + 5], p[i + 6], p[i + 7],
               p[i + 8], a[0], a[2], a[4], a[6], b[i]);
        MADROW(q[i + 1], q[i + 2], q[i + 3], q[i + 4], q[i + 5], q[i + 6], q[i + 7], q[i + 8],
               q[i + 9], a[1], a[3], a[5], a[7], b[i]);
    }
    fe_reduce(r, p, q);
}

INL void fe_sq(u32 *r, const u32 *a) { fe_mul(r, a, a); }

INL void fe_add(u32 *r, const u32 *a, const u32 *b) {
    u32 t[8], c, c2;
    asm("add.cc.u32 %0,%9,%17; addc.cc.u32 %1,%10,%18; addc.cc.u32 %2,%11,%19;"
        "addc.cc.u32 %3,%12,%20; addc.cc.u32 %4,%13,%21; addc.cc.u32 %5,%14,%22;"
        "addc.cc.u32 %6,%15,%23; addc.cc.u32 %7,%16,%24; addc.u32 %8,0,0;"
        : "=r"(t[0]), "=r"(t[1]), "=r"(t[2]), "=r"(t[3]), "=r"(t[4]), "=r"(t[5]), "=r"(t[6]),
          "=r"(t[7]), "=r"(c)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
    u32 k = (0u - c) & 38;
    asm("add.cc.u32 %0,%0,%9; addc.cc.u32 %1,%1,0; addc.cc.u32 %2,%2,0;"
        "addc.cc.u32 %3,%3,0; addc.cc.u32 %4,%4,0; addc.cc.u32 %5,%5,0;"
        "addc.cc.u32 %6,%6,0; addc.cc.u32 %7,%7,0; addc.u32 %8,0,0;"
        : "+r"(t[0]), "+r"(t[1]), "+r"(t[2]), "+r"(t[3]), "+r"(t[4]), "+r"(t[5]), "+r"(t[6]),
          "+r"(t[7]), "=r"(c2)
        : "r"(k));
    r[0] = t[0] + ((0u - c2) & 38);
    r[1] = t[1];
    r[2] = t[2];
    r[3] = t[3];
    r[4] = t[4];
    r[5] = t[5];
    r[6] = t[6];
    r[7] = t[7];
}

INL void fe_sub(u32 *r, const u32 *a, const u32 *b) {
    u32 t[8], c, c2;
    asm("sub.cc.u32 %0,%9,%17; subc.cc.u32 %1,%10,%18; subc.cc.u32 %2,%11,%19;"
        "subc.cc.u32 %3,%12,%20; subc.cc.u32 %4,%13,%21; subc.cc.u32 %5,%14,%22;"
        "subc.cc.u32 %6,%15,%23; subc.cc.u32 %7,%16,%24; subc.u32 %8,0,0;"
        : "=r"(t[0]), "=r"(t[1]), "=r"(t[2]), "=r"(t[3]), "=r"(t[4]), "=r"(t[5]), "=r"(t[6]),
          "=r"(t[7]), "=r"(c)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
    u32 k = c & 38;  // c is all ones after a borrow
    asm("sub.cc.u32 %0,%0,%9; subc.cc.u32 %1,%1,0; subc.cc.u32 %2,%2,0;"
        "subc.cc.u32 %3,%3,0; subc.cc.u32 %4,%4,0; subc.cc.u32 %5,%5,0;"
        "subc.cc.u32 %6,%6,0; subc.cc.u32 %7,%7,0; subc.u32 %8,0,0;"
        : "+r"(t[0]), "+r"(t[1]), "+r"(t[2]), "+r"(t[3]), "+r"(t[4]), "+r"(t[5]), "+r"(t[6]),
          "+r"(t[7]), "=r"(c2)
        : "r"(k));
    r[0] = t[0] - (c2 & 38);
    r[1] = t[1];
    r[2] = t[2];
    r[3] = t[3];
    r[4] = t[4];
    r[5] = t[5];
    r[6] = t[6];
    r[7] = t[7];
}

// t += k for a small k; the carry out of limb 7 is dropped.
#define ADDK(T, K)                                                                \
    asm("add.cc.u32 %0,%0,%8; addc.cc.u32 %1,%1,0; addc.cc.u32 %2,%2,0;"          \
        "addc.cc.u32 %3,%3,0; addc.cc.u32 %4,%4,0; addc.cc.u32 %5,%5,0;"          \
        "addc.cc.u32 %6,%6,0; addc.u32 %7,%7,0;"                                  \
        : "+r"(T[0]), "+r"(T[1]), "+r"(T[2]), "+r"(T[3]), "+r"(T[4]), "+r"(T[5]), \
          "+r"(T[6]), "+r"(T[7])                                                  \
        : "r"(K))

// The unique representative below p.
INL void fe_canon(u32 *r, const u32 *a) {
    u32 t[8], u[8];
    for (int i = 0; i < 8; i++) t[i] = a[i];
    u32 k = (t[7] >> 31) * 19;
    t[7] &= 0x7fffffffu;
    ADDK(t, k);  // t <= 2^255 + 18
    for (int i = 0; i < 8; i++) u[i] = t[i];
    const u32 k19 = 19;
    ADDK(u, k19);
    bool ge = (u[7] >> 31) != 0;  // t >= p
    u[7] &= 0x7fffffffu;
    for (int i = 0; i < 8; i++) r[i] = ge ? u[i] : t[i];
}

INL void fe_sqn(u32 *r, const u32 *a, int n) {
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = a[i];
    for (int i = 0; i < n; i++) fe_sq(t, t);
    for (int i = 0; i < 8; i++) r[i] = t[i];
}

// z^(p-2)
INL void fe_invert(u32 *r, const u32 *z) {
    u32 z2[8], z9[8], z11[8], a[8], b[8], c[8], t[8];
    fe_sq(z2, z);
    fe_sqn(t, z2, 2);
    fe_mul(z9, t, z);
    fe_mul(z11, z9, z2);
    fe_sq(t, z11);
    fe_mul(a, t, z9);  // 2^5 - 1
    fe_sqn(t, a, 5);
    fe_mul(b, t, a);  // 2^10 - 1
    fe_sqn(t, b, 10);
    fe_mul(c, t, b);  // 2^20 - 1
    fe_sqn(t, c, 20);
    fe_mul(t, t, c);  // 2^40 - 1
    fe_sqn(t, t, 10);
    fe_mul(a, t, b);  // 2^50 - 1
    fe_sqn(t, a, 50);
    fe_mul(b, t, a);  // 2^100 - 1
    fe_sqn(t, b, 100);
    fe_mul(t, t, b);  // 2^200 - 1
    fe_sqn(t, t, 50);
    fe_mul(t, t, a);  // 2^250 - 1
    fe_sqn(t, t, 5);
    fe_mul(r, t, z11);
}

// -------------------------------------------------------------- blake2b

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define G(a, b, c, d, x, y)                                                       \
    a = a + b + (x);                                                              \
    d = ROTR64(d ^ a, 32);                                                        \
    c = c + d;                                                                    \
    b = ROTR64(b ^ c, 24);                                                        \
    a = a + b + (y);                                                              \
    d = ROTR64(d ^ a, 16);                                                        \
    c = c + d;                                                                    \
    b = ROTR64(b ^ c, 63);
#define ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) \
    G(v0, v4, v8, v12, m[s0], m[s1])                                              \
    G(v1, v5, v9, v13, m[s2], m[s3])                                              \
    G(v2, v6, v10, v14, m[s4], m[s5])                                             \
    G(v3, v7, v11, v15, m[s6], m[s7])                                             \
    G(v0, v5, v10, v15, m[s8], m[s9])                                             \
    G(v1, v6, v11, v12, m[s10], m[s11])                                           \
    G(v2, v7, v8, v13, m[s12], m[s13])                                            \
    G(v3, v4, v9, v14, m[s14], m[s15])

// BLAKE2b of a message whose only non-zero words are the first five;
// returns the first four digest words.
INL void blake2b(u64 *h, u64 m0, u64 m1, u64 m2, u64 m3, u64 m4, u64 len, u64 outlen) {
    const u64 m[16] = {m0, m1, m2, m3, m4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const u64 h0 = 0x6a09e667f3bcc908ull ^ 0x01010000ull ^ outlen;
    u64 v0 = h0;
    u64 v1 = 0xbb67ae8584caa73bull;
    u64 v2 = 0x3c6ef372fe94f82bull;
    u64 v3 = 0xa54ff53a5f1d36f1ull;
    u64 v4 = 0x510e527fade682d1ull;
    u64 v5 = 0x9b05688c2b3e6c1full;
    u64 v6 = 0x1f83d9abfb41bd6bull;
    u64 v7 = 0x5be0cd19137e2179ull;
    u64 v8 = 0x6a09e667f3bcc908ull;
    u64 v9 = 0xbb67ae8584caa73bull;
    u64 v10 = 0x3c6ef372fe94f82bull;
    u64 v11 = 0xa54ff53a5f1d36f1ull;
    u64 v12 = 0x510e527fade682d1ull ^ len;
    u64 v13 = 0x9b05688c2b3e6c1full;
    u64 v14 = ~0x1f83d9abfb41bd6bull;
    u64 v15 = 0x5be0cd19137e2179ull;
    ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
    ROUND(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4)
    ROUND(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8)
    ROUND(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13)
    ROUND(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9)
    ROUND(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11)
    ROUND(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10)
    ROUND(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5)
    ROUND(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0)
    ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
    h[0] = h0 ^ v0 ^ v8;
    h[1] = 0xbb67ae8584caa73bull ^ v1 ^ v9;
    h[2] = 0x3c6ef372fe94f82bull ^ v2 ^ v10;
    h[3] = 0xa54ff53a5f1d36f1ull ^ v3 ^ v11;
}

// ---------------------------------------------------------------- search

INL void load_entry(const uint4 *table, u32 idx, u32 *a, u32 *b, u32 *c) {
    const uint4 *e = table + (u64)idx * 6;
    uint4 x0 = e[0], x1 = e[1], x2 = e[2], x3 = e[3], x4 = e[4], x5 = e[5];
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w;
    a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
    b[0] = x2.x; b[1] = x2.y; b[2] = x2.z; b[3] = x2.w;
    b[4] = x3.x; b[5] = x3.y; b[6] = x3.z; b[7] = x3.w;
    c[0] = x4.x; c[1] = x4.y; c[2] = x4.z; c[3] = x4.w;
    c[4] = x5.x; c[5] = x5.y; c[6] = x5.z; c[7] = x5.w;
}

INL u32 window(const u64 *A, u32 bit) {
    u32 q = bit >> 6, r = bit & 63;
    return (u32)((A[q] >> r) | ((A[q + 1] << 1) << (63 - r)));
}

INL u64 bswap64(u64 x) {
    x = ((x & 0x00ff00ff00ff00ffull) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffull);
    x = ((x & 0x0000ffff0000ffffull) << 16) | ((x >> 16) & 0x0000ffff0000ffffull);
    return (x << 32) | (x >> 32);
}

// Alphabet index of address character p; u = the address bits, left aligned.
INL u32 sym_at(const u64 *u, u32 p) {
    if (p == 0) return (u32)(u[0] >> 63);
    u32 b = 5 * p - 4;
    u32 w = b >> 6, r = b & 63;
    if (r <= 59) return (u32)(u[w] >> (59 - r)) & 31;
    return (u32)((u[w] << (r - 59)) | (u[w + 1] >> (123 - r))) & 31;
}

#define SEL8(R, C, A, B) \
    for (int _i = 0; _i < 8; _i++) R[_i] = (C) ? A[_i] : B[_i];
#define CPY8(R, A) \
    for (int _i = 0; _i < 8; _i++) R[_i] = A[_i];

template <int SEED, int FULL>
INL void body(const Params *pp, const uint4 *table, u32 *out, u64 counter) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 batch = pp->batch, w = pp->w, n = pp->n;
    const u32 half = 1u << (w - 1), cnt = half + 1, maskw = (1u << w) - 1;
    const u64 key0 = pp->key[0] + counter + (u64)gid * batch;
    const u64 key1 = pp->key[1], key2 = pp->key[2], key3 = pp->key[3];
    const u64 off0 = pp->off[0], off1 = pp->off[1], off2 = pp->off[2], off3 = pp->off[3];

    u32 num[BMAX][8], den[BMAX][8], pre[BMAX][8];
#if 1
    u32 xn[FULL ? BMAX : 1][8];
#endif
    u32 acc[8] = {1, 0, 0, 0, 0, 0, 0, 0};

    for (u32 j = 0; j < batch; j++) {
        // Secret -> scalar.
        u64 s[4];
        if (SEED) {
            u64 pr[4];
            blake2b(pr, key0 + j, key1, key2, key3, (u64)pp->index_be, 36, 32);
            blake2b(s, pr[0], pr[1], pr[2], pr[3], 0, 32, 64);
        } else {
            blake2b(s, key0 + j, key1, key2, key3, 0, 32, 64);
        }
        // r = scalar bits 3..=253, plus the recoding offset.
        u64 A[5];
        {
            u64 a0 = (s[0] >> 3) | (s[1] << 61);
            u64 a1 = (s[1] >> 3) | (s[2] << 61);
            u64 a2 = (s[2] >> 3) | (s[3] << 61);
            u64 a3 = (s[3] >> 3) & ((1ull << 59) - 1);
            u64 t, c;
            A[0] = a0 + off0;
            c = A[0] < a0;
            t = a1 + c;
            c = t < c;
            A[1] = t + off1;
            c |= A[1] < t;
            t = a2 + c;
            c = t < c;
            A[2] = t + off2;
            c |= A[2] < t;
            A[3] = a3 + off3 + c;
            A[4] = 0;
        }

        // Start point from the top sub-table (affine).
        u32 S[8], D[8], T[8], Z[8] = {1, 0, 0, 0, 0, 0, 0, 0};
        load_entry(table, (n - 1) * cnt + window(A, w * (n - 1)), S, D, T);

        u32 E[8], F[8], Gq[8], H[8];
        for (u32 i = n - 2;; i--) {
            u32 win = window(A, w * i) & maskw;
            bool neg = win < half;
            u32 mag = neg ? half - win : win - half;
            u32 e0[8], e1[8], e2[8], ypx[8], ymx[8];
            load_entry(table, i * cnt + mag, e0, e1, e2);
            SEL8(ypx, neg, e1, e0)
            SEL8(ymx, neg, e0, e1)
            u32 ta[8], tb[8], tc[8], f1[8], g1[8];
            fe_mul(ta, D, ymx);
            fe_mul(tb, S, ypx);
            fe_mul(tc, T, e2);
            fe_add(H, tb, ta);
            fe_sub(f1, Z, tc);
            if (FULL || i != 0) {
                fe_sub(E, tb, ta);
                fe_add(g1, Z, tc);
                SEL8(F, neg, g1, f1)
                SEL8(Gq, neg, f1, g1)
            } else {
                // Only F is needed; with a negated entry that is Z + C.
                fe_add(g1, Z, tc);
                SEL8(F, neg, g1, f1)
            }
            if (i == 0) break;
            u32 x3[8], y3[8];
            fe_mul(x3, E, F);
            fe_mul(y3, Gq, H);
            fe_add(S, y3, x3);
            fe_sub(D, y3, x3);
            fe_mul(Z, F, Gq);
            fe_mul(T, E, H);
        }

        // x = E/G, y = H/F
        u32 dn[8];
        if (FULL) {
            u32 t[8];
            fe_mul(dn, F, Gq);
            fe_mul(t, Gq, H);
            CPY8(num[j], t)
            fe_mul(t, E, F);
            CPY8(xn[j], t)
        } else {
            CPY8(dn, F)
            CPY8(num[j], H)
        }
        CPY8(den[j], dn)
        fe_mul(acc, acc, dn);
        CPY8(pre[j], acc)
    }

    u32 inv[8];
    fe_invert(inv, acc);

    const u64 m0 = pp->m0, v0 = pp->v0;
    const u32 nfix = pp->nfix, dump = pp->dump;
    for (u32 j = batch; j-- > 0;) {
        u32 di[8], dn[8], t[8], y[8];
        if (j > 0) {
            CPY8(t, pre[j - 1])
            fe_mul(di, inv, t);
        } else {
            CPY8(di, inv)
        }
        CPY8(dn, den[j])
        fe_mul(inv, inv, dn);
        CPY8(t, num[j])
        fe_mul(y, t, di);
        fe_canon(y, y);
        u64 pk0 = (u64)y[0] | ((u64)y[1] << 32);
        bool ok;
        if (FULL) {
            u32 x[8];
            CPY8(t, xn[j])
            fe_mul(x, t, di);
            fe_canon(x, x);
            u64 pk1 = (u64)y[2] | ((u64)y[3] << 32);
            u64 pk2 = (u64)y[4] | ((u64)y[5] << 32);
            u64 pk3 = (u64)y[6] | ((u64)y[7] << 32) | ((u64)(x[0] & 1) << 63);
            ok = (pk0 & m0) == v0;
            if (ok || dump) {
                u64 h[4];
                blake2b(h, pk0, pk1, pk2, pk3, 0, 32, 5);
                u64 cks = h[0] & 0xffffffffffull;
                if (dump) {
                    u64 *o = (u64 *)out + (u64)(gid * batch + j) * 5;
                    o[0] = pk0; o[1] = pk1; o[2] = pk2; o[3] = pk3; o[4] = cks;
                    continue;
                }
                ok = (cks & pp->mc) == pp->vc;
                if (ok) {
                    u64 u[5] = {bswap64(pk0), bswap64(pk1), bswap64(pk2), bswap64(pk3), cks << 24};
                    for (u32 k = 0; k < nfix; k++)
                        ok = ok && ((pp->fix_mask[k] >> sym_at(u, pp->fix_pos[k])) & 1);
                    // Floating segments, in order, at the earliest place each fits.
                    u32 at = pp->float_lo, fo = 0;
                    const u32 hi = pp->float_hi;
                    for (u32 sg = 0; ok && sg < pp->nseg; sg++) {
                        const u32 len = pp->seg_len[sg];
                        bool found = false;
                        while (at + len <= hi) {
                            bool all = true;
                            for (u32 k = 0; all && k < len; k++)
                                all = (pp->fl_mask[fo + k] >> sym_at(u, at + k)) & 1;
                            if (all) { found = true; break; }
                            at++;
                        }
                        ok = found;
                        at += len;
                        fo += len;
                    }
                }
            }
        } else {
            if (dump) {
                ((u64 *)out)[gid * batch + j] = pk0;
                continue;
            }
            ok = (pk0 & m0) == v0;
            if (ok) {
                const u64 be = bswap64(pk0);
                for (u32 k = 0; k < nfix; k++) {
                    const u32 pos = pp->fix_pos[k];
                    const u32 sym = pos == 0 ? (u32)(be >> 63) : (u32)(be >> (63 - 5 * pos)) & 31;
                    ok = ok && ((pp->fix_mask[k] >> sym) & 1);
                }
            }
        }
        if (ok) {
            u32 slot = atomicAdd(out, 1u);
            if (slot < pp->max_hits) out[1 + slot] = gid * batch + j;
        }
    }
}

extern "C" __global__ void search_pq(const Params *p, const uint4 *t, u32 *o, u64 c) { body<0, 0>(p, t, o, c); }
extern "C" __global__ void search_pf(const Params *p, const uint4 *t, u32 *o, u64 c) { body<0, 1>(p, t, o, c); }
extern "C" __global__ void search_sq(const Params *p, const uint4 *t, u32 *o, u64 c) { body<1, 0>(p, t, o, c); }
extern "C" __global__ void search_sf(const Params *p, const uint4 *t, u32 *o, u64 c) { body<1, 1>(p, t, o, c); }
