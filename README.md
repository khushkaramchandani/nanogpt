# NanoGPT-C

A GPT trained and inferred in pure C — no dependencies, no libraries, no frameworks.  
Every hot path is AVX2-vectorized with fixed-width kernels tuned to the exact weight shapes in the model.

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Embedding dim | 16 |
| Attention heads | 4 |
| Head dim | 4 |
| Layers | 1 |
| Context length | 16 tokens |
| MLP hidden dim | 64 (4 × N_EMBD) |
| Normalization | RMSNorm |
| Activation | Squared ReLU |
| Optimizer | Adam with cosine LR decay |
| Training steps | 60,000 |

---

## Optimizations

**Compute kernels**
- `dot16` — 2 AVX2 FMA instructions for N_EMBD=16 dot products (fits in exactly 2 registers)
- `dot4` — 1 SSE instruction for HEAD_DIM=4 dot products (fits in exactly 1 register)
- `dot64` — 8 unrolled AVX2 FMA passes for MLP_DIM=64 dot products
- `linear_fwd` — dispatches to the right kernel based on statically-known weight shape

**Forward pass**
- RMSNorm fully vectorized over N_EMBD=16 in two AVX2 loads
- Schraudolph fast exponential (`fexpf`) — 3–4× faster than `libm expf`, ~1 ULP error
- `lm_head` projection computes 4 output rows simultaneously, keeping input in registers

**Backward pass**
- Six shape-specialized backward kernels (`16×16`, `16×64`, `64×16`) for both input-grad and weight-grad
- All dispatch transparently through `linear_bwd_x` and `linear_bwd_w` wrappers — no call site changes
- Outer product accumulation uses `vbroadcastss` + FMA pattern

**Optimizer**
- `adam_update` processes 8 floats per iteration using AVX2 FMA
- Gradient zeroing fused into the same Adam loop pass — no separate memset

**Memory**
- Activation buffers declared `static` — allocated once in BSS, never re-initialized per token
- `#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")` forces AVX2 at translation-unit level

---

## Requirements

- GCC with AVX2 support
- A CPU with AVX2 (Intel Haswell / AMD Ryzen and newer)
- `new_names.txt` — training data, one name per line, in the same directory as the binary

---

## Compilation

```bash
gcc -O3 -march=native -ffast-math -o nanogpt nanogpt.c -lm
```

---

## Running

```bash
./nanogpt
```

Make sure `new_names.txt` is in the same directory. You should see loss decreasing, then generated samples:

```
num docs: 32033
vocab size: 27

step      1 / 60000 | loss 3.2941  (avg 3.3001)
step  15000 / 60000 | loss 2.1083  (avg 2.1944)
step  30000 / 60000 | loss 1.7534  (avg 1.9872)
step  60000 / 60000 | loss 1.6201  (avg 1.9144)



--- performance ---
Time: 0.373000 seconds
Tokens: 69643
Speed: 186710 tok/s
  c fp32+AVX2               1,239,521 tok/sec
```

---

## Performance (Intel i3-1315U)

| Metric | Value |
|---|---|
| Inference speed | 186,710 tok/s |
| Benchmark loop | 1,239,521 tok/s |

For higher throughput on Intel hybrid CPUs, pin to P-cores:

```bash
taskset -c 0,1 ./nanogpt
```

---

## Files

```
nanogpt.c       — full implementation, single file
new_names.txt   — training dataset (indian names, one per line)
README.md       — this file
```
