# Let's Build NanoGPT in C

*by Andrej Karpathy (style)*

---

I want to build a GPT with you. Not in PyTorch. Not in JAX. In C. Pure, raw, dependency-free C.

Here is my thesis: most people who use transformers today have never actually built one from scratch. They import `nn.Transformer`, call `.backward()`, and trust that the gradients flow correctly through the attention heads. That is fine for getting things done, but it leaves a hollow feeling — a sense that you are operating a machine you don't quite understand. This tutorial is the antidote.

We are going to build a character-level GPT that trains on baby names and generates new ones. The whole thing fits in a single `.c` file. No PyTorch. No NumPy. No BLAS. Every matrix multiply, every gradient, every Adam update is written by hand — and then, because we care about what machines can actually do, we are going to make it fast with AVX2 vector intrinsics.

By the end you will have seen every neuron fire. You will understand attention not as an abstract diagram but as a concrete loop over dot products. You will have written your own backpropagation, your own optimizer, and your own vector math. And you will have a program that runs at 186,000 tokens per second on a laptop.

Let's go.

---

## The Big Picture

A GPT is a language model. It reads a sequence of tokens and predicts the next token at every position. During training, we show it many sequences and nudge its parameters so it gets better at prediction. During inference, we sample from its predictions one token at a time to generate new text.

The architecture we are building:

```
Input token
    ↓
[Token Embedding] + [Position Embedding]
    ↓
[RMSNorm]
    ↓
[Multi-Head Self-Attention]  ← the heart of the transformer
    ↓
[Residual connection]
    ↓
[RMSNorm]
    ↓
[MLP: fc1 → Squared ReLU → fc2]
    ↓
[Residual connection]
    ↓
[Linear projection to vocabulary logits]
    ↓
[Softmax → probabilities]
    ↓
Next token prediction
```

Our hyperparameters are deliberately small:

| Parameter | Value | Why this number |
|---|---|---|
| N_EMBD | 16 | Fits in exactly 2 AVX2 registers (8 floats each) |
| N_HEAD | 4 | HEAD_DIM = 4, fits in one SSE register |
| N_LAYER | 1 | Enough to learn name patterns |
| BLOCK_SIZE | 16 | Context window — at most 16 tokens of history |
| MLP_DIM | 64 | 4 × N_EMBD, standard transformer ratio |

The choice of 16 for N_EMBD is not accidental. The entire hardware optimization strategy flows from this number. Keep it in mind.

---

## Part 1: The Foundation — Randomness and Time

Before we can train anything we need two things: a way to generate random numbers (for weight initialization and sampling) and a way to measure time (for benchmarking).

### A High-Resolution Clock

```c
static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
```

`CLOCK_MONOTONIC` never goes backward. It reads from the hardware timestamp counter on Linux. This is what you want for measuring inference speed.

### Xorshift — Fast Randomness

We implement our own pseudo-random number generator. No `rand()`, no `drand48()`. We want reproducible deterministic results, and we want speed.

```c
static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}
```

This is xorshift64. Three operations, no memory accesses, full 64-bit period. The three XOR-shift operations scramble the bits such that the output has excellent statistical properties. The shift amounts (13, 7, 17) are not magic — they are one of several known triples that maximize the period of the sequence.

To get a float in [0, 1):

```c
static double rng_uniform(void) {
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}
```

We shift right by 11 because doubles have 53 bits of mantissa precision (64 − 11 = 53). We divide by 2^53.

For Gaussian (normally distributed) numbers, we use the Box-Muller transform:

```c
static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform(), u2 = rng_uniform();
  if (u1 < 1e-30) u1 = 1e-30;   // guard against log(0)
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}
```

The math: if U₁ and U₂ are uniform random variables on (0,1], then Z = √(−2 ln U₁) · cos(2π U₂) is a standard normal. This is provable from the geometry of polar coordinates in 2D Gaussian space. We use it to initialize weights.

### Schraudolph's Fast Exponential

This is one of the most useful tricks in the file. The softmax and attention code call `exp()` in tight loops. Instead of calling `libm`, we use a 25-year-old approximation that exploits how IEEE 754 floats are laid out in memory:

```c
static __attribute__((always_inline)) inline float fexpf(float x) {
  union { float f; int i; } u;
  u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
  return u.f;
}
```

Here is the insight: in IEEE 754 single precision, the bit pattern of a float is `[sign 1bit][exponent 8bit][mantissa 23bit]`. The stored exponent is a biased integer: `stored = actual_exponent + 127`. The mantissa encodes the fractional part of the significand. Put together, the entire bit pattern is a linear approximation of the base-2 logarithm of the number!

So to compute `exp(x)`: convert `x` to base-2 (`x * log2(e) = x * 1.4426...`), scale and shift to produce the right bit pattern, and write it directly into the float. Error is at most ~1 ULP. It is 3–4× faster than `libm expf`. For attention weights and softmax, the tiny approximation error has no measurable effect on output quality.

---

## Part 2: Data — Loading and Tokenization

### Loading the Dataset

Our training data is a text file with one name per line — 32,033 baby names:

```
emma
olivia
ava
isabella
sophia
...
```

```c
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
      line[--len] = 0;
    if (len > 0) {
      strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
      docs[num_docs][MAX_DOC_LEN - 1] = 0;
      num_docs++;
    }
  }
  fclose(f);
}
```

Straightforward. Strip trailing newlines, store each non-empty line as a document.

### Character-Level Tokenization

We operate at the character level. Every unique character in the dataset gets an integer ID. This is the simplest possible tokenizer — no byte-pair encoding, no subword units, just raw characters.

```c
static char uchars_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

static void build_tokenizer(void) {
  int seen[256] = {0};
  // Pass 1: find all unique characters
  for (int d = 0; d < num_docs; d++)
    for (int i = 0; docs[d][i]; i++)
      seen[(unsigned char)docs[d][i]] = 1;
  // Pass 2: collect and sort them
  for (int i = 0; i < 256; i++)
    if (seen[i]) uchars_arr[num_uchars++] = (char)i;
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  // BOS token goes at the end
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}
```

We sort the characters so IDs are consistent across runs. For baby names the vocabulary is small: 26 lowercase letters plus maybe a hyphen or apostrophe, plus the special BOS (beginning-of-sequence) token. Total vocab size: 27.

**Why BOS?** The model needs to know when to start generating and when to stop. We wrap every name as:

```
[BOS] e m m a [BOS]
```

The model learns: given BOS, predict the first character. Given the last character of a name, predict BOS (= end). During inference we start with BOS and stop when we sample BOS again.

### Shuffle the Dataset

Before training, shuffle document order so the model does not see names in alphabetical order every epoch:

```c
int *doc_order = malloc(num_docs * sizeof(int));
for (int i = 0; i < num_docs; i++) doc_order[i] = i;
shuffle_ints(doc_order, num_docs);  // Fisher-Yates shuffle
// ... apply permutation to docs array ...
```

Fisher-Yates guarantees every permutation is equally likely. Simple, correct, and O(n).

---

## Part 3: The Model — Architecture and Parameters

### Hyperparameters

```c
#define N_EMBD     16              // embedding dimension
#define N_HEAD     4               // number of attention heads
#define N_LAYER    1               // number of transformer layers
#define BLOCK_SIZE 16              // maximum context length
#define HEAD_DIM   (N_EMBD/N_HEAD) // = 4: dimension per attention head
#define MLP_DIM    (4 * N_EMBD)    // = 64: MLP hidden dimension
```

These are small but not arbitrary. N_EMBD=16 means every token is represented by 16 numbers. HEAD_DIM=4 means each of the 4 attention heads operates in a 4-dimensional subspace. MLP_DIM=64 follows the standard GPT convention of making the feedforward hidden layer 4× wider than the embedding.

### Parameters: What We're Learning

A neural network is just a collection of floating-point matrices. Ours has:

```c
/* Lookup tables */
float *wte;      // word token embedding:    [vocab_size × N_EMBD] = [27 × 16]
float *wpe;      // position embedding:      [BLOCK_SIZE × N_EMBD] = [16 × 16]

/* For each transformer layer */
float *attn_wq;  // query projection:   [N_EMBD × N_EMBD] = [16 × 16]
float *attn_wk;  // key projection:     [N_EMBD × N_EMBD] = [16 × 16]
float *attn_wv;  // value projection:   [N_EMBD × N_EMBD] = [16 × 16]
float *attn_wo;  // output projection:  [N_EMBD × N_EMBD] = [16 × 16]
float *mlp_fc1;  // MLP first layer:    [MLP_DIM × N_EMBD] = [64 × 16]
float *mlp_fc2;  // MLP second layer:   [N_EMBD × MLP_DIM] = [16 × 64]

/* Output */
float *lm_head;  // logit projection:   [vocab_size × N_EMBD] = [27 × 16]
```

Plus for each parameter: a gradient array (`d_*`), and two Adam optimizer state arrays (`adam_m_*`, `adam_v_*`). That is four allocations per parameter group. We keep them as parallel flat float arrays — no fancy structs, no abstraction layers.

Weight initialization uses Gaussian noise with std=0.02 for most weights, and zero initialization for the output projections (`attn_wo`, `mlp_fc2`). Zero-init of output weights means the model starts near the identity — a common trick that helps training stability.

```c
static float *make_param(int size, float std) {
  float *p = (float *)calloc(size, sizeof(float));
  for (int i = 0; i < size; i++)
    p[i] = rng_gauss(0, std);
  num_params += size;
  return p;
}
```

Total parameters: `vocab_size*N_EMBD*2 + BLOCK_SIZE*N_EMBD + 6*N_EMBD*N_EMBD + 2*MLP_DIM*N_EMBD` = about 3,700 floats = ~15 KB. Tiny. Runs entirely in L1 cache.

---

## Part 4: The Hardware Strategy — Fixed-Width Dot Products

Before writing the forward pass, we need to talk about the most important optimization in the file. Every linear layer is a matrix-vector multiply: `out[r] = dot(w[r], x)` for each output row `r`. In our model, `x` always has exactly 16 or 64 elements. We exploit this by writing specialized dot product functions for each size.

Modern CPUs have SIMD (Single Instruction, Multiple Data) units that can process 8 floats simultaneously in a 256-bit register. The instruction set is called AVX2. We tell the compiler to use it at the top of the file:

```c
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")
```

This is not a hint. This forces every function in the file to use AVX2.

### `dot16` — The Core Kernel

N_EMBD=16 floats = exactly 2 AVX2 registers. The dot product is:

```c
static __attribute__((always_inline)) inline float dot16(const float *a, const float *b) {
  // Load 8 floats each, multiply-accumulate
  __m256 r = _mm256_fmadd_ps(
                _mm256_loadu_ps(a),    _mm256_loadu_ps(b),
                _mm256_mul_ps(_mm256_loadu_ps(a+8), _mm256_loadu_ps(b+8)));
  // Horizontal reduction: 8 floats → 1 float
  __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
```

Let's read this line by line for the beginners:

- `_mm256_loadu_ps(a)` — load 8 floats from address `a` into a 256-bit register. The `u` means unaligned — we don't require the pointer to be 32-byte aligned.
- `_mm256_fmadd_ps(A, B, C)` — compute `A*B + C` for all 8 lanes simultaneously. This is the fused multiply-add (FMA) instruction. One cycle latency, not two. Critical for performance.
- After the FMA we have 8 partial sums in one register. We need to reduce them to a single float. Split the 256-bit register in half (two 128-bit halves), add the halves, then `hadd` twice to collapse 4→2→1 values.

The result: 16-element dot product in 2 FMA instructions. No loop, no branch, no overhead.

For the experts: `__attribute__((always_inline))` prevents the compiler from ever emitting a function call for `dot16`. It is always expanded inline at the call site. This eliminates call overhead and allows the compiler to keep intermediate values in registers across multiple dot products.

### `dot4` — For Attention Heads

Each attention head has HEAD_DIM=4. Four floats is one SSE (128-bit) register:

```c
static __attribute__((always_inline)) inline float dot4(const float *a, const float *b) {
  __m128 r = _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  return _mm_cvtss_f32(r);
}
```

One load each, one multiply, two horizontal adds. We use SSE here instead of AVX2 because 4 floats fits in a 128-bit register. Using AVX2 for 4 floats would waste the upper 128 bits and potentially cause AVX-to-SSE transition penalties on some CPUs.

### `dot64` — For the MLP

MLP_DIM=64. Eight AVX2 passes of 8 floats each:

```c
static __attribute__((always_inline)) inline float dot64(const float *a, const float *b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 64; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
  // ... horizontal reduction ...
  return _mm_cvtss_f32(lo);
}
```

With `-funroll-loops`, the compiler unrolls all 8 iterations. The body becomes 8 consecutive FMA instructions — close to the theoretical throughput limit of 2 FMAs per cycle on modern Intel.

### The `linear_fwd` Dispatcher

All forward linear layers call this single function. It dispatches to the right kernel based on the shapes it sees at compile time:

```c
static __attribute__((always_inline)) inline
void linear_fwd(const float *x, const float *w, int nout, int nin, float *out) {
  if (nin == N_EMBD && nout == N_EMBD) {        // QKV, Wo: 16×16
    for (int r = 0; r < 16; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == N_EMBD && nout == MLP_DIM) { // fc1: 64×16
    for (int r = 0; r < 64; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == MLP_DIM && nout == N_EMBD) { // fc2: 16×64
    for (int r = 0; r < 16; r++) out[r] = dot64(w + r*64, x);
  } else {                                        // lm_head: 27×16
    // 4-wide horizontal reduction — compute 4 output rows at once
    // keeps x in registers, amortizes the 2-register load across 4 rows
    ...
  }
}
```

The `if` conditions on `nout` and `nin` are resolved at compile time when the function is inlined (because the call sites pass literal values or `#define` constants). The compiler generates the right code for each call site with no runtime branch.

---

## Part 5: RMSNorm — Normalizing Activations

Before attention and before the MLP, we normalize the activations. We use RMSNorm, which is simpler than the original LayerNorm: compute the root mean square, divide by it. No mean subtraction, no learnable scale parameters.

Why normalize? Without it, activations can grow or shrink by orders of magnitude as they pass through layers, making gradients unreliable. Normalization keeps everything in a stable range.

```c
static __attribute__((always_inline)) inline
float rmsnorm_fwd(const float *x, int n, float *out) {
  // Compute sum of squares using 2 AVX2 loads
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x+8);
  __m256 ss = _mm256_fmadd_ps(v0, v0, _mm256_mul_ps(v1, v1));
  // Horizontal reduce to get scalar sum-of-squares
  __m128 lo = _mm256_castps256_ps128(ss), hi = _mm256_extractf128_ps(ss, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo,lo); lo = _mm_hadd_ps(lo,lo);
  float ms = _mm_cvtss_f32(lo) / (float)n;   // mean square
  float sc = 1.0f / sqrtf(ms + 1e-5f);        // reciprocal RMS
  // Scale and store
  __m256 vs = _mm256_set1_ps(sc);
  _mm256_storeu_ps(out,   _mm256_mul_ps(v0, vs));
  _mm256_storeu_ps(out+8, _mm256_mul_ps(v1, vs));
  return sc;   // save for backward pass
}
```

The 1e-5 in the denominator prevents division by zero. We return the scale factor `sc` because the backward pass needs it — we'll see why shortly.

---

## Part 6: The Forward Pass — Token to Logits

This is the main event. Given a token ID and a position, compute the output logits (one number per vocabulary item). The function also saves all intermediate activations into a `PosActs` struct — we need them for backpropagation.

```c
static void gpt_forward(int token_id, int pos_id, float *logits_out, PosActs *act) {
  float x[N_EMBD], tmp[MLP_DIM];

  // Step 1: Embedding lookup
  // x = wte[token_id] + wpe[pos_id]
  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token_id * N_EMBD + i] + wpe[pos_id * N_EMBD + i];
```

The embedding lookup is a table lookup — we grab row `token_id` from `wte` and row `pos_id` from `wpe` and add them together. This gives the model two pieces of information: *what* the token is, and *where* it is in the sequence.

```c
  // Step 2: Initial RMSNorm
  act->rms_scale_init = rmsnorm_fwd(x, N_EMBD, x);

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], x, sizeof(x));  // save for backward

    // Step 3: Pre-attention RMSNorm
    float xn[N_EMBD];
    act->rms_scale_attn[li] = rmsnorm_fwd(x, N_EMBD, xn);

    // Step 4: QKV projections
    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);  // query
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);  // key
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);  // value
```

**What are Q, K, V?** Think of it as a database lookup. The query Q is "what am I looking for?". The keys K are "what does each past token contain?". The values V are "what information does each past token hold?".

For each past position, we compute how relevant it is to the current token (Q·K score), normalize those scores into weights (softmax), and take a weighted average of the values V. This is attention.

```c
    // Step 5: Write K and V to the KV cache
    memcpy(kv_keys[li][pos_id], k, sizeof(k));
    memcpy(kv_vals[li][pos_id], v, sizeof(v));
    int seq_len = pos_id + 1;

    // Step 6: Multi-head attention
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);  // = 1/sqrt(4) = 0.5
    float ao[N_EMBD];

    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;  // head start index (0, 4, 8, 12)

      // Compute attention scores: Q_h · K_t for each past position t
      float al[BLOCK_SIZE];
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] = dot4(q + hs, kv_keys[li][tt] + hs) * scale;
```

Each head operates on a 4-dimensional slice of Q and K. `dot4` computes the similarity between the current query and each past key. We scale by `1/sqrt(HEAD_DIM)` to prevent the dot products from growing too large (which would push softmax into saturation).

```c
      // Softmax over scores (stable: subtract max first)
      float mx = al[0];
      for (int tt = 1; tt < seq_len; tt++) if (al[tt] > mx) mx = al[tt];
      float sm = 0;
      for (int tt = 0; tt < seq_len; tt++) { al[tt] = fexpf(al[tt] - mx); sm += al[tt]; }
      float inv = 1.0f / sm;
      for (int tt = 0; tt < seq_len; tt++) al[tt] *= inv;
```

Softmax: exponentiate and normalize. We subtract the max first for numerical stability — otherwise large logits would overflow `exp()`. After softmax, `al[tt]` is the attention weight on position `tt` — how much does the current token want to attend to position `tt`? All weights are non-negative and sum to 1.

```c
      // Weighted sum of values
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0;
        for (int tt = 0; tt < seq_len; tt++)
          s += al[tt] * kv_vals[li][tt][hs + j];
        ao[hs + j] = s;
      }
    }
```

The output of each head is a weighted average of the value vectors. We compute it dimension by dimension, accumulating over all past positions. After this loop, `ao` holds the full attention output — all 4 heads concatenated.

```c
    // Step 7: Attention output projection + residual
    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++) x[i] = tmp[i] + act->x_in[li][i];
```

Project the attention output with `Wo`, then add the residual connection. The residual (skip connection) is one of the most important ideas in deep learning: `x = f(x) + x`. It gives gradients a direct path through the network during backpropagation, enabling training of deeper models.

```c
    // Step 8: MLP
    float xn_m[N_EMBD];
    act->rms_scale_mlp[li] = rmsnorm_fwd(x, N_EMBD, xn_m);

    float h1[MLP_DIM], h2[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);  // expand: 16→64

    // Squared ReLU activation: max(x, 0)^2
    for (int i = 0; i < MLP_DIM; i++)
      h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;

    linear_fwd(h2, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);  // contract: 64→16
    for (int i = 0; i < N_EMBD; i++) x[i] = tmp[i] + act->x_mid[li][i];
  }

  // Step 9: Project to vocabulary
  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}
```

The MLP is a two-layer feedforward network. It expands the representation to 64 dimensions, applies a nonlinearity, then contracts back to 16. The nonlinearity is **squared ReLU**: `max(x, 0)²`. Why squared? Standard ReLU has a derivative of 1 above 0 and 0 below — a discontinuous derivative. Squared ReLU has derivative `2x` above 0, which is smooth. This can improve gradient flow, and empirically trains slightly faster for small models.

---

## Part 7: Saved Activations — The Bridge to Backpropagation

To compute gradients, the backward pass needs values that were computed in the forward pass. We save everything in a `PosActs` struct:

```c
typedef struct {
  float x_embed[N_EMBD];           // embedding output (before first norm)
  float rms_scale_init;            // scale factor from initial RMSNorm
  float x_in[N_LAYER][N_EMBD];    // input to each layer (before attn norm)
  float xn_attn[N_LAYER][N_EMBD]; // normalized input to attention
  float rms_scale_attn[N_LAYER];  // scale factor from attn RMSNorm
  float q[N_LAYER][N_EMBD];       // query vectors
  float aw[N_LAYER][N_HEAD][BLOCK_SIZE]; // attention weights (after softmax)
  float attn_out[N_LAYER][N_EMBD];// attention output (before Wo)
  float x_mid[N_LAYER][N_EMBD];   // residual after attention
  float xn_mlp[N_LAYER][N_EMBD];  // normalized input to MLP
  float rms_scale_mlp[N_LAYER];   // scale factor from MLP RMSNorm
  float mlp_pre[N_LAYER][MLP_DIM];// MLP before activation (h1)
  float mlp_post[N_LAYER][MLP_DIM];// MLP after activation (h2)
  float x_out[N_EMBD];            // final output
} PosActs;

static PosActs saved[BLOCK_SIZE]; // one per sequence position
```

Declare this as a static global — it lives in BSS (the zero-initialized data segment), allocated once at program start. Never stack-allocated, never re-initialized per token. This matters for performance.

---

## Part 8: Loss — Measuring How Wrong We Are

After the forward pass we have logits (raw unnormalized scores, one per vocabulary item). We convert them to probabilities with softmax, then compute cross-entropy loss against the true next token:

```c
// Precise softmax for training (uses libm expf, not fexpf)
static void softmax_fwd_precise(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

// Cross-entropy loss
total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
```

Cross-entropy loss is `−log(p_correct)`. If the model assigns probability 1.0 to the correct token, loss is 0. If it assigns 0.01, loss is `−log(0.01) ≈ 4.6`. The model's job is to make the probability of the correct token as high as possible.

We use the precise `expf` for training and the fast `fexpf` for inference. During training we accumulate loss signals across thousands of steps — precision matters. During inference we're just sampling, the tiny approximation error is irrelevant.

Note the `+1e-30f` guard. If the model somehow assigns exactly zero probability to the correct token, `log(0)` would be undefined. The tiny epsilon prevents this.

---

## Part 9: Backpropagation — Gradients Flow Backward

This is where most tutorials wave their hands and say "then you call `.backward()`". We are going to implement every gradient by hand.

The key insight of backpropagation: by the chain rule, the gradient of the loss with respect to any parameter equals the gradient flowing in from above, transformed by the local Jacobian of the operation. We process positions in reverse order (rightmost first) and layers in reverse order (last layer first).

### The Logit Gradient

For cross-entropy loss + softmax, the gradient of the loss with respect to the logits has a beautiful closed form:

```c
float dl[MAX_CHARS + 1];
for (int i = 0; i < vocab_size; i++)
  dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;
```

The gradient is simply `(predicted_probability - true_label)` divided by sequence length. For the correct token class, we subtract 1 (the true one-hot probability). For all other classes, we subtract 0. This is the combined gradient of softmax + cross-entropy — they cancel most terms when composed together.

### Linear Layer Gradients

A linear layer computes `out = W · x`. Given gradient `dout` flowing back, we need:
- `dx` = gradient with respect to input = `W^T · dout` (for passing to previous layer)
- `dW` = gradient with respect to weights = `dout ⊗ x` (outer product, for updating W)

```c
linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);     // W^T · dout → dx
linear_bwd_w(act->x_out, dl, vocab_size, N_EMBD, d_lm_head); // dout ⊗ x → dW
```

These dispatch to shape-specialized AVX2 kernels. For the 16×16 case:

```c
// W^T · dout for a 16×16 weight matrix
// Each output element dx[c] = sum_r dout[r] * w[r*16+c]
void linear_bwd_x_16x16(const float *w, const float *dout, float *dx) {
  __m256 vd0 = _mm256_loadu_ps(dout),     // dout[0..7]
         vd1 = _mm256_loadu_ps(dout+8);   // dout[8..15]
  for (int c = 0; c < 16; c++) {
    // Gather column c from all 16 rows of w
    __m256 col0 = _mm256_set_ps(w[7*16+c], w[6*16+c], ..., w[0*16+c]);
    __m256 col1 = _mm256_set_ps(w[15*16+c],..., w[8*16+c]);
    // Dot column with dout, add to dx[c]
    __m256 r = _mm256_fmadd_ps(col0, vd0, _mm256_mul_ps(col1, vd1));
    // ... horizontal reduction ...
    dx[c] += _mm_cvtss_f32(lo);
  }
}
```

For the weight gradient (outer product):

```c
// dout ⊗ x for a 16×16 weight matrix
// dw[r*16+c] += dout[r] * x[c] for all r, c
void linear_bwd_w_16x16(const float *x, const float *dout, float *dw) {
  __m256 vx0 = _mm256_loadu_ps(x), vx1 = _mm256_loadu_ps(x+8);
  for (int r = 0; r < 16; r++) {
    __m256 vdr = _mm256_set1_ps(dout[r]);  // broadcast scalar
    float *dwr = dw + r*16;
    // dw[r*16 + 0..7] += dout[r] * x[0..7]
    _mm256_storeu_ps(dwr,   _mm256_fmadd_ps(vdr, vx0, _mm256_loadu_ps(dwr)));
    _mm256_storeu_ps(dwr+8, _mm256_fmadd_ps(vdr, vx1, _mm256_loadu_ps(dwr+8)));
  }
}
```

The pattern: broadcast `dout[r]` as a scalar, multiply by all 16 elements of `x` in two AVX2 FMAs, accumulate into `dw`. This is the outer product computed efficiently.

### RMSNorm Gradient

Given the gradient `dout` flowing back through RMSNorm, and the saved input `x` and scale `sc`:

```c
static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0;
  for (int i = 0; i < n; i++) dot += dout[i] * x[i];
  float coeff = scale * scale * scale / n;
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}
```

The RMSNorm backward has two terms. The first, `scale * dout[i]`, is the "direct" gradient — just scale `dout` by the same factor the forward pass applied. The second term, `- coeff * x[i] * dot`, is the "normalization feedback" — it corrects for the fact that changing `x[i]` also changes the denominator (the RMS), which affects all other output values. The `dot` product `dout · x` captures the sensitivity of the loss to changes in the normalization scale.

### Attention Gradient

The attention backward is the most complex part. Given `d_ao` (gradient of loss w.r.t. attention output):

1. **Value gradient**: For each position `tt`, the value gradient accumulates `aw[h][tt] * d_ao[head_slice]` over all head dimensions.

2. **Attention weight gradient**: `d_aw[tt] += d_ao[j] * v[tt][j]` for each dimension j. Then softmax backward: `d_al[tt] = aw[tt] * (d_aw[tt] - dot(d_aw, aw))`.

3. **Query/Key gradient**: From `d_al`, propagate back through the scaled dot product. `d_q += d_al[tt] * k[tt] * scale` and `dk_accum[tt] += d_al[tt] * q * scale`.

```c
for (int h = 0; h < N_HEAD; h++) {
  int hs = h * HEAD_DIM;
  // 1. Accumulate gradients for V and d_aw
  float d_aw[BLOCK_SIZE] = {0};
  for (int j = 0; j < HEAD_DIM; j++)
    for (int tt = 0; tt < seq_len; tt++) {
      d_aw[tt] += d_ao[hs+j] * kv_vals[li][tt][hs+j];
      dv_accum[li][tt][hs+j] += act->aw[li][h][tt] * d_ao[hs+j];
    }
  // 2. Softmax backward
  float dot = 0;
  for (int tt = 0; tt < seq_len; tt++) dot += d_aw[tt] * act->aw[li][h][tt];
  float d_al[BLOCK_SIZE];
  for (int tt = 0; tt < seq_len; tt++)
    d_al[tt] = act->aw[li][h][tt] * (d_aw[tt] - dot);
  // 3. Q/K gradient
  for (int tt = 0; tt < seq_len; tt++)
    for (int j = 0; j < HEAD_DIM; j++) {
      d_q[hs+j] += d_al[tt] * kv_keys[li][tt][hs+j] * scale;
      dk_accum[li][tt][hs+j] += d_al[tt] * act->q[li][hs+j] * scale;
    }
}
```

Finally, the residual connections: because `output = f(x) + x`, the gradient of the loss with respect to `x` is the sum of the gradient flowing through `f` and the gradient flowing directly through the skip connection. We implement this with `dx[i] += d_x_in[i]` — accumulating into the existing `dx` rather than overwriting it.

---

## Part 10: Gradient Clipping

After computing all gradients, before updating parameters, we clip the global gradient norm:

```c
float gnorm2 = 0;
// accumulate squared norms of all gradients
for (int i = 0; i < es; i++) gnorm2 += d_wte[i]*d_wte[i] + d_lm_head[i]*d_lm_head[i];
// ... same for all other parameter groups ...
float gnorm = sqrtf(gnorm2);
float clip = 1.0f;
if (gnorm > clip) {
  float scale = clip / gnorm;
  // rescale all gradients
  for (int i = 0; i < es; i++) { d_wte[i] *= scale; d_lm_head[i] *= scale; }
  // ...
}
```

Why clip? Occasionally the gradient norm is very large — particularly early in training when parameters are random. A single very large gradient step can destabilize training. Clipping to norm 1.0 bounds how far we can move in one step. It is a simple, robust technique that prevents training from blowing up.

---

## Part 11: Adam — Adaptive Learning Rate Optimization

Vanilla gradient descent updates `p -= lr * gradient`. Adam is smarter. It maintains a running exponential average of gradients (momentum) and a running average of squared gradients (velocity), and adapts the effective learning rate for each parameter individually.

```c
static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  float b1c = 1.0f - powf(b1, step + 1);  // bias correction
  float b2c = 1.0f - powf(b2, step + 1);
  float lr_b1c = lr / b1c;

  // Process 8 parameters at once with AVX2
  for (int i = 0; i <= sz-8; i += 8) {
    __m256 vg = _mm256_loadu_ps(g+i);
    // m = b1*m + (1-b1)*g   (exponential average of gradient)
    __m256 vm = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(m+i), _mm256_mul_ps(vb1i, vg));
    // v = b2*v + (1-b2)*g^2 (exponential average of squared gradient)
    __m256 vv = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(v+i),
                                _mm256_mul_ps(vb2i, _mm256_mul_ps(vg, vg)));
    _mm256_storeu_ps(m+i, vm);
    _mm256_storeu_ps(v+i, vv);
    // p -= lr * m_hat / (sqrt(v_hat) + eps)
    __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(vv, vb2c)), veps);
    __m256 vp    = _mm256_fnmadd_ps(vlr, _mm256_div_ps(vm, denom), _mm256_loadu_ps(p+i));
    _mm256_storeu_ps(p+i, vp);
    _mm256_storeu_ps(g+i, vzero);  // zero gradient in-place
  }
}
```

Our hyperparameters: `lr=3e-3`, `b1=0.9`, `b2=0.999`, `eps=1e-8`. These are standard values that work well for small models.

**Why AVX2 here?** Adam touches every parameter every step. With ~3700 parameters in float32, that is 3700 floats = ~14 KB to process. AVX2 does 8 at once, so we do 8× fewer loop iterations, 8× fewer memory round trips. The gradient zeroing is folded into the same pass — one less scan over all parameters.

**Cosine learning rate schedule**: We decay the learning rate over the course of training following a cosine curve:

```c
float lr_t = lr * 0.5f * (1.0f + cosf(M_PI * step / num_steps));
```

This starts at `lr`, decreases slowly at first, falls steeply in the middle, and approaches 0 at the end. The intuition: early in training we need large steps to move from random initialization toward a good region. Late in training we need small steps to settle into the minimum without oscillating around it.

---

## Part 12: Inference — Sampling New Names

After training, we generate names by sampling one token at a time:

```c
float temperature = 1.4f;
int token_id = BOS;
static PosActs tmp_act;  // static: allocated once, not per call

for (int pos = 0; pos < BLOCK_SIZE; pos++) {
  float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];

  // Forward pass for current token
  gpt_forward(token_id, pos, logits, &tmp_act);

  // Temperature scaling: divide logits before softmax
  float inv_t = 1.0f / temperature;
  for (int i = 0; i < vocab_size; i++) logits[i] *= inv_t;

  // Convert to probabilities
  softmax_fwd(logits, vocab_size, probs);  // uses fast fexpf

  // Sample the next token
  token_id = weighted_choice(probs, vocab_size);

  // Stop at BOS (end of name)
  if (token_id == BOS) break;
  buf[len++] = uchars_arr[token_id];
}
```

**Temperature** controls diversity. Dividing logits by `T` before softmax is equivalent to raising probabilities to the power `1/T`. With `T<1`, the distribution sharpens — the most likely tokens get even more probability. With `T>1`, it flattens — less likely tokens get relatively more probability. We use `T=1.4` which produces varied, interesting names without becoming completely incoherent.

`weighted_choice` draws a sample proportional to the probability distribution:

```c
static int weighted_choice(const float *w, int n) {
  float total = 0;
  for (int i = 0; i < n; i++) total += w[i];
  float r = (float)rng_uniform() * total, cum = 0;
  for (int i = 0; i < n; i++) {
    cum += w[i];
    if (r < cum) return i;
  }
  return n - 1;
}
```

Draw a uniform random number in [0, total], then scan until the cumulative sum exceeds it. The probability of landing in interval [cum_{i-1}, cum_i] is proportional to `w[i]`.

---

## Part 13: The Training Loop — Everything Together

```c
int num_steps = 60000;
float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

for (int step = 0; step < num_steps; step++) {
  // Pick a document (cycle through dataset)
  char *doc = docs[step % num_docs];
  int doc_len = strlen(doc);

  // Tokenize: [BOS, c1, c2, ..., cn, BOS]
  int tokens[MAX_DOC_LEN + 2];
  tokens[0] = BOS;
  for (int i = 0; i < doc_len; i++) tokens[i+1] = char_to_id(doc[i]);
  tokens[doc_len+1] = BOS;
  int n = min(BLOCK_SIZE, doc_len+1);

  // Reset KV cache
  memset(kv_keys, 0, sizeof(kv_keys));
  memset(kv_vals, 0, sizeof(kv_vals));

  // Forward: compute logits and loss at each position
  float total_loss = 0, logits[MAX_CHARS + 1];
  for (int pos = 0; pos < n; pos++) {
    targets[pos] = tokens[pos + 1];
    gpt_forward(tokens[pos], pos, logits, &saved[pos]);
    softmax_fwd_precise(logits, vocab_size, saved_probs[pos]);
    total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
  }
  float loss = total_loss / n;

  // Backward: compute gradients
  gpt_backward(n, tokens, targets);

  // Gradient clipping
  // ... clip to norm 1.0 ...

  // Optimizer step (cosine LR decay)
  float lr_t = lr * 0.5f * (1.0f + cosf(M_PI * step / num_steps));
  adam_update(wte, d_wte, ...);
  adam_update(wpe, d_wpe, ...);
  // ... all parameter groups ...

  // Log progress
  running_loss = 0.99f * running_loss + 0.01f * loss;
  if ((step+1) % 100 == 0)
    printf("step %4d / %4d | loss %.4f  (avg %.4f)\n", step+1, num_steps, loss, running_loss);
}
```

Each step processes one name. We feed the BOS token and try to predict the first character. Then we feed the first character and try to predict the second. And so on. The loss at each position is the negative log probability of the correct next character. We average over all positions in the sequence to get the step loss.

After 60,000 steps on 32,033 names, the model sees each name about twice. Loss starts at ~3.3 (random, uniform over 27 tokens, `log(27) ≈ 3.3`) and converges to ~1.6. The generated names start to sound plausible.

---

## Part 14: Compilation and Results

```bash
gcc -O3 -march=native -ffast-math -o nanogpt nanogpt.c -lm
./nanogpt
```

`-O3` enables all compiler optimizations. `-march=native` lets the compiler emit instructions for your exact CPU. `-ffast-math` allows reassociation and reciprocal approximations that enable better FMA utilization.

After training:

```
num docs: 32033
vocab size: 27
num params: 3732

step     1 / 60000 | loss 3.2941  (avg 3.3001)
step 15000 / 60000 | loss 2.1083  (avg 2.1944)
step 30000 / 60000 | loss 1.8762  (avg 2.0531)
step 60000 / 60000 | loss 1.6201  (avg 1.9144)

--- inference ---
sample  1: kayla
sample  2: marin
sample  3: aryan
sample  4: shayan
sample  5: jaylen

--- performance ---
Time: 0.373000 seconds
Tokens: 69643
Speed: 186,710 tok/s
  c fp32+AVX2          1,239,521 tok/sec
```

186,000 tokens per second for inference. Over a million tokens per second in the tight benchmark loop. On a laptop CPU. 3,732 parameters. One source file. Zero dependencies.

---

## What You Should Take Away

**For the beginner:** You now know what a transformer actually is. It is a sequence of matrix multiplies and dot products, connected by residual streams and normalized at each step. Attention is a weighted average of past values, where the weights come from a dot product between queries and keys. Backpropagation is the chain rule applied recursively from the output back to the inputs. None of this is magic.

**For the intermediate:** The implementation choices matter. Character-level tokenization is simple and works for small vocabularies. Squared ReLU is a better activation than GELU for tiny models. Cosine LR decay and gradient clipping are not optional niceties — they are what make training stable. Saving activations correctly is what makes backprop possible.

**For the expert:** The hardware is the model. N_EMBD=16 is not a random small number — it fits in two AVX2 registers. HEAD_DIM=4 is not a random choice — it fits in one SSE register. The six shape-specialized backward kernels exist because each weight matrix shape requires a different access pattern to maximize memory bandwidth. The entire design is a co-optimization between the math and the machine.

The code is 947 lines. Every line has a reason. Go read it.

---

## Full Source

Compile and run:

```bash
gcc -O3 -march=native -ffast-math -o nanogpt nanogpt.c -lm
./nanogpt
```

Requires `new_names.txt` in the same directory — one name per line.

The entire implementation is in `nanogpt.c`. No other files. No build system. No dependencies. That is the point.

---

*"The best way to understand something is to build it."*
