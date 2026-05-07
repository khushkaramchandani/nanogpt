/*
 * microGPT-C : inference-optimized build
 * Compile: gcc -O3 -march=native -ffast-math -o microgpt microgpt.c -lm
 */

#ifndef _MSC_VER
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")
#endif

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static double now_s(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
  }
#else
  #include <time.h>
  static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
#endif
#include <immintrin.h>

/* Schraudolph fast exp: ~1 ULP error, 3-4x faster than libm expf */
static __attribute__((always_inline)) inline float fexpf(float x) {
  union { float f; int i; } u;
  u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
  return u.f;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Minimal xorshift PRNG (seeded deterministically like Python's 42) */
static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static double rng_uniform(void) {
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform(), u2 = rng_uniform();
  if (u1 < 1e-30)
    u1 = 1e-30;
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void shuffle_ints(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_uniform() * (i + 1));
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/* Dataset loading */
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(1);
  }
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (len > 0) {
      strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
      docs[num_docs][MAX_DOC_LEN - 1] = 0;
      num_docs++;
    }
  }
  fclose(f);
}

/* Tokenizer */
static char uchars_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

static int char_to_id(char c) {
  for (int i = 0; i < num_uchars; i++)
    if (uchars_arr[i] == c)
      return i;
  return -1;
}

static int cmp_char(const void *a, const void *b) {
  return *(const char *)a - *(const char *)b;
}

static void build_tokenizer(void) {
  int seen[256] = {0};
  for (int d = 0; d < num_docs; d++)
    for (int i = 0; docs[d][i]; i++)
      seen[(unsigned char)docs[d][i]] = 1;
  for (int i = 0; i < 256; i++)
    if (seen[i])
      uchars_arr[num_uchars++] = (char)i;
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}

/* Model hyper-parameters */
#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 16
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_DIM (4 * N_EMBD)

/* Parameters & gradients (float arrays) */
static float *wte, *d_wte;
static float *wpe, *d_wpe;
static float *lm_head, *d_lm_head;

static float *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];
static float *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];
static float *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];
static float *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];
static float *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];
static float *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];

/* Adam optimizer buffers */
static float *adam_m_wte, *adam_v_wte;
static float *adam_m_wpe, *adam_v_wpe;
static float *adam_m_lm, *adam_v_lm;
static float *adam_m_wq[N_LAYER], *adam_v_wq[N_LAYER];
static float *adam_m_wk[N_LAYER], *adam_v_wk[N_LAYER];
static float *adam_m_wv[N_LAYER], *adam_v_wv[N_LAYER];
static float *adam_m_wo[N_LAYER], *adam_v_wo[N_LAYER];
static float *adam_m_fc1[N_LAYER], *adam_v_fc1[N_LAYER];
static float *adam_m_fc2[N_LAYER], *adam_v_fc2[N_LAYER];

static int num_params = 0;

static float *make_param(int size, float std) {
  float *p = (float *)calloc(size, sizeof(float));
  for (int i = 0; i < size; i++)
    p[i] = rng_gauss(0, std);
  num_params += size;
  return p;
}

static float *make_zero(int size) {
  return (float *)calloc(size, sizeof(float));
}

static void init_params(void) {
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  wte = make_param(es, 0.02f);
  d_wte = make_zero(es);
  adam_m_wte = make_zero(es);
  adam_v_wte = make_zero(es);
  wpe = make_param(ps, 0.02f);
  d_wpe = make_zero(ps);
  adam_m_wpe = make_zero(ps);
  adam_v_wpe = make_zero(ps);
  lm_head = make_param(es, 0.02f);
  d_lm_head = make_zero(es);
  adam_m_lm = make_zero(es);
  adam_v_lm = make_zero(es);
  for (int i = 0; i < N_LAYER; i++) {
    attn_wq[i] = make_param(as, 0.02f);
    d_attn_wq[i] = make_zero(as);
    adam_m_wq[i] = make_zero(as);
    adam_v_wq[i] = make_zero(as);
    attn_wk[i] = make_param(as, 0.02f);
    d_attn_wk[i] = make_zero(as);
    adam_m_wk[i] = make_zero(as);
    adam_v_wk[i] = make_zero(as);
    attn_wv[i] = make_param(as, 0.02f);
    d_attn_wv[i] = make_zero(as);
    adam_m_wv[i] = make_zero(as);
    adam_v_wv[i] = make_zero(as);
    attn_wo[i] = make_param(as, 0.0f);
    d_attn_wo[i] = make_zero(as);
    adam_m_wo[i] = make_zero(as);
    adam_v_wo[i] = make_zero(as);
    mlp_fc1[i] = make_param(ms, 0.02f);
    d_mlp_fc1[i] = make_zero(ms);
    adam_m_fc1[i] = make_zero(ms);
    adam_v_fc1[i] = make_zero(ms);
    mlp_fc2[i] = make_param(ms, 0.0f);
    d_mlp_fc2[i] = make_zero(ms);
    adam_m_fc2[i] = make_zero(ms);
    adam_v_fc2[i] = make_zero(ms);
  }
  printf("num params: %d\n", num_params);
}

/* Saved activations for backward pass */
typedef struct {
  float x_embed[N_EMBD];
  float rms_scale_init;
  float x_in[N_LAYER][N_EMBD];
  float xn_attn[N_LAYER][N_EMBD];
  float rms_scale_attn[N_LAYER];
  float q[N_LAYER][N_EMBD];
  float aw[N_LAYER][N_HEAD][BLOCK_SIZE];
  float attn_out[N_LAYER][N_EMBD];
  float x_mid[N_LAYER][N_EMBD];
  float xn_mlp[N_LAYER][N_EMBD];
  float rms_scale_mlp[N_LAYER];
  float mlp_pre[N_LAYER][MLP_DIM];
  float mlp_post[N_LAYER][MLP_DIM];
  float x_out[N_EMBD];
} PosActs;

static PosActs saved[BLOCK_SIZE];
static float saved_probs[BLOCK_SIZE][MAX_CHARS + 1];

/* KV cache & gradient accumulators */
static float kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dk_accum[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dv_accum[N_LAYER][BLOCK_SIZE][N_EMBD];

/* Forward building blocks (inlined for speed) */
/* Hand-tuned fixed-width kernels (N_EMBD=16 specialised) */

/* dot16: N_EMBD=16 → exactly 2 FMA256 + hsum — no loop overhead */
static __attribute__((always_inline)) inline float dot16(const float *a, const float *b) {
  __m256 r = _mm256_fmadd_ps(_mm256_loadu_ps(a),   _mm256_loadu_ps(b),
             _mm256_mul_ps (_mm256_loadu_ps(a+8),  _mm256_loadu_ps(b+8)));
  __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* dot4: HEAD_DIM=4 → 1 SSE load, no AVX overhead */
static __attribute__((always_inline)) inline float dot4(const float *a, const float *b) {
  __m128 r = _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
  r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
  return _mm_cvtss_f32(r);
}

/* dot64: MLP_DIM=64 → 8 FMA256 passes, fully unrolled */
static __attribute__((always_inline)) inline float dot64(const float *a, const float *b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 64; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
  __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* Drop-in replacements */

/* linear_fwd: dispatch on nin to use the right fixed-width dot */
static __attribute__((always_inline)) inline void linear_fwd(const float *restrict x,
                                       const float *restrict w,
                                       int nout, int nin,
                                       float *restrict out) {
  if (nin == N_EMBD && nout == N_EMBD) {       /* 16×16: QKV, Wo */
    for (int r = 0; r < 16; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == N_EMBD && nout == MLP_DIM) { /* 64×16: fc1 */
    for (int r = 0; r < 64; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == MLP_DIM && nout == N_EMBD) { /* 16×64: fc2 */
    for (int r = 0; r < 16; r++) out[r] = dot64(w + r*64, x);
  } else {                                       /* lm_head: vocab×16, 4-wide */
    __m256 vx0 = _mm256_loadu_ps(x), vx1 = _mm256_loadu_ps(x+8);
    int r = 0;
    for (; r <= nout-4; r += 4) {
      __m256 a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+r*16),    vx0,_mm256_mul_ps(_mm256_loadu_ps(w+r*16+8),    vx1));
      __m256 a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+1)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+1)*16+8),vx1));
      __m256 a2=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+2)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+2)*16+8),vx1));
      __m256 a3=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+3)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+3)*16+8),vx1));
      __m256 t0=_mm256_hadd_ps(a0,a1), t1=_mm256_hadd_ps(a2,a3);
      __m256 t2=_mm256_hadd_ps(t0,t1);
      __m128 lo=_mm256_castps256_ps128(t2), hi=_mm256_extractf128_ps(t2,1);
      _mm_storeu_ps(out+r, _mm_add_ps(lo,hi));
    }
    for (; r < nout; r++) out[r] = dot16(w + r*16, x);
  }
}

/* rmsnorm_fwd: N_EMBD=16 → 2 AVX loads, no loop */
static __attribute__((always_inline)) inline float rmsnorm_fwd(const float *x, int n, float *out) {
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x+8);
  __m256 ss = _mm256_fmadd_ps(v0, v0, _mm256_mul_ps(v1, v1));
  __m128 lo = _mm256_castps256_ps128(ss), hi = _mm256_extractf128_ps(ss, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  float ms = _mm_cvtss_f32(lo) / (float)n;
  float sc = 1.0f / sqrtf(ms + 1e-5f);
  __m256 vs = _mm256_set1_ps(sc);
  _mm256_storeu_ps(out,   _mm256_mul_ps(v0, vs));
  _mm256_storeu_ps(out+8, _mm256_mul_ps(v1, vs));
  return sc;
}

/* fast softmax for inference using Schraudolph exp */
static __attribute__((always_inline)) inline void softmax_fwd(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = fexpf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* precise softmax for training */
static __attribute__((always_inline)) inline void softmax_fwd_precise(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* Backward building blocks */
/* ── Vectorised backward kernels ─────────────────────────────────────────── */

/* linear_bwd_x_16x16: wT(16×16) · dout(16) → dx(16)
   Each column of w becomes a dot-product against dout. */
static __attribute__((always_inline)) inline
void linear_bwd_x_16x16(const float *restrict w,
                         const float *restrict dout,
                         float *restrict dx) {
    __m256 vd0 = _mm256_loadu_ps(dout),     /* dout[0..7]  */
           vd1 = _mm256_loadu_ps(dout + 8); /* dout[8..15] */
    /* For each output col c, gather the c-th element from every row of w.
       With nin=16 rows and nout=16, col c = {w[0*16+c], w[1*16+c], ...}
       We do it 8 cols at a time using a transpose + FMA approach:
       accumulate dout[r] * w[r*16+c] across all 16 rows. */
    for (int c = 0; c < 16; c++) {
        /* Build a vector of column c across all 16 rows */
        __m256 col0 = _mm256_set_ps(w[7*16+c], w[6*16+c], w[5*16+c], w[4*16+c],
                                    w[3*16+c], w[2*16+c], w[1*16+c], w[0*16+c]);
        __m256 col1 = _mm256_set_ps(w[15*16+c],w[14*16+c],w[13*16+c],w[12*16+c],
                                    w[11*16+c],w[10*16+c],w[9*16+c], w[8*16+c]);
        __m256 r = _mm256_fmadd_ps(col0, vd0, _mm256_mul_ps(col1, vd1));
        __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        dx[c] += _mm_cvtss_f32(lo);
    }
}

/* linear_bwd_x_16x64: wT(16×64) · dout(16) → dx(64)
   w is [16 rows × 64 cols], dout is 16-wide.
   dx[c] += sum_r dout[r] * w[r*64+c]  for c in 0..63 */
static __attribute__((always_inline)) inline
void linear_bwd_x_16x64(const float *restrict w,
                          const float *restrict dout,
                          float *restrict dx) {
    __m256 vd[16];
    for (int r = 0; r < 16; r++) vd[r] = _mm256_set1_ps(dout[r]);
    /* Process 8 output cols at a time */
    for (int c = 0; c < 64; c += 8) {
        __m256 acc = _mm256_loadu_ps(dx + c); /* accumulate into existing dx */
        for (int r = 0; r < 16; r++)
            acc = _mm256_fmadd_ps(vd[r], _mm256_loadu_ps(w + r*64 + c), acc);
        _mm256_storeu_ps(dx + c, acc);
    }
}

/* linear_bwd_x_64x16: wT(64×16) · dout(64) → dx(16)
   w is [64 rows × 16 cols], dout is 64-wide.
   dx[c] += sum_r dout[r] * w[r*16+c]  → same as dot of col c vs dout */
static __attribute__((always_inline)) inline
void linear_bwd_x_64x16(const float *restrict w,
                          const float *restrict dout,
                          float *restrict dx) {
    for (int c = 0; c < 16; c++) {
        /* dot col c of w (stride 16, 64 elems) with dout[64] */
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int r = 0; r < 64; r += 16) {
            __m256 col0 = _mm256_set_ps(w[(r+7)*16+c],w[(r+6)*16+c],w[(r+5)*16+c],w[(r+4)*16+c],
                                        w[(r+3)*16+c],w[(r+2)*16+c],w[(r+1)*16+c],w[(r+0)*16+c]);
            __m256 col1 = _mm256_set_ps(w[(r+15)*16+c],w[(r+14)*16+c],w[(r+13)*16+c],w[(r+12)*16+c],
                                        w[(r+11)*16+c],w[(r+10)*16+c],w[(r+9)*16+c], w[(r+8)*16+c]);
            acc0 = _mm256_fmadd_ps(col0, _mm256_loadu_ps(dout + r),     acc0);
            acc1 = _mm256_fmadd_ps(col1, _mm256_loadu_ps(dout + r + 8), acc1);
        }
        __m256 r2 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(r2), hi = _mm256_extractf128_ps(r2, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        dx[c] += _mm_cvtss_f32(lo);
    }
}

/* linear_bwd_w_16x16: dout(16) ⊗ x(16) → dw(16×16)
   dw[r*16+c] += dout[r] * x[c]  — outer product, 16 rows */
static __attribute__((always_inline)) inline
void linear_bwd_w_16x16(const float *restrict x,
                          const float *restrict dout,
                          float *restrict dw) {
    __m256 vx0 = _mm256_loadu_ps(x), vx1 = _mm256_loadu_ps(x + 8);
    for (int r = 0; r < 16; r++) {
        __m256 vdr = _mm256_set1_ps(dout[r]);
        float *dwr = dw + r * 16;
        _mm256_storeu_ps(dwr,     _mm256_fmadd_ps(vdr, vx0, _mm256_loadu_ps(dwr)));
        _mm256_storeu_ps(dwr + 8, _mm256_fmadd_ps(vdr, vx1, _mm256_loadu_ps(dwr + 8)));
    }
}

/* linear_bwd_w_64x16: dout(64) ⊗ x(16) → dw(64×16) */
static __attribute__((always_inline)) inline
void linear_bwd_w_64x16(const float *restrict x,
                          const float *restrict dout,
                          float *restrict dw) {
    __m256 vx0 = _mm256_loadu_ps(x), vx1 = _mm256_loadu_ps(x + 8);
    for (int r = 0; r < 64; r++) {
        __m256 vdr = _mm256_set1_ps(dout[r]);
        float *dwr = dw + r * 16;
        _mm256_storeu_ps(dwr,     _mm256_fmadd_ps(vdr, vx0, _mm256_loadu_ps(dwr)));
        _mm256_storeu_ps(dwr + 8, _mm256_fmadd_ps(vdr, vx1, _mm256_loadu_ps(dwr + 8)));
    }
}

/* linear_bwd_w_16x64: dout(16) ⊗ x(64) → dw(16×64) */
static __attribute__((always_inline)) inline
void linear_bwd_w_16x64(const float *restrict x,
                          const float *restrict dout,
                          float *restrict dw) {
    for (int r = 0; r < 16; r++) {
        __m256 vdr = _mm256_set1_ps(dout[r]);
        float *dwr = dw + r * 64;
        for (int c = 0; c < 64; c += 8)
            _mm256_storeu_ps(dwr+c, _mm256_fmadd_ps(vdr, _mm256_loadu_ps(x+c),
                                                    _mm256_loadu_ps(dwr+c)));
    }
}

/* Generic fallbacks (only used for lm_head: vocab×16) */
static inline void linear_bwd_x(const float *restrict w,
                                 const float *restrict dout, int nout, int nin,
                                 float *restrict dx) {
    if (nout == 16 && nin == 16) { linear_bwd_x_16x16(w, dout, dx); return; }
    if (nout == 16 && nin == 64) { linear_bwd_x_16x64(w, dout, dx); return; }
    if (nout == 64 && nin == 16) { linear_bwd_x_64x16(w, dout, dx); return; }
    /* vocab×16 fallback */
    __m256 vd0 = _mm256_loadu_ps(dout), vd1 = _mm256_loadu_ps(dout + 8);
    for (int c = 0; c < nin; c++) {
        float s = dx[c];
        for (int r = 0; r < nout; r++) s += dout[r] * w[r * nin + c];
        dx[c] = s;
    }
}

static inline void linear_bwd_w(const float *restrict x,
                                 const float *restrict dout, int nout, int nin,
                                 float *restrict dw) {
    if (nout == 16 && nin == 16) { linear_bwd_w_16x16(x, dout, dw); return; }
    if (nout == 64 && nin == 16) { linear_bwd_w_64x16(x, dout, dw); return; }
    if (nout == 16 && nin == 64) { linear_bwd_w_16x64(x, dout, dw); return; }
    /* vocab×16 fallback */
    for (int r = 0; r < nout; r++) {
        float dr = dout[r];
        float *dwr = dw + r * nin;
        for (int c = 0; c < nin; c++) dwr[c] += dr * x[c];
    }
}

static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0;
  for (int i = 0; i < n; i++)
    dot += dout[i] * x[i];
  float coeff = scale * scale * scale / n;
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}

/* GPT forward pass (one token, fills saved acts) */
static void gpt_forward(int token_id, int pos_id, float *logits_out,
                        PosActs *act) {
  float x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];

  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token_id * N_EMBD + i] + wpe[pos_id * N_EMBD + i];
  memcpy(act->x_embed, x, sizeof(x));

  act->rms_scale_init = rmsnorm_fwd(x, N_EMBD, x);

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], x, sizeof(x));

    float xn[N_EMBD];
    act->rms_scale_attn[li] = rmsnorm_fwd(x, N_EMBD, xn);
    memcpy(act->xn_attn[li], xn, sizeof(xn));

    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
    memcpy(act->q[li], q, sizeof(q));

    memcpy(kv_keys[li][pos_id], k, sizeof(k));
    memcpy(kv_vals[li][pos_id], v, sizeof(v));
    int seq_len = pos_id + 1;
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

    float ao[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float al[BLOCK_SIZE];
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] = dot4(q + hs, kv_keys[li][tt] + hs) * scale;
      float mx = al[0];
      for (int tt = 1; tt < seq_len; tt++)
        if (al[tt] > mx)
          mx = al[tt];
      float sm = 0;
      for (int tt = 0; tt < seq_len; tt++) {
        al[tt] = fexpf(al[tt] - mx);
        sm += al[tt];
      }
      float inv = 1.0f / sm;
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] *= inv;
      for (int tt = 0; tt < seq_len; tt++)
        act->aw[li][h][tt] = al[tt];
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0;
        for (int tt = 0; tt < seq_len; tt++)
          s += al[tt] * kv_vals[li][tt][hs + j];
        ao[hs + j] = s;
      }
    }
    memcpy(act->attn_out[li], ao, sizeof(ao));

    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_in[li][i];
    memcpy(act->x_mid[li], x, sizeof(x));

    float xn_m[N_EMBD];
    act->rms_scale_mlp[li] = rmsnorm_fwd(x, N_EMBD, xn_m);
    memcpy(act->xn_mlp[li], xn_m, sizeof(xn_m));

    float h1[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
    memcpy(act->mlp_pre[li], h1, MLP_DIM * sizeof(float));

    float h2[MLP_DIM];
    for (int i = 0; i < MLP_DIM; i++)
      h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    memcpy(act->mlp_post[li], h2, MLP_DIM * sizeof(float));

    linear_fwd(h2, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_mid[li][i];
  }

  memcpy(act->x_out, x, sizeof(x));
  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}

/* Backward pass for all positions */
static void gpt_backward(int n, const int *tokens, const int *targets) {
  memset(dk_accum, 0, sizeof(dk_accum));
  memset(dv_accum, 0, sizeof(dv_accum));
  float inv_n = 1.0f / n;

  for (int pos = n - 1; pos >= 0; pos--) {
    PosActs *act = &saved[pos];
    int seq_len = pos + 1;

    float dl[MAX_CHARS + 1];
    for (int i = 0; i < vocab_size; i++)
      dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;

    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));
    linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);
    linear_bwd_w(act->x_out, dl, vocab_size, N_EMBD, d_lm_head);

    for (int li = N_LAYER - 1; li >= 0; li--) {
      /* MLP backward */
      float d_h2[MLP_DIM];
      memset(d_h2, 0, sizeof(d_h2));
      linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
      linear_bwd_w(act->mlp_post[li], dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);

      float d_h1[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_h1[i] =
            act->mlp_pre[li][i] > 0 ? 2.0f * act->mlp_pre[li][i] * d_h2[i] : 0;

      float d_xn_mlp[N_EMBD];
      memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
      linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
      linear_bwd_w(act->xn_mlp[li], d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);

      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], d_xn_mlp, N_EMBD,
                  d_x_mid);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] += d_x_mid[i];

      /* Attention backward */
      float d_ao[N_EMBD];
      memset(d_ao, 0, sizeof(d_ao));
      linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
      linear_bwd_w(act->attn_out[li], dx, N_EMBD, N_EMBD, d_attn_wo[li]);

      float d_q[N_EMBD];
      memset(d_q, 0, sizeof(d_q));
      float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        float d_aw[BLOCK_SIZE];
        memset(d_aw, 0, sizeof(d_aw));
        for (int j = 0; j < HEAD_DIM; j++) {
          for (int tt = 0; tt < seq_len; tt++) {
            d_aw[tt] += d_ao[hs + j] * kv_vals[li][tt][hs + j];
            dv_accum[li][tt][hs + j] += act->aw[li][h][tt] * d_ao[hs + j];
          }
        }
        float dot = 0;
        for (int tt = 0; tt < seq_len; tt++)
          dot += d_aw[tt] * act->aw[li][h][tt];
        float d_al[BLOCK_SIZE];
        for (int tt = 0; tt < seq_len; tt++)
          d_al[tt] = act->aw[li][h][tt] * (d_aw[tt] - dot);
        for (int tt = 0; tt < seq_len; tt++) {
          for (int j = 0; j < HEAD_DIM; j++) {
            d_q[hs + j] += d_al[tt] * kv_keys[li][tt][hs + j] * scale;
            dk_accum[li][tt][hs + j] += d_al[tt] * act->q[li][hs + j] * scale;
          }
        }
      }

      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));
      linear_bwd_x(attn_wq[li], d_q, N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], d_q, N_EMBD, N_EMBD, d_attn_wq[li]);
      linear_bwd_x(attn_wk[li], dk_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dk_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wk[li]);
      linear_bwd_x(attn_wv[li], dv_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dv_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wv[li]);

      float d_x_in[N_EMBD];
      memset(d_x_in, 0, sizeof(d_x_in));
      rmsnorm_bwd(act->x_in[li], act->rms_scale_attn[li], d_xn, N_EMBD, d_x_in);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] = dx[i] + d_x_in[i];
    }

    float d_embed[N_EMBD];
    memset(d_embed, 0, sizeof(d_embed));
    rmsnorm_bwd(act->x_embed, act->rms_scale_init, dx, N_EMBD, d_embed);

    int tok = tokens[pos];
    for (int i = 0; i < N_EMBD; i++) {
      d_wte[tok * N_EMBD + i] += d_embed[i];
      d_wpe[pos * N_EMBD + i] += d_embed[i];
    }
  }
}

/* Adam update helper */
/* Adam update helper — AVX2 vectorised */
static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  float b1c = 1.0f - powf(b1, step + 1);
  float b2c = 1.0f - powf(b2, step + 1);
  float lr_b1c = lr / b1c;

  __m256 vb1   = _mm256_set1_ps(b1);
  __m256 vb1i  = _mm256_set1_ps(1.0f - b1);
  __m256 vb2   = _mm256_set1_ps(b2);
  __m256 vb2i  = _mm256_set1_ps(1.0f - b2);
  __m256 vlr   = _mm256_set1_ps(lr_b1c);
  __m256 vb2c  = _mm256_set1_ps(b2c);
  __m256 veps  = _mm256_set1_ps(eps);
  __m256 vzero = _mm256_setzero_ps();

  int i = 0;
  for (; i <= sz - 8; i += 8) {
    __m256 vg = _mm256_loadu_ps(g + i);
    __m256 vm = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(m + i), _mm256_mul_ps(vb1i, vg));
    __m256 vv = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(v + i), _mm256_mul_ps(vb2i, _mm256_mul_ps(vg, vg)));
    _mm256_storeu_ps(m + i, vm);
    _mm256_storeu_ps(v + i, vv);
    __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(vv, vb2c)), veps);
    __m256 vp   = _mm256_fnmadd_ps(vlr, _mm256_div_ps(vm, denom), _mm256_loadu_ps(p + i));
    _mm256_storeu_ps(p + i, vp);
    _mm256_storeu_ps(g + i, vzero);
  }
  /* scalar tail */
  for (; i < sz; i++) {
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    p[i] -= lr_b1c * (m[i] / (sqrtf(v[i] / b2c) + eps));
    g[i] = 0;
  }
}

/* Weighted random choice */
static int weighted_choice(const float *w, int n) {
  float total = 0;
  for (int i = 0; i < n; i++)
    total += w[i];
  float r = (float)rng_uniform() * total, cum = 0;
  for (int i = 0; i < n; i++) {
    cum += w[i];
    if (r < cum)
      return i;
  }
  return n - 1;
}

/* Main: training + inference */
int main(void) {
  load_dataset("new_names.txt");

  int *doc_order = (int *)malloc(num_docs * sizeof(int));
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_ints(doc_order, num_docs);
  char (*docs_tmp)[MAX_DOC_LEN] = malloc((size_t)num_docs * MAX_DOC_LEN);
  for (int i = 0; i < num_docs; i++)
    memcpy(docs_tmp[i], docs[doc_order[i]], MAX_DOC_LEN);
  memcpy(docs, docs_tmp, (size_t)num_docs * MAX_DOC_LEN);
  free(docs_tmp);
  free(doc_order);

  printf("num docs: %d\n", num_docs);
  build_tokenizer();
  printf("vocab size: %d\n", vocab_size);
  init_params();

  float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  float running_loss = 3.3f;
  int num_steps = 60000;

  for (int step = 0; step < num_steps; step++) {
    char *doc = docs[step % num_docs];
    int doc_len = (int)strlen(doc);

    int tokens[MAX_DOC_LEN + 2], targets[BLOCK_SIZE];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);

    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));

    float total_loss = 0;
    float logits[MAX_CHARS + 1];
    for (int pos = 0; pos < n; pos++) {
      targets[pos] = tokens[pos + 1];
      gpt_forward(tokens[pos], pos, logits, &saved[pos]);
      softmax_fwd_precise(logits, vocab_size, saved_probs[pos]);
      total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
    }
    float loss = total_loss / n;

    gpt_backward(n, tokens, targets);

    /* gradient clipping: scale all grads if global norm > 1.0 */
    {
      float gnorm2 = 0;
      int es2 = vocab_size * N_EMBD, ps2 = BLOCK_SIZE * N_EMBD;
      int as2 = N_EMBD * N_EMBD, ms2 = MLP_DIM * N_EMBD;
      for (int i = 0; i < es2; i++) gnorm2 += d_wte[i]*d_wte[i] + d_lm_head[i]*d_lm_head[i];
      for (int i = 0; i < ps2; i++) gnorm2 += d_wpe[i]*d_wpe[i];
      for (int l = 0; l < N_LAYER; l++) {
        for (int i = 0; i < as2; i++)
          gnorm2 += d_attn_wq[l][i]*d_attn_wq[l][i] + d_attn_wk[l][i]*d_attn_wk[l][i]
                  + d_attn_wv[l][i]*d_attn_wv[l][i] + d_attn_wo[l][i]*d_attn_wo[l][i];
        for (int i = 0; i < ms2; i++)
          gnorm2 += d_mlp_fc1[l][i]*d_mlp_fc1[l][i] + d_mlp_fc2[l][i]*d_mlp_fc2[l][i];
      }
      float gnorm = sqrtf(gnorm2);
      float clip = 1.0f;
      if (gnorm > clip) {
        float scale = clip / gnorm;
        for (int i = 0; i < es2; i++) { d_wte[i]*=scale; d_lm_head[i]*=scale; }
        for (int i = 0; i < ps2; i++) d_wpe[i]*=scale;
        for (int l = 0; l < N_LAYER; l++) {
          for (int i = 0; i < as2; i++) {
            d_attn_wq[l][i]*=scale; d_attn_wk[l][i]*=scale;
            d_attn_wv[l][i]*=scale; d_attn_wo[l][i]*=scale;
          }
          for (int i = 0; i < ms2; i++) { d_mlp_fc1[l][i]*=scale; d_mlp_fc2[l][i]*=scale; }
        }
      }
    }
    float lr_t =
        lr * 0.5f * (1.0f + cosf((float)M_PI * step / (float)num_steps));
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
    adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps,
                step);
    adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps,
                step);
    adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps,
                step);
    for (int i = 0; i < N_LAYER; i++) {
      adam_update(attn_wq[i], d_attn_wq[i], adam_m_wq[i], adam_v_wq[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wk[i], d_attn_wk[i], adam_m_wk[i], adam_v_wk[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wv[i], d_attn_wv[i], adam_m_wv[i], adam_v_wv[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wo[i], d_attn_wo[i], adam_m_wo[i], adam_v_wo[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i], adam_v_fc1[i], ms,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i], adam_v_fc2[i], ms,
                  lr_t, b1, b2, eps, step);
    }

    running_loss = running_loss * 0.99f + loss * 0.01f;
    if ((step + 1) % 100 == 0 || step == 0 || step == num_steps - 1)
      printf("step %4d / %4d | loss %.4f  (avg %.4f)\n",
             step + 1, num_steps, loss, running_loss);
  }

  float temperature = 1.4f;
  clock_t start = clock();
  int total_tokens = 0;
  printf("\n--- inference ---\n");

  /* Show 10 sample names */
  for (int si = 0; si < 5000; si++) {
    int token_id = BOS;
    static PosActs tmp_act;
    char buf[BLOCK_SIZE + 1] = {0};
    int len = 0;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
          total_tokens++;
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(token_id, pos, logits, &tmp_act);
      float inv_t = 1.0f / temperature;
      for (int i = 0; i < vocab_size; i++) logits[i] *= inv_t;
      softmax_fwd(logits, vocab_size, probs);
      token_id = weighted_choice(probs, vocab_size);
      if (token_id == BOS) break;
      if (token_id < num_uchars) buf[len++] = uchars_arr[token_id];
    }
    printf("sample %2d: %s\n", si + 1, buf);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }
  clock_t end = clock();
float time_taken = (float)(end - start) / CLOCKS_PER_SEC;

printf("\n--- performance ---\n");
printf("Time: %f seconds\n", time_taken);
printf("Tokens: %d\n", total_tokens);
printf("Speed: %f tok/s\n", total_tokens / time_taken);

  /* Benchmark: continuous wrap, no KV reset — matches talos-vs-macbook methodology */
  long long N = 5000000LL;
  long long emitted = 0;
  int tok = BOS, pos = 0;
  double t0 = now_s();

  for (long long i = 0; i < N; i++) {
    if (pos >= BLOCK_SIZE) { pos = 0; }
    float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
    static PosActs tmp_act;
    gpt_forward(tok, pos, logits, &tmp_act);
    float inv_t = 1.0f / temperature;
    for (int i2 = 0; i2 < vocab_size; i2++) logits[i2] *= inv_t;
    softmax_fwd(logits, vocab_size, probs);
    int nxt = weighted_choice(probs, vocab_size);
    if (nxt == BOS) { tok = BOS; pos = 0; }
    else { tok = nxt; pos++; }
    emitted++;
  }

  double elapsed = now_s() - t0;
  printf("  c fp32+AVX2              %14.0f tok/sec\n", emitted / elapsed);

  /* cleanup */
  free(wte);
  free(d_wte);
  free(adam_m_wte);
  free(adam_v_wte);
  free(wpe);
  free(d_wpe);
  free(adam_m_wpe);
  free(adam_v_wpe);
  free(lm_head);
  free(d_lm_head);
  free(adam_m_lm);
  free(adam_v_lm);
  for (int i = 0; i < N_LAYER; i++) {
    free(attn_wq[i]);
    free(d_attn_wq[i]);
    free(adam_m_wq[i]);
    free(adam_v_wq[i]);
    free(attn_wk[i]);
    free(d_attn_wk[i]);
    free(adam_m_wk[i]);
    free(adam_v_wk[i]);
    free(attn_wv[i]);
    free(d_attn_wv[i]);
    free(adam_m_wv[i]);
    free(adam_v_wv[i]);
    free(attn_wo[i]);
    free(d_attn_wo[i]);
    free(adam_m_wo[i]);
    free(adam_v_wo[i]);
    free(mlp_fc1[i]);
    free(d_mlp_fc1[i]);
    free(adam_m_fc1[i]);
    free(adam_v_fc1[i]);
    free(mlp_fc2[i]);
    free(d_mlp_fc2[i]);
    free(adam_m_fc2[i]);
    free(adam_v_fc2[i]);
  }
  return 0;
}
