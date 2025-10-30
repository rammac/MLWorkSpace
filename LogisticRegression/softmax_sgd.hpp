#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

namespace softmax_detail {

// Dense dot product: dot(a, b) over D floats
static inline float dot(const float* __restrict a,
                        const float* __restrict b,
                        int D) 
{
    float s = 0.f;
    for (int j = 0; j < D; ++j) 
        s += a[j] * b[j];
    return s;
}

// Compute logits z = W·x + b, where W is class-major: W[k*D + j]
static inline void compute_logits(const float* __restrict W,
                                  const float* __restrict b,
                                  const float* __restrict x,
                                  int D, int K,
                                  float* __restrict z_out) 
{
    for (int k = 0; k < K; ++k) 
    {
        const float* wk = &W[k * D];
        z_out[k] = dot(wk, x, D) + b[k];
    }
}

// In-place, numerically-stable softmax on length-K vector z.
// Returns nothing; z becomes probabilities. Uses subtract-max trick.
static inline void softmax_inplace(float* __restrict z, int K) {
    float z_max = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < K; ++k) 
    {
        if (z[k] > z_max) 
            z_max = z[k];
    }

    double denom = 0.0;
    for (int k = 0; k < K; ++k) 
    {
        z[k] = std::exp(z[k] - z_max);
        denom += z[k];
    }
    const float inv = static_cast<float>(1.0 / denom);
    for (int k = 0; k < K; ++k) 
        z[k] *= inv;
}

// Cross-entropy for a single example given probabilities p and true class y
static inline float cross_entropy(const float* __restrict p, int y) 
{
    const float eps = 1e-30f;
    return -std::log(p[y] > eps ? p[y] : eps);
}

// Accumulate gradients for one example into gW (K*D) and gb (K)
// delta_k = p_k - 1[y==k];  gW_k += delta_k * x;  gb_k += delta_k
static inline void accumulate_grad(const float* __restrict x,
                                   const float* __restrict p,
                                   int y,
                                   int D, int K,
                                   float* __restrict gW,
                                   float* __restrict gb) {
    for (int k = 0; k < K; ++k) 
    {
        const float delta = p[k] - (k == y ? 1.f : 0.f);
        gb[k] += delta;
        float* gWk = &gW[k * D];
        for (int j = 0; j < D; ++j) gWk[j] += delta * x[j];
    }
}

} // namespace softmax_detail

struct SoftmaxSGD {
    int D;                // features (e.g., 784)
    int K;                // classes  (e.g., 10)
    float lambda;         // L2 coefficient
    std::vector<float> W; // size K*D, layout: [k][j] => W[k*D + j]
    std::vector<float> b; // size K

    explicit SoftmaxSGD(int D_, int K_, float lambda_=1e-4f)
        : D(D_), K(K_), lambda(lambda_), W(K_*D_, 0.f), b(K_, 0.f) {}

    void init(unsigned seed=42, float scale=0.01f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, scale);
        for (auto &w : W) w = nd(rng);
        std::fill(b.begin(), b.end(), 0.f);
    }

    // One minibatch SGD step (fused). X: N x D (row-major), y: N ints in [0,K)
    // Returns mean loss (cross-entropy + L2/2).
    float batch_update(const float* __restrict X,
                       const int*   __restrict y,
                       int N, float lr) {
        using namespace softmax_detail;

        std::vector<float> gW(K * D, 0.f);
        std::vector<float> gb(K, 0.f);
        std::vector<float> z(K); // scratch for logits/probs; reused per sample

        double loss_sum = 0.0;

        for (int i = 0; i < N; ++i) {
            const float* xi = X + static_cast<size_t>(i) * D;

            // z = W·x + b
            compute_logits(W.data(), b.data(), xi, D, K, z.data());

            // p = softmax(z)
            softmax_inplace(z.data(), K);

            // accumulate loss and gradients
            loss_sum += cross_entropy(z.data(), y[i]);
            accumulate_grad(xi, z.data(), y[i], D, K, gW.data(), gb.data());
        }

        const float invN = 1.f / static_cast<float>(N);

        // Average grads and add L2
        for (int k = 0; k < K; ++k) {
            float* gWk = &gW[k * D];
            for (int j = 0; j < D; ++j) {
                gWk[j] = gWk[j] * invN + lambda * W[k * D + j];
            }
            gb[k] *= invN; // no L2 on bias
        }

        // SGD update
        for (int k = 0; k < K; ++k) {
            float* Wk  = &W[k * D];
            float* gWk = &gW[k * D];
            for (int j = 0; j < D; ++j) Wk[j] -= lr * gWk[j];
            b[k] -= lr * gb[k];
        }

        // Mean loss + L2/2
        double l2 = 0.0;
        for (float w : W) l2 += double(w) * double(w);
        double loss = (loss_sum * invN) + 0.5 * lambda * l2;
        return static_cast<float>(loss);
    }

    // Predict top-1 for a single sample (for quick accuracy checks)
    int predict_one(const float* __restrict x) const {
        int argmax = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < K; ++k) {
            const float* wk = &W[k * D];
            float s = b[k];
            for (int j = 0; j < D; ++j) s += wk[j] * x[j];
            if (s > best) { best = s; argmax = k; }
        }
        return argmax;
    }

    // Helpers
    // Mean accuracy over N samples
    float eval_accuracy(const float* X, const int* y, int N) const 
    {
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const float* xi = X + size_t(i)*D;
            if (predict_one(xi) == y[i]) ++correct;
        }
        return float(correct) / float(N);
    }

    // Mean log-loss over N samples (no L2 term)
    float eval_logloss(const float* X, const int* y, int N) const {
        using namespace softmax_detail;
        std::vector<float> z(K);
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const float* xi = X + size_t(i)*D;
            compute_logits(W.data(), b.data(), xi, D, K, z.data());
            softmax_inplace(z.data(), K);
            sum += cross_entropy(z.data(), y[i]);
        }
        return float(sum / double(N));
    }

    // One epoch over shuffled mini-batches; returns mean batch loss
    float train_one_epoch(const float* X, const int* y, int N,
                        int batch_size, float lr, unsigned seed) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        double loss_accum = 0.0;
        int batches = 0;

        std::vector<float> Xbuf; Xbuf.resize(size_t(batch_size)*D);
        std::vector<int>   ybuf; ybuf.resize(batch_size);

        for (int start = 0; start < N; start += batch_size) {
            int n = std::min(batch_size, N - start);
            // gather contiguous minibatch
            for (int i = 0; i < n; ++i) {
                int src = idx[start + i];
                const float* xi = X + size_t(src)*D;
                std::copy(xi, xi + D, Xbuf.data() + size_t(i)*D);
                ybuf[i] = y[src];
            }
            loss_accum += batch_update(Xbuf.data(), ybuf.data(), n, lr);
            ++batches;
        }
        return float(loss_accum / std::max(1, batches));
    }
};
