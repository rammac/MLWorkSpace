#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

namespace softmax_detail 
{

template<typename T>
static inline T dot(const T* __restrict a, const T* __restrict b, int D) {
    T s = T(0);
    for (int j = 0; j < D; ++j) s += a[j] * b[j];
    return s;
}

template<typename T>
static inline void compute_logits(const T* __restrict W,
                                  const T* __restrict b,
                                  const T* __restrict x,
                                  int D, int K,
                                  T* __restrict z_out) {
    for (int k = 0; k < K; ++k) {
        const T* wk = &W[k * D];
        z_out[k] = dot<T>(wk, x, D) + b[k];
    }
}

template<typename T>
static inline void softmax_inplace(T* __restrict z, int K) {
    T z_max = -std::numeric_limits<T>::infinity();
    for (int k = 0; k < K; ++k) if (z[k] > z_max) z_max = z[k];
    long double denom = 0.0L;
    for (int k = 0; k < K; ++k) { z[k] = std::exp(z[k] - z_max); denom += z[k]; }
    const T inv = T(1.0L / denom);
    for (int k = 0; k < K; ++k) z[k] *= inv;
}

template<typename T>
static inline T cross_entropy(const T* __restrict p, int y) {
    const T eps = T(1e-30);
    return -std::log(p[y] > eps ? p[y] : eps);
}

template<typename T>
static inline void accumulate_grad(const T* __restrict x,
                                   const T* __restrict p,
                                   int y,
                                   int D, int K,
                                   T* __restrict gW,
                                   T* __restrict gb) {
    for (int k = 0; k < K; ++k) {
        const T delta = p[k] - (k == y ? T(1) : T(0));
        gb[k] += delta;
        T* gWk = &gW[k * D];
        for (int j = 0; j < D; ++j) gWk[j] += delta * x[j];
    }
}

} // namespace softmax_detail

template<typename T>
struct SoftmaxSGD {
    int D, K;
    T lambda;
    std::vector<T> W; // K*D, class-major
    std::vector<T> b; // K

    explicit SoftmaxSGD(int D_, int K_, T lambda_=T(1e-4))
        : D(D_), K(K_), lambda(lambda_), W(size_t(K_)*D_, T(0)), b(K_, T(0)) {}

    void init(unsigned seed=42, T scale=T(0.01)) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> nd(0.0, double(scale));
        for (auto &w : W) w = T(nd(rng));
        std::fill(b.begin(), b.end(), T(0));
    }

    T batch_update(const T* __restrict X, const int* __restrict y, int N, T lr) {
        using namespace softmax_detail;
        std::vector<T> gW(K * D, T(0));
        std::vector<T> gb(K, T(0));
        std::vector<T> z(K);
        long double loss_sum = 0.0L;

        for (int i = 0; i < N; ++i) {
            const T* xi = X + size_t(i) * D;
            compute_logits<T>(W.data(), b.data(), xi, D, K, z.data());
            softmax_inplace<T>(z.data(), K);
            loss_sum += cross_entropy<T>(z.data(), y[i]);
            accumulate_grad<T>(xi, z.data(), y[i], D, K, gW.data(), gb.data());
        }

        const T invN = T(1) / T(N);
        for (int k = 0; k < K; ++k) {
            T* gWk = &gW[k * D];
            for (int j = 0; j < D; ++j) gWk[j] = gWk[j] * invN + lambda * W[k * D + j];
            gb[k] *= invN;
        }

        for (int k = 0; k < K; ++k) {
            T* Wk  = &W[k * D];
            T* gWk = &gW[k * D];
            for (int j = 0; j < D; ++j) Wk[j] -= lr * gWk[j];
            b[k] -= lr * gb[k];
        }

        long double l2 = 0.0L;
        for (T w : W) l2 += (long double)w * (long double)w;
        return T(loss_sum / (long double)N + 0.5L * (long double)lambda * l2);
    }

    int predict_one(const T* __restrict x) const {
        int argmax = 0;
        T best = -std::numeric_limits<T>::infinity();
        for (int k = 0; k < K; ++k) {
            const T* wk = &W[k * D];
            T s = b[k];
            for (int j = 0; j < D; ++j) s += wk[j] * x[j];
            if (s > best) { best = s; argmax = k; }
        }
        return argmax;
    }

    T eval_accuracy(const T* X, const int* y, int N) const {
        int correct = 0;
        for (int i = 0; i < N; ++i)
            if (predict_one(X + size_t(i)*D) == y[i]) ++correct;
        return T(correct) / T(N);
    }

    T eval_logloss(const T* X, const int* y, int N) const {
        using namespace softmax_detail;
        std::vector<T> z(K);
        long double sum = 0.0L;
        for (int i = 0; i < N; ++i) {
            const T* xi = X + size_t(i)*D;
            compute_logits<T>(W.data(), b.data(), xi, D, K, z.data());
            softmax_inplace<T>(z.data(), K);
            sum += cross_entropy<T>(z.data(), y[i]);
        }
        return T(sum / (long double)N);
    }

    T train_one_epoch(const T* X, const int* y, int N,
                      int batch_size, T lr, unsigned seed) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        long double loss_accum = 0.0L;
        int batches = 0;
        std::vector<T> Xbuf(size_t(batch_size)*D);
        std::vector<int> ybuf(batch_size);

        for (int start = 0; start < N; start += batch_size) {
            int n = std::min(batch_size, N - start);
            for (int i = 0; i < n; ++i) {
                int src = idx[start + i];
                const T* xi = X + size_t(src)*D;
                std::copy(xi, xi + D, Xbuf.data() + size_t(i)*D);
                ybuf[i] = y[src];
            }
            loss_accum += batch_update(Xbuf.data(), ybuf.data(), n, lr);
            ++batches;
        }
        return T(loss_accum / std::max(1, batches));
    }
};
