#include "softmax_sgd.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <sys/resource.h> // getrusage (macOS/Unix)

static double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static double rss_megabytes() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__) && defined(__MACH__)
    // macOS reports ru_maxrss in bytes
    return ru.ru_maxrss / (1024.0 * 1024.0);
#else
    // Linux reports in kilobytes
    return ru.ru_maxrss / 1024.0;
#endif
}

int main() {
    const int D = 784;
    const int K = 10;
    const int N = 1024;          // one minibatch for the smoke test
    const float lr = 0.1f;
    const float lambda = 1e-4f;
    const unsigned seed = 42;

    // Fake data: standard normal features, random labels
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> uid(0, K - 1);

    std::vector<float> X(size_t(N) * D);
    std::vector<int>   y(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) X[size_t(i)*D + j] = nd(rng);
        y[i] = uid(rng);
    }

    SoftmaxSGD sgd(D, K, lambda);
    sgd.init(seed);

    double t0 = now_seconds();
    float loss = sgd.batch_update(X.data(), y.data(), N, lr);
    double t1 = now_seconds();

    std::cout << "Batch loss: " << loss << "\n";
    std::cout << "Time (s):   " << (t1 - t0) << "\n";
    std::cout << "Peak RSS (MB): " << rss_megabytes() << "\n";

    // Quick functional sanity check
    int pred0 = sgd.predict_one(X.data());
    std::cout << "predict_one on first sample -> class " << pred0 << "\n";
    return 0;
}
