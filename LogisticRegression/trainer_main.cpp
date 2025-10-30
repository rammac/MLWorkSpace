// trainer_main.cpp
#include "softmax_sgd.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sys/resource.h>   // getrusage
#include <sys/time.h>       // getrusage
#include <cstdlib>
#include <cstring>

using Scalar = float;

#include <type_traits>

template <typename T>
constexpr std::string type_name() 
{
    if (std::is_same<T, float>::value) return "float";
    else if (std::is_same<T, double>::value) return "double";
    else if (std::is_same<T, long double>::value) return "long double";
    else return "unknown";
}

static double wall_now() 
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static double cpu_now() 
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    double ut = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec/1e6;
    double st = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec/1e6;
    return ut + st;
}

static double peak_rss_mb() 
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__) && defined(__MACH__)
    return ru.ru_maxrss / (1024.0 * 1024.0); // bytes → MB
#else
    return ru.ru_maxrss / 1024.0;            // KiB → MB
#endif
}

template<class T>
bool read_bin(const std::string& path, std::vector<T>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz % sizeof(T) != 0) return false;
    out.resize(static_cast<size_t>(sz / sizeof(T)));
    return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), sz));
}

static const char* get_arg(char** begin, char** end, const std::string& flag, const char* def=nullptr) {
    char** it = std::find(begin, end, flag);
    if (it != end && ++it != end) return *it;
    return def;
}

int main(int argc, char** argv) 
{
    std::cout << "Using Scalar type: " << type_name<Scalar>() << "\n";
    // Required args
    const char* train_x_path = get_arg(argv, argv+argc, std::string("--train_x"));
    const char* train_y_path = get_arg(argv, argv+argc, std::string("--train_y"));
    const char* test_x_path  = get_arg(argv, argv+argc, std::string("--test_x"));
    const char* test_y_path  = get_arg(argv, argv+argc, std::string("--test_y"));

    // Hyperparams / shapes
    const int   D      = std::atoi(get_arg(argv, argv+argc, std::string("--d"),      "784"));
    const int   K      = std::atoi(get_arg(argv, argv+argc, std::string("--k"),      "10"));
    const int   epochs = std::atoi(get_arg(argv, argv+argc, std::string("--epochs"), "10"));
    const int   bs     = std::atoi(get_arg(argv, argv+argc, std::string("--bs"),     "1024"));
    const float lr     = std::atof(get_arg(argv, argv+argc, std::string("--lr"),     "0.1"));
    const float lambda = std::atof(get_arg(argv, argv+argc, std::string("--lambda"), "1e-4"));
    const unsigned seed= std::atoi(get_arg(argv, argv+argc, std::string("--seed"),   "42"));

    if (!train_x_path || !train_y_path || !test_x_path || !test_y_path) 
    {
        std::cerr << "Missing --train_x/--train_y/--test_x/--test_y\n";
        return 2;
    }

    std::vector<float> Xtr_f32, Xte_f32;
    std::vector<int32_t> ytr_i32, yte_i32;

    if (!read_bin<float>(train_x_path, Xtr_f32)) { std::cerr<<"Failed "<<train_x_path<<"\n"; return 3; }
    if (!read_bin<int32_t>(train_y_path, ytr_i32)) { std::cerr<<"Failed "<<train_y_path<<"\n"; return 3; }
    if (!read_bin<float>(test_x_path, Xte_f32)) { std::cerr<<"Failed "<<test_x_path<<"\n"; return 3; }
    if (!read_bin<int32_t>(test_y_path, yte_i32)) { std::cerr<<"Failed "<<test_y_path<<"\n"; return 3; }

    std::vector<Scalar> Xtr(Xtr_f32.begin(), Xtr_f32.end());
    std::vector<Scalar> Xte(Xte_f32.begin(), Xte_f32.end());
    std::vector<int>    ytr(ytr_i32.begin(), ytr_i32.end());
    std::vector<int>    yte(yte_i32.begin(), yte_i32.end());

    const int Ntr = (int)(Xtr.size() / D);
    const int Nte = (int)(Xte.size() / D);
    if ((size_t)Ntr * (size_t)D != Xtr.size() || (int)ytr_i32.size() != Ntr ||
        (size_t)Nte * (size_t)D != Xte.size() || (int)yte_i32.size() != Nte) {
        std::cerr << "Shape mismatch; check D and file sizes.\n"; return 4;
    }

   // Model init
    SoftmaxSGD<Scalar> model(D, K, Scalar(lambda));
    model.init(seed);

    // Train (time just the epochs)
    double w0 = wall_now(), c0 = cpu_now();
    float last_train_loss = 0.f;
    for (int e=1; e<=epochs; ++e) 
    {
        last_train_loss = model.train_one_epoch(Xtr.data(), ytr.data(), Ntr, bs, lr, seed + (unsigned)e);
    }
    double c1 = cpu_now(), w1 = wall_now();

    // Eval
    float acc = model.eval_accuracy(Xte.data(), yte.data(), Nte);
    float ll  = model.eval_logloss(Xte.data(), yte.data(), Nte);

    // JSON one-liner
    std::cout.setf(std::ios::fixed); std::cout.precision(6);
    std::cout << "{"
              << "\"epochs\":"      << epochs
              << ",\"batch_size\":" << bs
              << ",\"lr\":"         << lr
              << ",\"lambda\":"     << lambda
              << ",\"wall_time_s\":"<< (w1 - w0)
              << ",\"cpu_time_s\":" << (c1 - c0)
              << ",\"peak_rss_mb\":"<< peak_rss_mb()
              << ",\"train_loss\":" << last_train_loss
              << ",\"test_acc\":"   << acc
              << ",\"test_logloss\":"<< ll
              << "}\n";
    return 0;
}
