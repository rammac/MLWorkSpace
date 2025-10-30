import os, json, time, subprocess, tempfile, shutil
import numpy as np

# --- Fair single-thread baseline ---
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import resource  # for ru_maxrss on mac/Linux

def load_fashion_mnist():
    """
    Returns: (X_train, y_train, X_test, y_test) with X in float32 [N,784], y int64.
    Prefers OpenML via scikit-learn; falls back to torchvision if installed.
    """
    #try:
    from sklearn.datasets import fetch_openml
    #X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, cache=True, return_X_y=True)
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    # Conventional 60k/10k split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
    return X_tr, y_tr, X_te, y_te
    #except Exception as e1:
    #    try:
    #        import torch
    #        from torchvision import datasets, transforms
    #        tfm = transforms.Compose([transforms.ToTensor()])  # 0..1
    #        tr = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=tfm)
    #        te = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    #        Xtr = tr.data.numpy().astype(np.float32) / 255.0
    #        ytr = tr.targets.numpy().astype(np.int64)
    #        Xte = te.data.numpy().astype(np.float32) / 255.0
    #        yte = te.targets.numpy().astype(np.int64)
    #        Xtr = Xtr.reshape(len(Xtr), -1)
    #        Xte = Xte.reshape(len(Xte), -1)
    #        return Xtr, ytr, Xte, yte
    #    except Exception as e2:
    #        raise RuntimeError(
    #            "Could not load Fashion-MNIST. "
    #            "Install scikit-learn with OpenML or torchvision.\n"
    #            f"OpenML error: {e1}\nTorch error: {e2}"
    #        )

def standardize_train_test(Xtr, Xte, eps=1e-8):
    mu = Xtr.mean(axis=0, dtype=np.float64)
    sd = Xtr.std(axis=0, dtype=np.float64) + eps
    Xtrz = ((Xtr - mu) / sd).astype(np.float32, copy=False)
    Xtez = ((Xte - mu) / sd).astype(np.float32, copy=False)
    return Xtrz, Xtez

def run_sklearn_lbfgs(Xtrz, ytr, Xtez, ytez, max_iter=1000):
    start_wall = time.perf_counter()
    start_cpu  = time.process_time()
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",   # lbfgs or saga
        penalty="l2", C=1.0, tol=1e-3, max_iter=max_iter, n_jobs=1
    )
    model.fit(Xtrz, ytr)
    wall = time.perf_counter() - start_wall
    cpu  = time.process_time() - start_cpu
    # Metrics
    yprob = model.predict_proba(Xtez)
    acc   = accuracy_score(ytez, yprob.argmax(axis=1))
    ll    = log_loss(ytez, yprob, labels=np.arange(10))
    # Peak RSS in MB (max so far in this process)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if hasattr(ru, "ru_maxrss"):
        peak_mb = ru.ru_maxrss / (1024.0 if os.uname().sysname != "Darwin" else (1024.0*1024.0))
    else:
        peak_mb = float("nan")
    return {"impl":"sklearn_lbfgs","wall_time_s":wall,"cpu_time_s":cpu,
            "peak_rss_mb":peak_mb,"test_acc":acc,"test_logloss":ll}

def write_bin(path, arr, dtype):
    arr = np.asarray(arr, dtype=dtype, order="C")
    arr.tofile(path)

def run_cpp_trainer(Xtrz, ytr, Xtez, ytez, exe="./trainer_main",
                    epochs=10, bs=1024, lr=0.1, lamb=1.7e-5, seed=42):  # lanb=1e-4
    tmp = tempfile.mkdtemp(prefix="bench_fmnist_")
    try:
        tx = os.path.join(tmp, "Xtr.bin"); ty = os.path.join(tmp, "ytr.bin")
        vx = os.path.join(tmp, "Xte.bin"); vy = os.path.join(tmp, "yte.bin")
        write_bin(tx, Xtrz, np.float32)
        write_bin(ty, ytr,  np.int32)
        write_bin(vx, Xtez, np.float32)
        write_bin(vy, ytez, np.int32)

        D = Xtrz.shape[1]; K = 10
        cmd = [exe,
               "--train_x", tx, "--train_y", ty,
               "--test_x",  vx, "--test_y",  vy,
               "--d", str(D), "--k", str(K),
               "--epochs", str(epochs),
               "--bs", str(bs),
               "--lr", str(lr),
               "--lambda", str(lamb),
               "--seed", str(seed)]
        # Single-thread fairness (also for the child process)
        env = os.environ.copy()
        for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
            env[k] = "1"

        p = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        line = p.stdout.strip().splitlines()[-1]
        res = json.loads(line)
        res["impl"] = "cpp_sgd_fused"
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    print("Loading Fashion-MNIST …")
    Xtr, ytr, Xte, yte = load_fashion_mnist()
    # Flatten if needed (OpenML already 784-wide)
    if Xtr.ndim == 3:
        Xtr = Xtr.reshape(len(Xtr), -1)
        Xte = Xte.reshape(len(Xte), -1)
    Xtrz, Xtez = standardize_train_test(Xtr, Xte)

    # Run sklearn LBFGS (multinomial)
    lb = run_sklearn_lbfgs(Xtrz, ytr, Xtez, yte, max_iter=1500)
    print("sklearn:", lb)

    # Run C++ fused SGD (multinomial softmax)
    cpp = run_cpp_trainer(Xtrz, ytr, Xtez, yte,
                          exe="./trainer_main",
                          epochs=10, bs=1024, lr=0.1, lamb=1e-4, seed=42)
    print("c++    :", cpp)

    # Tiny summary
    print("\nSummary (single run):")
    for k in ["impl","wall_time_s","cpu_time_s","peak_rss_mb","test_acc","test_logloss"]:
        print(f"{k:>14s} | {lb.get(k,'')}  | {cpp.get(k,'')}")

if __name__ == "__main__":
    main()
