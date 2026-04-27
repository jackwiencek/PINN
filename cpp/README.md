# PINN C++ — libtorch port

C++ port of `src/` using the PyTorch C++ API (libtorch). Mirrors `model.py`, `physics.py`, `sampling.py`, `train.py`, `run_experiments.py`. Outputs perf CSVs to `c_logs/` and model state-dicts to `c_runs/`.

## Prerequisites

- CMake ≥ 3.18
- MSVC 2019/2022 (C++17) or equivalent
- libtorch (CPU build, see below)

## 1. Download libtorch

**CPU build (Windows):**

```
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip
```

Unzip to a directory, e.g. `C:\libs\libtorch`.

> For CUDA: pick the matching CUDA build from https://pytorch.org/get-started/locally/ — select C++/LibTorch.

## 2. Configure

From the repo root (`c:\PINN`):

```
cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="C:/libs/libtorch"
```

## 3. Build

```
cmake --build cpp/build --config Release
```

DLLs are auto-copied next to the executables by the post-build step.

## 4. Run all 8 experiments

```
cpp/build/Release/pinn_experiments.exe
```

Produces:
- `c_logs/perf_<run_name>_<HHMMSS>.csv` — per-epoch timing (columns: `epoch,phase,wall_time_s,epoch_time_s`)
- `c_logs/evals.csv` — error metrics (columns: `name,timestamp,run_id,max_abs_error,l2_error`)
- `c_logs/experiment_manifest.json` — run_id + run_path per experiment
- `c_runs/run_<run_id>.pt` — saved model state-dicts (C++ only, not Python-loadable)

## 5. Eval CLI

```
cpp/build/Release/pinn_eval.exe c_runs/run_<id>.pt --name <label> [--grid 100] [--pde HeatEquation1D|ViscousBurgers1D]
```

## CSV schema parity with Python logs

| File | Header | Notes |
|------|--------|-------|
| `c_logs/perf_*.csv` | `epoch,phase,wall_time_s,epoch_time_s` | Matches `logs/perf_*.csv` |
| `c_logs/evals.csv` | `name,timestamp,run_id,max_abs_error,l2_error` | Matches `logs/evals.csv` |

## Notes

- **CosineAnnealingLR**: implemented inline (`lr = 0.5 * base_lr * (1 + cos(π·epoch/T_max))`) — libtorch scheduler API may not be available in all versions.
- **L-BFGS**: strong-Wolfe line search used; may be slower than Python for first few steps due to graph overhead.
- **Checkpointing**: not implemented — `run_experiments.py` also uses `use_checkpoint=False`.
- **Model .pt files**: saved via `torch::save(model, path)` — C++ serializer, not interoperable with Python `torch.save` bundles.
