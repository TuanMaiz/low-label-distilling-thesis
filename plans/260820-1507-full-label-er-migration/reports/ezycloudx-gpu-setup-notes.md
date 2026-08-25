# EzyCloudX GPU setup notes (historical)

These notes describe a retired Phase-05 screening run. Its execution plan and
runner were deleted during the full-label migration and are available only in
Git history. Do not execute the old commands on the migration branch; Phase 7
will replace this file with commands verified against
`scripts/run_full_label_experiments.sh`.

The reusable VM prerequisite was a CUDA-visible Linux environment with a C/C++
compiler and the Python development headers required by the chosen runtime.

Verify the runtime before training:

```bash
gcc --version
g++ --version
python3 -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())'
ldconfig -p | grep libcuda
```

## If Triton reports a missing C compiler

Install the compiler and Python headers, then export the compiler paths:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.10-dev
export CC=gcc
export CXX=g++
```

The Python development package is needed because Triton may compile a CUDA
helper against `/usr/include/python3.10/Python.h` during generation or
validation. Check CUDA driver visibility with:

```bash
ldconfig -p | grep libcuda
```

Output packaging and handoff commands are intentionally omitted because the old
artifact grammar is incompatible with the sealed full-label experiment.
