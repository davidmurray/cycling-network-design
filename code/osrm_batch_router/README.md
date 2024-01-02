# OSRM batch router
A C++ utility to perform routing calculations in batch using OSRM.
Provided with a CSV file with all od routes, it will output the result of the routes as one big JSON object.

Internally, it uses LibOSRM from the OSRM package. We recommend installing OSRM by [building from source](https://github.com/Project-OSRM/osrm-backend#building-from-source).

There are two versions: osrm_batch_router and osrm_batch_router_parallel. The former does routing calculations one at a time, while the second use does all calculations in parallel by using all available processor cores. The parallel version uses [libdispatch](https://github.com/apple/swift-corelibs-libdispatch), which was developed by Apple for macOS, but there is a Linux port available, so it should technically work there too.

## Compilation instructions

# Prerequisites
1. A C++ compiler such as `clang` or `gcc`
2. [LibOSRM](https://github.com/Project-OSRM/osrm-backend#building-from-source) library
3. [fmt](https://github.com/fmtlib/fmt) library
4. `cmake` build utility (see [installation instructions](https://cgold.readthedocs.io/en/latest/first-step/installation.html))

# Compilation

From the `osrm_batch_router` folder, run:
```bash
mkdir -p build
cd build
cmake ..
make
```

Executables will be create in the `build` folder. These are not used directly, but they are used internally by the Python code in `genetic_algorithm.py`