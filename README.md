# Monty Hall Problem HIP Simulation

A high-performance GPU simulation of the Monty Hall problem using AMD ROCm HIP and hipRAND.

## Overview

This program simulates the classic Monty Hall problem, allowing you to choose between "stay" and "switch" strategies and run millions or billions of iterations efficiently on your GPU. It uses HIP for portability across AMD and NVIDIA GPUs.

## Features

- Fast GPU simulation using HIP and hipRAND
- Supports both "stay" and "switch" strategies
- Configurable number of iterations via command-line
- Reports win/loss rates and total runtime
- Robust error handling and resource management

## Requirements

- AMD ROCm stack (tested with ROCm 6.4+)
- hipcc compiler
- hipRAND library
- Linux (recommended)
- Compatible AMD or NVIDIA GPUs

## Build Instructions

1. **Install ROCm and hipRAND**  
   Follow [ROCm installation instructions](https://rocm.docs.amd.com/en/latest/) for your system.

2. **Clone the repository**
   ```sh
   git clone https://github.com/Avicted/monty_hall_problem_hip.git
   cd monty_hall_problem_hip
   ```

3. **Build the project**
   ```sh
   ./build.sh
   ```

   This will compile the code using `hipcc`, producing `build/monty_hall_problem_hip.out`.

## Usage

Run the simulation with optional arguments:

```sh
./build/monty_hall_problem_hip.out [--stay | --switch] [--iterations=N]
```

- `--stay`    Use the "stay" strategy (default is "switch")
- `--switch`   Use the "switch" strategy
- `--iterations=N` Run N iterations (default: 100,000,000)

**Example:**
```sh
./build/monty_hall_problem_hip.out --switch --iterations=10000000
```

## Output

The program prints device info, memory usage, kernel configuration, and simulation results:

```
Monty Hall Problem:
    HIP Device Count: 1
    Device 0: AMD Radeon RX 6900 XT
        Compute Capability: ------------ = 10.3
        Total Global Memory: ----------- = 17163091968
        Shared Memory per Block: ------- = 65536
        Registers per Block: ----------- = 65536
        Warp Size: --------------------- = 32
        Max Threads per Block: --------- = 1024
        Max Threads Dimension: --------- = (1024, 1024, 1024)
        Max Grid Size: ----------------- = (2147483647, 65536, 65536)
        Clock Rate: -------------------- = 2660000
        Total Constant Memory: --------- = 2147483647
        Multiprocessor Count: ---------- = 40
        L2 Cache Size: ----------------- = 4194304
        Max Threads per Multiprocessor:  = 2048
        Unified Addressing: ------------ = 0
        Memory Clock Rate: ------------- = 1000000
        Memory Bus Width: -------------- = 256
        Peak Memory Bandwidth: --------- = 64.000000

====================================================================
    Using GPU 0
    Total GPU memory allocated: 4.47 GB
    threadBlocks:               {97657, 1, 1} blocks.
    threadsInBlock:             1024 threads.
    Total number of threads:    100000768
    deviceCount:                1

Monty Hall Problem Results:
   Strategy:            Switch
   Total Iterations:    100000000
   Wins:                66661336
   Losses:              33338664
   Win Rate:            66.661%
   Loss Rate:           33.339%

Total runtime: 687.55 ms
```

## Troubleshooting

- **Out of Memory:**  
  If you request more iterations than your GPU memory can handle, the program will warn you and may fail. Reduce the number of iterations.
- **Grid Size Exceeded:**  
  If you request more blocks than your GPU supports, the program will warn you and exit. Reduce the number of iterations or increase `threadsPerBlock` if possible.
- **No HIP Device Found:**  
  Ensure your system has a supported AMD or NVIDIA GPU and ROCm is installed correctly.

## License

MIT License. See [LICENSE](LICENSE) for details.
