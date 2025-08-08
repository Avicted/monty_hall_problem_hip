/*
 * Monty Hall Problem HIP Simulation
 * Copyright (c) 2025 Victor Anderssén
 * MIT License
 * See LICENSE file for details.
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#include "utils.hip"

typedef enum
{
    STRATEGY_STAY = 0,
    STRATEGY_SWITCH,

    STRATEGY_COUNT
} Strategy;

static const char *
    strategy_names[STRATEGY_COUNT] = {
        "Stay",
        "Switch",
};

typedef enum
{
    DOOR_CAR = 0,
    DOOR_GOAT,

    DOOR_COUNT
} DoorContent;

unsigned long long int wins = 0;
unsigned long long int losses = 0;

// Constants -------------------------------------------------------------------
#define RANDOM_SEED 123456789
#define MIN_DOORS 3
#define MAX_DOORS 128

Strategy strategy = STRATEGY_SWITCH;
unsigned long long int iterations = 100'000'000;

int door_count = 3; // Default to 3 doors, can be changed for N-door Monty Hall
// -----------------------------------------------------------------------------

__global__ void
setup_kernel(hiprandState *state, unsigned long long seed, int iterations)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < iterations)
    {
        hiprand_init(seed, id, 0, &state[id]);
    }
}

__global__ void
monty_hall_problem_kernel(
    unsigned long long int *wins,
    unsigned long long int *losses,
    hiprandState *state,
    Strategy strategy,
    unsigned long long int iterations,
    int d_door_count)
{
    unsigned long long int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= iterations)
    {
        return;
    }

    hiprandState localState = state[thread_id];

    uint8_t doors[MAX_DOORS];
    bool opened[MAX_DOORS];

    // Initialize all doors to goat
    for (uint8_t i = 0; i < d_door_count; ++i)
    {
        doors[i] = DOOR_GOAT;
        opened[i] = false;
    }

    // Place the prize (car) randomly
    uint8_t prize_door = hiprand(&localState) % d_door_count;
    doors[prize_door] = DOOR_CAR;

    // Contestant picks a random door
    uint8_t chosen_door = hiprand(&localState) % d_door_count;
    opened[chosen_door] = true;

    // Host opens N-2 doors that are goats and not chosen
    uint8_t doors_to_open = d_door_count - 2;
    while (doors_to_open > 0)
    {
        uint8_t door = hiprand(&localState) % d_door_count;
        if (!opened[door] && doors[door] == DOOR_GOAT && door != chosen_door)
        {
            opened[door] = true;
            doors_to_open--;
        }
    }

    // If switching, pick the one unopened door that isn't the original choice
    if (strategy == STRATEGY_SWITCH)
    {
        for (uint8_t i = 0; i < d_door_count; ++i)
        {
            if (!opened[i] && i != chosen_door)
            {
                chosen_door = i;
                break;
            }
        }
    }

    // Win/Loss record
    if (doors[chosen_door] == DOOR_CAR)
    {
        atomicAdd(wins, 1);
    }
    else
    {
        atomicAdd(losses, 1);
    }

    // Save RNG state
    state[thread_id] = localState;
}

static void
parse_input_args(
    int argc,
    char **argv,
    Strategy *strategy,
    unsigned long long int *iterations,
    int *door_count)
{
    if (argc == 2 &&
        (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
    {
        fprintf(stderr, "Usage: %s [--stay | --switch] [--iterations=N] [--doors=N]\n", argv[0]);
        fprintf(stderr, "Default strategy is --switch, iterations is %llu, and doors is %d.\n",
                *iterations, *door_count);
        exit(0);
    }

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--stay") == 0)
        {
            *strategy = STRATEGY_STAY;
        }
        else if (strcmp(argv[i], "--switch") == 0)
        {
            *strategy = STRATEGY_SWITCH;
        }
        else if (strncmp(argv[i], "--iterations=", 13) == 0)
        {
            const char *iter_str = argv[i] + 13;
            char *endptr = NULL;
            errno = 0;

            if (iter_str[0] == '-')
            {
                fprintf(stderr, "Invalid value for --iterations ('%s'), must be a positive integer. Using default (%llu).\n",
                        iter_str, *iterations);
                continue;
            }

            unsigned long long int val = strtoull(iter_str, &endptr, 10);

            // Check for non-numeric, zero, or overflow
            if (endptr == iter_str || *endptr != '\0' || errno == ERANGE || val == 0)
            {
                fprintf(stderr, "Invalid value for --iterations ('%s'), must be a positive integer in range. Using default (%llu).\n",
                        iter_str, *iterations);
                continue;
            }
            else
            {
                *iterations = val;
            }
        }
        else if (strncmp(argv[i], "--doors=", 8) == 0)
        {
            const char *doors_str = argv[i] + 8;
            char *endptr = NULL;
            errno = 0;

            if (doors_str[0] == '-')
            {
                fprintf(stderr, "Invalid value for --doors ('%s'), must be a positive integer. Using default (%d).\n",
                        doors_str, *door_count);
                continue;
            }

            int val = strtol(doors_str, &endptr, 10);

            // Check for non-numeric, out-of-range, or overflow
            if (endptr == doors_str || *endptr != '\0' || errno == ERANGE || val < MIN_DOORS || val > MAX_DOORS)
            {
                fprintf(stderr, "Invalid value for --doors ('%s'), must be an integer >= %d and <= %d. Using default (%d).\n",
                        doors_str, MIN_DOORS, MAX_DOORS, *door_count);
                continue;
            }
            else
            {
                *door_count = val;
            }
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--stay | --switch] [--iterations=N] [--doors=N]\n", argv[0]);
            fprintf(stderr, "Default strategy is --switch, iterations is %llu, and doors is %d.\n",
                    *iterations, *door_count);
            exit(1);
        }
    }
}
int main(int argc, char **argv)
{
    printf("Monty Hall Problem:\n");

    parse_input_args(argc, argv, &strategy, &iterations, &door_count);

    if (door_count < MIN_DOORS || door_count > MAX_DOORS)
    {
        fprintf(stderr, "Error: door_count must be between %d and %d.\n", MIN_DOORS, MAX_DOORS);
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int deviceCount = get_hip_devices();
    if (deviceCount == -1 || deviceCount == 0)
    {
        fprintf(stderr, "ERROR: No HIP devices (supported GPU's) found!\n");
        return 1;
    }

    // Kernel specification
    int maxThreadsPerBlock;
    hipDeviceProp_t prop;
    HIP_ERRCHK(hipGetDeviceProperties(&prop, 0));
    maxThreadsPerBlock = prop.maxThreadsPerBlock;

    // Auto-tune threadsPerBlock: use the largest power-of-two <= maxThreadsPerBlock, up to 1024
    int threadsPerBlock = 256;
    if (maxThreadsPerBlock >= 1024)
    {
        threadsPerBlock = 1024;
    }
    else if (maxThreadsPerBlock >= 512)
    {
        threadsPerBlock = 512;
    }
    else if (maxThreadsPerBlock >= 256)
    {
        threadsPerBlock = 256;
    }
    else
    {
        threadsPerBlock = maxThreadsPerBlock; // fallback for small devices
    }

    unsigned long long int blocks = (iterations + threadsPerBlock - 1) / threadsPerBlock;

    if (blocks > static_cast<unsigned long long int>(prop.maxGridSize[0]))
    {
        fprintf(stderr,
                "Warning: Requested number of blocks (%llu) exceeds device's maxGridSize[0] (%d).\n"
                "Kernel launch may fail or produce incorrect results.\n",
                blocks, prop.maxGridSize[0]);

        return 1;
    }

    dim3 threadsInBlock(threadsPerBlock);
    dim3 threadBlocks(static_cast<unsigned int>(blocks));

    const long int number_of_threads = blocks * threadsPerBlock;

    HIP_ERRCHK(hipSetDevice(0));
    printf("====================================================================\n");
    printf("    Using GPU 0\n");

    unsigned long long int *d_wins;
    unsigned long long int *d_losses;
    size_t wins_array_bytes = sizeof(unsigned long long int);
    size_t losses_array_bytes = sizeof(unsigned long long int);
    size_t hiprand_states_bytes = iterations * sizeof(hiprandState);

    size_t total_allocated_bytes = wins_array_bytes + losses_array_bytes + hiprand_states_bytes;
    double total_allocated_gb = total_allocated_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("    Total GPU memory allocated: %.2f GB\n", total_allocated_gb);

    size_t free_mem = 0, total_mem = 0;
    HIP_ERRCHK(hipMemGetInfo(&free_mem, &total_mem));
    if (total_allocated_bytes > free_mem)
    {
        fprintf(stderr, "Warning: Requested memory (%.2f GB) exceeds available GPU memory (%.2f GB).\n",
                total_allocated_gb, free_mem / (1024.0 * 1024.0 * 1024.0));
    }

    // Allocate device memory
    HIP_ERRCHK(hipMalloc((void **)&d_wins, wins_array_bytes));
    HIP_ERRCHK(hipMalloc((void **)&d_losses, losses_array_bytes));

    // Copy initial values to device
    HIP_ERRCHK(hipMemcpy(d_wins, &wins, wins_array_bytes, hipMemcpyHostToDevice));
    HIP_ERRCHK(hipMemcpy(d_losses, &losses, losses_array_bytes, hipMemcpyHostToDevice));

    printf("    threadBlocks:\t\t{%d, %d, %d} blocks.\n    threadsInBlock:\t\t%d threads.\n",
           threadBlocks.x, threadBlocks.y, threadBlocks.z, threadsInBlock.x * threadsInBlock.y * threadsInBlock.z);

    printf("    Total number of threads:\t%ld\n", number_of_threads);
    printf("    deviceCount:\t\t%d\n", deviceCount);

    hiprandState *d_state;
    HIP_ERRCHK(hipMalloc(&d_state, iterations * sizeof(hiprandState)));

    // Run kernels
    setup_kernel<<<threadBlocks, threadsInBlock>>>(d_state, RANDOM_SEED, iterations);
    monty_hall_problem_kernel<<<threadBlocks, threadsInBlock>>>(d_wins, d_losses, d_state, strategy, iterations, door_count);

    HIP_ERRCHK(hipGetLastError());
    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy results back to host
    HIP_ERRCHK(hipMemcpy(&wins, d_wins, wins_array_bytes, hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipMemcpy(&losses, d_losses, losses_array_bytes, hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipDeviceSynchronize());

    // Free device memory
    HIP_ERRCHK(hipFree(d_wins));
    HIP_ERRCHK(hipFree(d_losses));
    HIP_ERRCHK(hipFree(d_state));

    printf("\nMonty Hall Problem Results:\n");
    printf("   Strategy:\t\t%s\n", strategy_names[strategy]);
    printf("   Doors:\t\t%d\n", door_count);
    printf("   Total Iterations:\t%llu\n", iterations);
    printf("   Wins:\t\t%llu\n", wins);
    printf("   Losses:\t\t%llu\n", losses);

    printf("   Win Rate:\t\t%.3f%%\n", static_cast<double>((wins * 100.0) / iterations));
    printf("   Loss Rate:\t\t%.3f%%\n", static_cast<double>((losses * 100.0) / iterations));

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    printf("\nTotal runtime: %.2f ms\n", elapsed_ms);

    return 0;
}
