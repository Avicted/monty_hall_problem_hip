import matplotlib.pyplot as plt
import numpy as np
import time

def simulate(strategy, iterations, door_count):
    wins = []
    win_count = 0
    start = time.time()
    for i in range(1, iterations + 1):
        car = np.random.randint(door_count)
        choice = np.random.randint(door_count)
        if strategy == 'stay':
            win = (choice == car)
        else:
            win = (choice != car)
        win_count += win
        wins.append(win_count / i * 100)
    runtime_ms = (time.time() - start) * 1000
    return wins, win_count, runtime_ms

def print_summary(strategy, win_count, iterations, runtime_ms):
    losses = iterations - win_count
    winrate = win_count / iterations * 100
    lossrate = losses / iterations * 100
    print(f"{strategy.capitalize()} Strategy:")
    print(f"  Wins:      {win_count}")
    print(f"  Losses:    {losses}")
    print(f"  Win Rate:  {winrate:.2f}%")
    print(f"  Loss Rate: {lossrate:.2f}%")
    print(f"  Runtime:   {runtime_ms:.2f} ms\n")

def main():
    only_run_simulation = False
    np.random.seed(42)
    iterations = 1_000
    door_count = 3
    skip = 10

    stay_win_rate, stay_wins, stay_runtime = simulate('stay', iterations, door_count)
    switch_win_rate, switch_wins, switch_runtime = simulate('switch', iterations, door_count)

    print_summary("stay", stay_wins, iterations, stay_runtime)
    print_summary("switch", switch_wins, iterations, switch_runtime)

    total_runtime = stay_runtime + switch_runtime
    print(f"Total simulation runtime: {total_runtime:.2f} ms")

    if not only_run_simulation:
        plt.figure(figsize=(16, 9))
        x = np.arange(skip + 1, iterations + 1)
        plt.axhline(33.33, color='red', linestyle='--', alpha=0.5)
        plt.axhline(66.67, color='green', linestyle='--', alpha=0.5)
        plt.plot(x, stay_win_rate[skip:], label='Stay', color='red')
        plt.plot(x, switch_win_rate[skip:], label='Switch', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Win Rate (%)')
        plt.title('Monty Hall: Win Rate Over Iterations (3 Doors)')
        plt.legend()
        plt.xlim(0, iterations)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('monty_hall_winrate_over_time_3doors.svg')
        plt.show()

if __name__ == "__main__":
    main()
