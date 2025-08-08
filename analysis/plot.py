import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
iterations = 1000
door_count = 3
skip = 10

def simulate(strategy, iterations, door_count):
    wins = []
    win_count = 0
    for i in range(1, iterations + 1):
        car = np.random.randint(door_count)
        choice = np.random.randint(door_count)
        if strategy == 'stay':
            win = (choice == car)
        else:
            win = (choice != car)
        win_count += win
        wins.append(win_count / i * 100)
    return wins

stay_win_rate = simulate('stay', iterations, door_count)
switch_win_rate = simulate('switch', iterations, door_count)

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
