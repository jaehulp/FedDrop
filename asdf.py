import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.linspace(0, 10, 100)
y_mean = np.sin(x)
y_std = 0.2 + 0.2 * np.abs(np.cos(x))  # Example: std changes with x

# Plot mean line
plt.plot(x, y_mean, label='Mean')

# Fill between (mean-std) and (mean+std)
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Std Dev')

# Add labels and legend
plt.title('Line Plot with Std Deviation Shading')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.savefig('./shit.png')