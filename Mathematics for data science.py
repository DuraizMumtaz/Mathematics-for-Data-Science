############### STU1208025 ###########
def gaussian_elimination(A, b):
    n = len(A)

    for i in range(n):
        # Pivot
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Zero pivot encountered!")

        for j in range(i, n):
            A[i][j] /= pivot
        b[i] /= pivot

        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))

    return x

A = [
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
]

b = [100, 100, 0, 0]

solution = gaussian_elimination(A, b)

print("Solution:")
for i, x in enumerate(solution, start=1):
    print(f"x{i} = {x:.4f}")

# ------------------------------------------------
# STUDENT NO : SSTU1208025
# -----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_1samp

mu = 0
sigma = 2   # standard deviation

prob_error_gt_3 = 1 - norm.cdf(3, loc=mu, scale=sigma)
print("Probability that error > 3°C:", prob_error_gt_3)

x = np.linspace(0, 10, 400)
T = 100 * np.sin(np.pi * x / 10)

plt.figure()
plt.plot(x, T)
plt.xlabel("x")
plt.ylabel("T(x)")
plt.title("Temperature Distribution: T(x) = 100 sin(πx/10)")
plt.grid(True)
plt.show()

data = np.array([99, 100, 98, 101, 97, 99, 100, 98])

t_stat, p_val = ttest_1samp(data, 100)

print("Sample Mean:", np.mean(data))
print("t-statistic:", t_stat)
print("p-value:", p_val)

plt.figure()
plt.hist(data)
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Histogram of Temperature Measurements")
plt.grid(True)
plt.show()

## ---------------------------------------
## STU1208025
## ---------------------------------------
import numpy as np
import matplotlib.pyplot as plt
def temperature(x):
    return 100 * np.sin((np.pi * x) / 10)
def temperature_rate(x):
    return 100 * (np.pi / 10) * np.cos((np.pi * x) / 10)
x_range = np.linspace(0, 10, 400)
temp_values = temperature(x_range)
rate_values = temperature_rate(x_range)
max_rate_value = np.max(np.abs(rate_values))
max_rate_positions = x_range[np.isclose(np.abs(rate_values), max_rate_value)]

print(f"Greatest temperature change = {max_rate_value:.4f} °C/unit.")
print(f"Occurs at x-position(s): {np.unique(np.round(max_rate_positions, 6))}")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_range, temp_values)
plt.title("Temperature Along Plate")
plt.xlabel("x-position")
plt.ylabel("Temperature (°C)")

plt.subplot(1, 2, 2)
plt.plot(x_range, rate_values)
plt.title("Rate of Temperature Change")
plt.xlabel("x-position")
plt.ylabel("dT/dx (°C per unit)")

plt.tight_layout()
plt.show()

## ------------------------------------------
## STU1208025
## ------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
mean_err = 0
std_err = 2
limit = 3
p_exceeds = 1 - norm.cdf(limit, loc=mean_err, scale=std_err)
print(f"Probability that the temperature error exceeds {limit}°C: {p_exceeds:.6f}")
x_vals = np.linspace(-10, 10, 500)
density_vals = norm.pdf(x_vals, loc=mean_err, scale=std_err)
plt.figure(figsize=(7, 4))
plt.plot(x_vals, density_vals, color='blue', linewidth=2, label='Normal PDF N(0,2)')
plt.fill_between(x_vals, density_vals, where=(x_vals > limit),
                 color='orange', alpha=0.4, label='P(error > 3°C)')
plt.axvline(x=limit, color='black', linestyle='--')
plt.text(limit+0.2, max(density_vals)*0.8, f"x={limit}", fontsize=10)
plt.xlabel("Error value (°C)")
plt.ylabel("Probability Density")
plt.title("Error Distribution with Region Where Error > 3°C Highlighted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## ------------------------------------------
## STU1208025
## ------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

temps = np.array([99, 100, 98, 101, 97, 99, 100, 98])
mean_val = np.mean(temps)
plt.figure(figsize=(7, 5))
sns.violinplot(data=temps, inner=None, color="skyblue")
sns.stripplot(data=temps, color='black', size=8, jitter=True)
plt.scatter(0, mean_val, color='red', s=100, zorder=10, label=f'Mean = {mean_val:.2f}°C')
plt.title("Temperature Distribution with Violin Plot and Sample Points")
plt.ylabel("Temperature (°C)")
plt.xticks([])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()