import numpy as np
import matplotlib.pyplot as plt

random_cov = np.load("coverage_random.npy")
trained_cov = np.load("coverage_trained.npy")

plt.figure(figsize=(8, 5))
plt.plot(random_cov, label="Random Policy", linestyle="--", color="red")
plt.plot(trained_cov, label="Trained Policy", color="green")

plt.xlabel("Time step")
plt.ylabel("Coverage")
plt.title("Random vs Trained Policy (Coverage Comparison)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
