import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print("[PART 1] 시그모이드 함수 수치 테스트")

s_0 = sigmoid(0)
print(f"입력값 0  => 압축된 확률: {s_0:5f} (50%)")

s_100 = sigmoid(100)
print(f"입력값 100 => 압축된 확률: {s_100:5f} (약 100%)")

s_m100 = sigmoid(-100)
print(f"입력값 -100  => 압축된 확률: {s_m100:5f} (약 0%)")

z_values = np.linspace(-10, 10, 200)

probabilities = sigmoid(z_values)

plt.figure(figsize=(10, 6))

plt.plot(z_values, probabilities, color='red', linewidth=3, label='Sigmoid Curve')

plt.axhline(y=1.0, color='gray', linestyle='--',alpha=0.5)
plt.axhline(y=0.0, color='gray', linestyle='--',alpha=0.5)

plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')

plt.axvline(x=0.0, color='gray', linestyle='--',alpha=0.5)

plt.scatter(0, 0.5, color='blue', s=100, zorder=5, label='sigmoid(0) = 0.5')

plt.title('Sigmoid Function (Magic Compressor)')
plt.xlabel('Raw Score (z) - from AI model')
plt.ylabel('Probability (0.0 ~ 1.0) - output')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

plt.show()