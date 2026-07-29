import math
import random


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a):
    # If a = sigmoid(z), then sigmoid'(z) = a(1-a)
    return a * (1.0 - a)


# XOR dataset
training_data = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]

random.seed(42)

# Input -> hidden weights
w11 = random.uniform(-1, 1)  # x1 -> h1
w21 = random.uniform(-1, 1)  # x2 -> h1

w12 = random.uniform(-1, 1)  # x1 -> h2
w22 = random.uniform(-1, 1)  # x2 -> h2

# Hidden -> output weights
w31 = random.uniform(-1, 1)  # h1 -> y_hat
w32 = random.uniform(-1, 1)  # h2 -> y_hat

# Biases
b1 = random.uniform(-1, 1)   # h1 bias
b2 = random.uniform(-1, 1)   # h2 bias
b3 = random.uniform(-1, 1)   # output bias

learning_rate = 0.5
epochs = 20_000

for epoch in range(1, epochs + 1):
    total_loss = 0.0

    for x1, x2, y in training_data:

        # --------------------
        # Forward propagation
        # --------------------

        z1 = (w11 * x1) + (w21 * x2) + b1
        h1 = sigmoid(z1)

        z2 = (w12 * x1) + (w22 * x2) + b2
        h2 = sigmoid(z2)

        z3 = (w31 * h1) + (w32 * h2) + b3
        y_hat = sigmoid(z3)

        # L = 1/2 * (y - y_hat)^2
        loss = 0.5 * (y - y_hat) ** 2
        total_loss += loss

        # --------------------
        # Backpropagation
        # --------------------

        # Output-layer delta:
        #
        # dL/dz3
        # = dL/dy_hat * dy_hat/dz3
        # = (y_hat - y) * y_hat(1-y_hat)

        delta3 = (y_hat - y) * sigmoid_derivative(y_hat)

        # Output-layer gradients
        dL_dw31 = delta3 * h1
        dL_dw32 = delta3 * h2
        dL_db3 = delta3

        # Hidden-layer deltas
        #
        # dL/dz1 = delta3 * w31 * h1(1-h1)
        # dL/dz2 = delta3 * w32 * h2(1-h2)

        delta1 = delta3 * w31 * sigmoid_derivative(h1)
        delta2 = delta3 * w32 * sigmoid_derivative(h2)

        # Hidden-layer gradients
        dL_dw11 = delta1 * x1
        dL_dw21 = delta1 * x2
        dL_db1 = delta1

        dL_dw12 = delta2 * x1
        dL_dw22 = delta2 * x2
        dL_db2 = delta2

        # --------------------
        # Gradient descent
        # --------------------

        w11 = w11 - learning_rate * dL_dw11
        w21 = w21 - learning_rate * dL_dw21

        w12 = w12 - learning_rate * dL_dw12
        w22 = w22 - learning_rate * dL_dw22

        w31 = w31 - learning_rate * dL_dw31
        w32 = w32 - learning_rate * dL_dw32

        b1 = b1 - learning_rate * dL_db1
        b2 = b2 - learning_rate * dL_db2
        b3 = b3 - learning_rate * dL_db3

    if epoch == 1 or epoch % 2000 == 0:
        mean_loss = total_loss / len(training_data)
        print(f"Epoch {epoch:5d} | Loss: {mean_loss:.8f}")


print("\nPredictions\n")

for x1, x2, y in training_data:
    z1 = (w11 * x1) + (w21 * x2) + b1
    h1 = sigmoid(z1)

    z2 = (w12 * x1) + (w22 * x2) + b2
    h2 = sigmoid(z2)

    z3 = (w31 * h1) + (w32 * h2) + b3
    y_hat = sigmoid(z3)

    predicted_class = 1 if y_hat >= 0.5 else 0

    print(
        f"{int(x1)} XOR {int(x2)} "
        f"= {y_hat:.6f} "
        f"-> {predicted_class} "
        f"(target: {int(y)})"
    )


print("\nFinal parameters\n")

print("w11 =", w11)
print("w21 =", w21)
print("w12 =", w12)
print("w22 =", w22)
print("w31 =", w31)
print("w32 =", w32)

print("b1  =", b1)
print("b2  =", b2)
print("b3  =", b3)
