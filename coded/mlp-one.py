# One input and one target
x = 2.0
y = 20.0

# Initial parameters
w = 0.0
b = 0.0

# Learning rate
learning_rate = 0.05

# Number of training epochs
epochs = 10

for epoch in range(1, epochs + 1):
    # Forward pass
    y_hat = w * x + b

    # MSE loss for one sample
    loss = (y - y_hat) ** 2

    # Gradients
    dw = 2 * (y_hat - y) * x
    db = 2 * (y_hat - y)

    # Save values before update
    w_before = w
    b_before = b

    # Gradient descent update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    print(f"Epoch {epoch}")
    print(f"  w before   : {w_before:.6f}")
    print(f"  b before   : {b_before:.6f}")
    print(f"  prediction : {y_hat:.6f}")
    print(f"  loss       : {loss:.6f}")
    print(f"  dw         : {dw:.6f}")
    print(f"  db         : {db:.6f}")
    print(f"  w after    : {w:.6f}")
    print(f"  b after    : {b:.6f}")
    print("-" * 35)

# Prediction after all 10 updates
final_prediction = w * x + b

print("Final result")
print(f"w = {w:.6f}")
print(f"b = {b:.6f}")
print(f"Prediction for x = {x}: {final_prediction:.6f}")
print(f"Target: {y}")
