x = 2
w = 1
y = x*10

for epoch in range(10):
    y_hat = w*x
    loss = abs(y - y_hat)
    if loss>0:
        w = w+1
    print(f"EPOCH {epoch+1} ::: LOSS {loss}")

print(w*x)