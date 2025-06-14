
# Shallow Neural Networks – Simple Guide

This README explains how shallow (2-layer) neural networks work, both conceptually and in code.

## 1. Overview

In **logistic regression**, we compute:
```
z = W·X + b
a = sigmoid(z)
```

In a **neural network** with one hidden layer:
```
z1 = W1·X + b1
a1 = sigmoid(z1)
z2 = W2·a1 + b2
a2 = sigmoid(z2)
```

Where:
- `a1` = activations of hidden layer
- `a2` = final prediction

This is essentially a **stack of logistic regressions**.

---

## 2. Architecture

A shallow neural network contains:
- Input layer: `a0 = X`
- Hidden layer: `a1 = sigmoid(W1·X + b1)`
- Output layer: `a2 = sigmoid(W2·a1 + b2)`

We **do not count the input layer** when saying it's a 2-layer network.

---

## 3. Shapes and Dimensions

Assume:
- `X.shape = (3, m)` (3 features, m examples)
- 4 hidden neurons


### 📌 Why is `X.shape = (n_x, m)` and not `(m, n_x)`?

In deep learning, we use the convention:

> **Each column of `X` is a single training example.**

So if:
- `n_x = 3` (number of features)
- `m = 4` (number of examples)

Then:

```
X = [ [x₁⁽¹⁾  x₁⁽²⁾  x₁⁽³⁾  x₁⁽⁴⁾ ]
      [x₂⁽¹⁾  x₂⁽²⁾  x₂⁽³⁾  x₂⁽⁴⁾ ]
      [x₃⁽¹⁾  x₃⁽²⁾  x₃⁽³⁾  x₃⁽⁴⁾ ] ]
```

- `X.shape = (3, 4)` → (features, examples)

This layout allows us to compute **forward propagation** efficiently:

```python
Z = np.dot(W, X) + b
```

- `W.shape = (n_h, n_x)`
- `X.shape = (n_x, m)`
- Result: `Z.shape = (n_h, m)` → one result per neuron per example

---

💡 In contrast:
- scikit-learn uses `X.shape = (m, n_x)` (examples as rows)
- But neural networks use `X.shape = (n_x, m)` (examples as columns) for easier **vectorization**.


| Var |  Shape  |  
|:-----|:--------:|
| W1       | (4, 3)|
| b1       | (4, 1) |
| Z1       | (4, m) |
| A1       | (4, m) |
| W2       | (1, 4) |
| b2       | (1, 1) |
| Z2       | (1, m) |
| A2       | (1, m) |

---

## 4. Forward Propagation (Vectorized)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[x1], [x2], [x3]])  # shape (3, m)
W1 = np.random.randn(4, 3)
b1 = np.zeros((4, 1))
W2 = np.random.randn(1, 4)
b2 = np.zeros((1, 1))

Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
```

---

## 5. Mini Example

```python
X = np.array([[1], [0], [1]])  # input features (3, 1)

W1 = np.array([
    [0.2, 0.4, 0.1],
    [0.3, 0.5, 0.7],
    [0.6, 0.9, 0.8],
    [0.1, 0.2, 0.3]
])  # shape (4, 3)

b1 = np.zeros((4, 1))

W2 = np.array([[0.5, 0.6, 0.1, 0.2]])  # shape (1, 4)
b2 = np.array([[0.1]])

Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

print("Output:", A2)
```

---

## 6. Diagram Explanation: Neural Network Representation

The diagram breaks down forward propagation in a 2-layer neural network using 3 input features and 4 hidden neurons.

Each hidden neuron computes:

```
z[i] = W[i]^T · x + b[i]
a[i] = sigmoid(z[i])
```

- x = (3, 1)
- W1 = (4, 3)
- b1 = (4, 1)
- Z1 = W1 · x + b1 → (4, 1)
- A1 = sigmoid(Z1)

Each row of W1 is the transposed weights vector of one hidden neuron.

Final output:

```
z2 = W2 · A1 + b2
a2 = sigmoid(z2)
```

---

## 7. Forward Propagation Equations

Let:

- `x ∈ ℝⁿˣ×1` — input vector  
- `W[1] ∈ ℝⁿʰ×ⁿˣ` — weights for hidden layer  
- `b[1] ∈ ℝⁿʰ×1` — biases for hidden layer  
- `W[2] ∈ ℝ¹×ⁿʰ` — weights for output layer  
- `b[2] ∈ ℝ¹×1` — bias for output layer  

### Step-by-step:

1. **Hidden Layer Linear Activation**  
   `Z[1] = W[1] · x + b[1]`

2. **Hidden Layer Activation (Sigmoid)**  
   `A[1] = sigmoid(Z[1])`

3. **Output Layer Linear Activation**  
   `Z[2] = W[2] · A[1] + b[2]`

4. **Output Activation (Sigmoid)**  
   `A[2] = ŷ = sigmoid(Z[2])`

5. **Sigmoid Function**  
   `sigmoid(z) = 1 / (1 + e^(–z))`

---
