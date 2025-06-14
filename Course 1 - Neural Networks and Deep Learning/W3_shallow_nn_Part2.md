
# 🧠 Shallow Neural Networks – Part 2 Study Notes

This guide is a detailed, example-rich breakdown of the key topics covered in Part 2 of Week 3 in Course 1 - Deep Learning Specializtion. 

It includes:

- Activation functions and when to use them
- Why non-linearities are critical
- Derivatives of common activation functions
- Backpropagation with equations and intuition
- Why and how to do random initialization

---

## 1. Activation Functions

### What is an Activation Function?

An activation function applies a **non-linearity** to each neuron's output. Without it, the neural network would behave like a linear model, regardless of the number of layers.

### Common Activation Functions:

| Function        | Range       | Formula                              | Notes |
|----------------|-------------|--------------------------------------|-------|
| Sigmoid         | (0, 1)      | `σ(z) = 1 / (1 + e^(–z))`             | Good for **binary output**, poor for hidden layers |
| Tanh            | (–1, 1)     | `tanh(z) = (e^z – e^(–z)) / (e^z + e^(–z))` | Centered at 0, better for hidden layers than sigmoid |
| ReLU            | [0, ∞)      | `ReLU(z) = max(0, z)`                | Most common today; fast and simple |
| Leaky ReLU      | (–∞, ∞)     | `LeakyReLU(z) = max(α·z, z)`         | Avoids zero-gradient for z < 0 |

### 🧠 Rules of Thumb:

- **Use `sigmoid` for output layer only** when doing binary classification.
- **Use `ReLU` by default** for hidden layers.
- Use **`tanh`** for zero-centered data when you want something smoother.
- Try **`Leaky ReLU`** if you face dead neurons with ReLU.

### 📌 Example:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Test example
z = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", sigmoid(z))
print("Tanh:", tanh(z))
print("ReLU:", relu(z))
print("Leaky ReLU:", leaky_relu(z))
```

---

## 2.  Why Non-Linear Activation Functions?

Without non-linear activation functions, your network is **just a linear function**, no matter how many layers you stack:

Example:
```
a1 = z1 = W1·x + b1
a2 = z2 = W2·a1 + b2
```

Then,
```
a2 = W2·(W1·x + b1) + b2
   = W'·x + b'      ← Still linear!
```

### 🧠 Linear + Linear = Linear
You **must** apply a nonlinear function like ReLU, tanh, etc., in the hidden layers to allow your network to model complex data.

#### Exception:
- Linear output (no activation) is OK **only for regression tasks**, e.g., predicting house prices.

---

## 3.  Derivatives of Activation Functions

When implementing **backpropagation**, you need the derivatives of each activation function.

### ✅ Sigmoid:
```
a = sigmoid(z)
sigmoid'(z) = a * (1 - a)
```

### ✅ Tanh:
```
a = tanh(z)
tanh'(z) = 1 - a²
```

### ✅ ReLU:
```
relu'(z) = 1 if z > 0 else 0
```

### ✅ Leaky ReLU:
```
leaky_relu'(z) = 1 if z > 0 else alpha
```

### Example:

```python
def sigmoid_derivative(a):
    return a * (1 - a)

def tanh_derivative(a):
    return 1 - np.square(a)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

---

## 4. Backpropagation: Vectorized Equations

For a 2-layer NN:

### Forward Prop:
```
Z1 = W1·X + b1
A1 = g1(Z1)

Z2 = W2·A1 + b2
A2 = g2(Z2)    ← output
```

### Cost Function (Binary):
```
J = -(1/m) * Σ [y log(a2) + (1 - y) log(1 - a2)]
```

### Backward Prop:
```
dZ2 = A2 - Y
dW2 = (1/m) * dZ2·A1.T
db2 = (1/m) * sum(dZ2)

dZ1 = W2.T·dZ2 * g1'(Z1)
dW1 = (1/m) * dZ1·X.T
db1 = (1/m) * sum(dZ1)
```

---

## 5. 📐 Backpropagation Intuition


# 🧠 Understanding Backpropagation in Neural Networks

Backpropagation is the backbone of learning in neural networks. This guide breaks down what it is, how it works, and derives the math behind it step-by-step—including answers to common follow-up questions.

---

## 🔁 What is Backpropagation?

Backpropagation computes the **gradients of the loss** with respect to each parameter (weights and biases). These gradients are used by gradient descent to update the parameters in the direction that reduces the loss.

**In summary:**
- You **forward propagate** to compute the predictions.
- You **compute the loss** between predictions and truth.
- You **backward propagate** using calculus (chain rule) to compute how the error flows back through the network.

---

## 🧭 Forward Propagation Setup

Consider a 2-layer neural network:

```
X → [W1, b1] → Z1 → A1 → [W2, b2] → Z2 → A2 → Loss
```

### Equations:
```
Z1 = W1 · X + b1
A1 = g1(Z1)       # Activation function like ReLU or tanh

Z2 = W2 · A1 + b2
A2 = g2(Z2)       # Final activation, usually sigmoid
```

---

## 🎯 Cost Function

We’ll use **binary cross-entropy loss** for classification:

```text
J = (1/m) * sum[ -Y·log(A2) - (1-Y)·log(1-A2) ]
```

---

## 🔄 Full Chain Rule Explanation

We want to compute the gradients of J with respect to parameters using the **chain rule**:

- `dJ/dW2 = dJ/dZ2 * dZ2/dW2`
- `dJ/db2 = dJ/dZ2 * dZ2/db2`
- `dJ/dZ2 = dJ/dA2 * dA2/dZ2`
- and so on for hidden layers

Each gradient is computed by propagating the derivative of the cost backward through each layer, multiplying by the derivative of each intermediate function.

The chain rule tells us how a small change in one variable affects another when they are **composed functions**. In backpropagation, we apply the chain rule layer by layer starting from the loss, working backwards to the inputs.

---

### Backpropagation Derivation

We want to compute:
- dW2, db2
- dW1, db1

---

#### Step 1: Derivative of w.r.t. A2 and Z2
∂J/∂A2 =−((Y/A2)−(1-Y)/(1-A2)) 

Using the simplification from sigmoid and binary cross-entropy:

```text
dZ2 = A2 - Y
```

This comes from applying the chain rule to:
```text
A2 = sigmoid(Z2)
J = loss(A2, Y)
∂J/∂A2 . ∂A2/∂Z2
```

---

#### Step 2: Derivative w.r.t. W2

```text
Z2 = W2 · A1 + b2
```
from ∂J/∂W2 = ∂J/∂Z2 . ∂Z2/∂W2

∂J/∂Z2 = dZ2

and ∂Z2/∂W2 = A1.T


Then:
```text
∂J/∂W2 = dZ2 · A1.T
```


### ❓ Why is ∂Z2/∂W2 = A1.T?

Zoom in on one example:
```text
Z2 = W2 · A1 + b2 = sum_j (w_2j · a_j)
```

Taking the derivative of scalar Z2 w.r.t. vector W2:
```text
∂Z2/∂w_2j = a_j → ∂Z2/∂W2 = A1.T
```

This transposes A1 so the shape matches W2, which is (1, n₁).

---

#### Step 3: Derivatives for db2

```text
db2 = (1/m) * sum(dZ2, axis=1, keepdims=True)
```

---

#### Step 4: Backprop to Hidden Layer

```text
dZ1 = (W2.T · dZ2) * g1'(Z1)
```

Chain rule explains:
- `∂J/∂A1 = W2.T · dZ2`
- `∂A1/∂Z1 = g1'(Z1)`

So:
```text
dZ1 = (W2.T · dZ2) ⊙ g1'(Z1)
```

---

#### Step 5: Derivatives for W1 and b1

```text
dW1 = (1/m) · dZ1 · X.T
db1 = (1/m) · sum(dZ1, axis=1, keepdims=True)
```

---

### Dimensions Overview

| Variable | Shape |
|----------|--------|
| X        | (nₓ, m) |
| W1       | (n₁, nₓ) |
| b1       | (n₁, 1) |
| A1       | (n₁, m) |
| W2       | (1, n₁) |
| b2       | (1, 1) |
| A2       | (1, m) |

---

### Final Backprop Equations

```python
# Forward
Z1 = W1 @ X + b1
A1 = g1(Z1)
Z2 = W2 @ A1 + b2
A2 = sigmoid(Z2)

# Backward
dZ2 = A2 - Y
dW2 = (1/m) * dZ2 @ A1.T
db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

dZ1 = (W2.T @ dZ2) * g1_derivative(Z1)
dW1 = (1/m) * dZ1 @ X.T
db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
```

---

### Tips

- Always match the **shapes** of gradients to the parameters.
- `A.T` ensures your shapes align correctly for matrix multiplication.
- The **transpose** comes from applying matrix calculus and needing the gradient to match the shape of the parameter you're differentiating.

---

## 6. 🎲 Random Initialization

If you initialize **all weights to 0**, every neuron learns the same thing (symmetry). You must **break symmetry** by using **random weights**.

### Good practice:
```python
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))

W2 = np.random.randn(n2, n1) * 0.01
b2 = np.zeros((n2, 1))
```

- **Weights**: small random values
- **Biases**: zeros are fine

---

## ✅ Summary Table

| Component         | Recommendation                            |
|------------------|--------------------------------------------|
| Hidden Layer Act | ReLU (default), Tanh (sometimes), Leaky ReLU |
| Output Layer Act | Sigmoid (binary), Linear (regression)     |
| Derivatives      | Use simple formulas like `a*(1-a)`         |
| Init Weights     | Random small values (e.g., × 0.01)         |
| Init Biases      | Zeros are okay                             |

---

This completes Part 2! 