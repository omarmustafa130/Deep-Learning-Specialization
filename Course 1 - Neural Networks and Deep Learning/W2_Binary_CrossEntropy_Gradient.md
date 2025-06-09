
# Binary Cross-Entropy Loss and Gradient Derivation

---

## 🔍 Step 1: Binary Cross-Entropy Loss

For binary classification, the loss function for a single example is:

$$
\mathcal{L}(\hat{y}, y) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

Where:
- $y$ is the true label (0 or 1)
- $\hat{y}$ is the predicted probability from the sigmoid function

---

## 🧠 Step 2: Goal – Compute the Gradient w.r.t $z$

In logistic regression:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b
$$

We want:

$$
\frac{d\mathcal{L}}{dz}
$$

Apply the chain rule:

$$
\frac{d\mathcal{L}}{dz} = \frac{d\mathcal{L}}{d\hat{y}} \cdot \frac{d\hat{y}}{dz}
$$

---

## 🔧 Step 3: Compute Each Part

### ① Derivative of Loss w.r.t $\hat{y}$

$$
\frac{d\mathcal{L}}{d\hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}
$$

### ② Derivative of Sigmoid w.r.t $z$

$$
\frac{d\hat{y}}{dz} = \hat{y}(1 - \hat{y})
$$

---

## 🧮 Step 4: Combine Using Chain Rule

Multiply the two:

$$
\frac{d\mathcal{L}}{dz} = \left(-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} \right) \cdot \hat{y}(1 - \hat{y})
$$

Distribute $\hat{y}(1 - \hat{y})$:

$$
= -y(1 - \hat{y}) + (1 - y)\hat{y}
$$

Simplify:

$$
= -y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y
$$

---

## ✅ Final Gradient

$$
\frac{d\mathcal{L}}{dz} = \hat{y} - y
$$

---

## 🧠 Why This Matters

- Simple and efficient to compute
- Directly used in training logistic regression and neural networks

---

## 🔁 Example in Gradient Descent

```python
dz = y_hat - y             # (1, m)
dw = (1/m) * np.dot(X, dz.T)
db = (1/m) * np.sum(dz)
```

This gradient is used to update weights and bias during training.
