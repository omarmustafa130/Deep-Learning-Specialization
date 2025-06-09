
# Python and Vectorization – In-Depth Study Notes

---

## 1. Vectorization

Vectorization replaces explicit loops with matrix and vector operations. It's faster and cleaner.

**Example:**

### Non-vectorized:
```python
result = np.zeros(3)
for i in range(3):
    result[i] = a[i] + b[i]
```

### Vectorized:
```python
result = a + b
```

---

## 2. More Vectorization Examples

### Scalar multiplication:
```python
a = np.array([1, 2, 3])
a = a * 2  # Vectorized
```

### Dot product:
```python
dot = np.dot(a, b)  # Vectorized
```

---

## 3. Vectorizing Logistic Regression

For m examples:

### Non-vectorized:
```python
for i in range(m):
    z[i] = np.dot(w, X[i]) + b
    a[i] = sigmoid(z[i])
```
X[i] is the i-th training example, shape: (n,)

w is the weight vector, shape: (n,)

np.dot(w, X[i]) → scalar

This runs once per training example. No need to transpose anything — you're just computing a dot product for a single vector.
### Vectorized:
```python
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
```
Here we compute all examples at once:

X shape: (n, m) → each column is an example

w shape: (n, 1) (column vector)

So to compute the dot product across all examples:

w.T shape: (1, n)

X shape: (n, m)

Z = w.T @ X → shape: (1, m) → one activation per example

### Why Transpose?
We transpose w to make its shape compatible for matrix multiplication with X. If we didn’t transpose, the shapes would mismatch:

Without .T: (n, 1) @ (n, m) → ❌ invalid

With .T: (1, n) @ (n, m) → ✅ gives (1, m)


---

## 4. Vectorizing Logistic Regression’s Gradient Output

### Equations:
$$
\frac{\partial J}{\partial w} = \frac{1}{m} X(\hat{y} - y)^T
$$
$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum(\hat{y} - y)
$$

### Code:
```python
dw = (1 / m) * np.dot(X, (A - Y).T)
db = (1 / m) * np.sum(A - Y)
```

---

## 5. Broadcasting in Python

Automatically expands dimensions to make shapes compatible.

**Example:**
```python
A = np.array([[1, 2], [3, 4]])
b = np.array([1, 0])
A + b  # b is broadcast over rows
```

Result:
```
[[2, 2],
 [4, 4]]
```

---

## 6. A Note on Python/NumPy Vectors

Python lists vs. NumPy arrays:
```python
[1, 2, 3] * 2     # [1,2,3,1,2,3]
np.array([1, 2, 3]) * 2  # [2, 4, 6]
```

Vector shapes matter:
- Row vector: (1, n)
- Column vector: (n, 1)

---

## 7. Quick Tour of Jupyter/iPython Notebooks

- Interactive coding with live output
- Markdown and code cells
- Use `Shift + Enter` to run cells

**Example:**
```python
import numpy as np
X = np.array([[1, 2], [3, 4]])
X.shape  # Output: (2, 2)
```

---

## 8. Explanation of Logistic Regression Cost Function (Optional)

Log-likelihood:

$$
\log(P) = y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})
$$

Negative log-likelihood (loss):

$$
\mathcal{L} = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
$$

This penalizes confident incorrect predictions more.
