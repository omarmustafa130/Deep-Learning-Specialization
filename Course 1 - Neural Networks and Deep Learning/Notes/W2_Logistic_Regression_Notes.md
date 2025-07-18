
# Logistic Regression as a Neural Network – In-Depth Study Notes

## 1. Binary Classification
Binary classification is the task of classifying data into two categories such as:
- Email: spam (1) or not spam (0)
- Tumor: malignant (1) or benign (0)

We train a model on labeled examples to learn a decision boundary.

---

## 2. Logistic Regression
Logistic regression predicts the probability that a given input belongs to class 1 using:

$$
\hat{y} = \sigma(w^T x + b)
$$

Where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function
- $w$ is the weight vector
- $b$ is the bias term

**Example:**  
Given $x = [1.5, 2.0], w = [0.4, -0.6], b = 0.1$:  
$z = -0.5$, and $\hat{y} = \sigma(-0.5) \approx 0.377$

---

## 3. Logistic Regression Cost Function
The loss (Binary Crossentropy) for one example is:

$$
\mathcal{L}(\hat{y}, y) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
$$

Cost over all examples:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
$$

---

## 4. Gradient Descent
Used to minimize $J(w, b)$:

$$
w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}
$$

---

## 5. Derivatives
Measure the rate of change. For example:

$$
f(x) = x^2 \Rightarrow \frac{df}{dx} = 2x
$$

---

## 6. More Derivative Examples
Chain rule example:

$$
f(x) = \sin(x^2) \Rightarrow \frac{df}{dx} = \cos(x^2) \cdot 2x
$$

---

## 7. Computation Graph
Breaks functions into operations. Example:

$$
J = (3x + 2)^2
$$

Nodes:
1. $z = 3x + 2$
2. $J = z^2$

Backpropagation:
$$
\frac{dJ}{dx} = \frac{dJ}{dz} \cdot \frac{dz}{dx} = 2z \cdot 3
$$

---

## 8. Derivatives with a Computation Graph
Example:

$$
a = x^2,\quad b = a + 2,\quad c = \sin(b)
$$

$$
\frac{dc}{dx} = \cos(b) \cdot 1 \cdot 2x
$$

---

## 9. Logistic Regression Gradient Descent
Gradients:

$$
\frac{\partial J}{\partial w} = \frac{1}{m} X^T(\hat{y} - y), \quad
\frac{\partial J}{\partial b} = \frac{1}{m} \sum(\hat{y} - y)
$$

---

## 10. Gradient Descent on m Examples
Batch gradient descent over all training examples:

```python
dw = (1/m) * np.dot(X, (y_hat - y).T)
db = (1/m) * np.sum(y_hat - y)
```

---

## 11. Derivation of DL/dz (Optional)
$$
\frac{d\mathcal{L}}{dz} = \hat{y} - y
$$

Simplified due to sigmoid and binary cross-entropy combination.
