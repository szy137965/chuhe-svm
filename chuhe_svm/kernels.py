"""核函数：把"过河"这件事交给内积。"""
import numpy as np


def linear(A, B):
    return A @ B.T


def poly(A, B, degree=3, coef0=1.0, gamma=1.0):
    return (gamma * (A @ B.T) + coef0) ** degree


def rbf(A, B, gamma=0.5):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return np.exp(-gamma * d2)


def get_kernel(name, gamma=0.5, degree=3, coef0=1.0):
    if name == "linear":
        return linear
    if name == "rbf":
        return lambda A, B: rbf(A, B, gamma=gamma)
    if name == "poly":
        return lambda A, B: poly(A, B, degree=degree, coef0=coef0, gamma=gamma)
    raise ValueError(f"未知核函数: {name!r}，可选 linear / rbf / poly")
