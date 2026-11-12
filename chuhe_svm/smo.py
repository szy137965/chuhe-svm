"""手写 SMO（序列最小优化）求解软间隔支持向量机的对偶问题。

    max_a  sum(a) - 1/2 * sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.t.   0 <= a_i <= C,   sum(a_i y_i) = 0

每一步只挑两个"棋子"(a_i, a_j) 走子，其余不动，这样等式约束始终成立，
而两个变量的子问题有闭式解。挑哪两个？挑违反 KKT 条件最严重的一对
（maximal violating pair），直到没有哪一对还值得动。
"""
import numpy as np

from .kernels import get_kernel


class SVC:
    def __init__(self, C=1.0, kernel="linear", gamma=0.5, degree=3, coef0=1.0,
                 tol=1e-3, max_iter=100000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------ 训练
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("本实现只支持二分类（红方 / 黑方）")
        t = np.where(y == self.classes_[1], 1.0, -1.0)

        C = self.C
        self._kfunc = get_kernel(self.kernel, self.gamma, self.degree, self.coef0)
        K = self._kfunc(X, X)

        eps = 1e-10 * C          # 贴边容差：离 0 或 C 这么近就算"到边了"
        a = np.zeros(X.shape[0])
        F = -t.copy()            # F_i = sum_j a_j t_j K_ij - t_i（不含 b 的误差）
        it = 0
        while it < self.max_iter:
            # 还能"往上走"的棋子 / 还能"往下走"的棋子
            up = ((t > 0) & (a < C - eps)) | ((t < 0) & (a > eps))
            low = ((t > 0) & (a > eps)) | ((t < 0) & (a < C - eps))
            i = int(np.flatnonzero(up)[np.argmin(F[up])])
            j = int(np.flatnonzero(low)[np.argmax(F[low])])
            if F[j] - F[i] < self.tol:      # 没有违反 KKT 的一对了：收官
                break

            # 两个变量的闭式解，再剪到盒子 [L, H] 里
            if t[i] != t[j]:
                L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
            else:
                L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
            eta = max(K[i, i] + K[j, j] - 2.0 * K[i, j], 1e-12)
            aj_new = float(np.clip(a[j] + t[j] * (F[i] - F[j]) / eta, L, H))
            ai_new = a[i] + t[i] * t[j] * (a[j] - aj_new)
            ai_new = min(max(ai_new, 0.0), C)

            F += t[i] * (ai_new - a[i]) * K[:, i] + t[j] * (aj_new - a[j]) * K[:, j]
            a[i], a[j] = ai_new, aj_new
            it += 1

        up = ((t > 0) & (a < C - eps)) | ((t < 0) & (a > eps))
        low = ((t > 0) & (a > eps)) | ((t < 0) & (a < C - eps))
        b = -(F[up].min() + F[low].max()) / 2.0

        sv = a > 1e-8 * C
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (a * t)[sv]
        self.alpha_ = a[sv]
        self.intercept_ = float(b)
        self.n_iter_ = it
        self.coef_ = self.dual_coef_ @ self.support_vectors_ if self.kernel == "linear" else None
        return self

    # ------------------------------------------------------------ 预测
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self._kfunc(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def margin_(self):
        """线性核下的间隔宽度 2/||w||（楚河有多宽）。"""
        if self.coef_ is None:
            raise AttributeError("只有线性核才有显式的 w 和间隔宽度")
        return 2.0 / float(np.linalg.norm(self.coef_))
