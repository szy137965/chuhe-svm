"""三局棋：线性对峙 / 软间隔 / 围城（RBF 核）。

运行： python examples/demo.py
会在 assets/ 下生成三张图，并在终端打印和 scikit-learn 的对比。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chuhe_svm import SVC, draw_board, make_siege, make_two_armies  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)


def versus_sklearn(X, y, **kw):
    """如果装了 scikit-learn，就拿它当裁判。"""
    try:
        from sklearn.svm import SVC as SkSVC
    except ImportError:
        return None
    return SkSVC(**kw).fit(X, y)


def main():
    # ---------- 第一局：隔河对峙（线性核，C 很大 ≈ 硬间隔）
    X, y = make_two_armies(gap=2.6, spread=0.6)
    m = SVC(C=1e3, kernel="linear").fit(X, y)
    print("【第一局 · 隔河对峙】")
    print(f"  w = {np.round(m.coef_, 4)}, b = {m.intercept_:.4f}")
    print(f"  河宽 2/||w|| = {m.margin_:.4f}，支持向量 {len(m.support_)} 个，训练准确率 {m.score(X, y):.3f}")
    sk = versus_sklearn(X, y, C=1e3, kernel="linear")
    if sk is not None:
        print(f"  sklearn: w = {np.round(sk.coef_[0], 4)}, b = {sk.intercept_[0]:.4f}, 支持向量 {len(sk.support_)} 个")
    draw_board(m, X, y, "第一局 · 隔河对峙（线性核）", os.path.join(ASSETS, "game1_linear.png"))

    # ---------- 第二局：短兵相接（两军有交叠，看 C 的作用）
    X, y = make_two_armies(gap=1.1, spread=0.9, seed=11)
    print("【第二局 · 短兵相接】")
    for C in (0.01, 1, 100):
        m = SVC(C=C, kernel="linear").fit(X, y)
        print(f"  C={C:<6} 河宽={m.margin_:.3f}  支持向量={len(m.support_):>2}  训练准确率={m.score(X, y):.3f}")
        if C == 1:
            draw_board(m, X, y, "第二局 · 短兵相接（软间隔 C=1）", os.path.join(ASSETS, "game2_soft.png"))

    # ---------- 第三局：围城（RBF 核）
    X, y = make_siege()
    lin = SVC(C=1, kernel="linear").fit(X, y)
    m = SVC(C=1, kernel="rbf", gamma=0.5).fit(X, y)
    print("【第三局 · 围城】")
    print(f"  线性核训练准确率 {lin.score(X, y):.3f}  →  RBF 核 {m.score(X, y):.3f}（支持向量 {len(m.support_)} 个）")
    sk = versus_sklearn(X, y, C=1, kernel="rbf", gamma=0.5)
    if sk is not None:
        agree = np.mean(sk.predict(X) == m.predict(X))
        print(f"  与 sklearn 预测一致率 {agree:.3f}")
    draw_board(m, X, y, "第三局 · 围城（RBF 核）", os.path.join(ASSETS, "game3_rbf.png"))


if __name__ == "__main__":
    main()
