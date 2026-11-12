"""运行： python tests/test_smo.py   或   pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chuhe_svm import SVC, make_siege, make_two_armies  # noqa: E402


def test_hand_example():
    # 《统计学习方法》例 7.1：正例 (3,3),(4,3)，负例 (1,1) → w=(0.5,0.5), b=-2
    X = np.array([[3.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
    y = np.array([1, 1, -1])
    m = SVC(C=1e6, kernel="linear").fit(X, y)
    assert np.allclose(m.coef_, [0.5, 0.5], atol=1e-3), m.coef_
    assert abs(m.intercept_ + 2.0) < 1e-3, m.intercept_
    assert sorted(m.support_) == [0, 2]


def test_kkt_and_constraint():
    X, y = make_two_armies(gap=1.1, spread=0.9, seed=11)
    m = SVC(C=1.0, kernel="linear").fit(X, y)
    assert abs(m.dual_coef_.sum()) < 1e-6          # sum(a_i y_i) = 0
    assert m.alpha_.min() > 0 and m.alpha_.max() <= 1.0 + 1e-9


def test_rbf_solves_siege():
    X, y = make_siege()
    assert SVC(C=1, kernel="linear").fit(X, y).score(X, y) < 0.8
    assert SVC(C=1, kernel="rbf", gamma=0.5).fit(X, y).score(X, y) > 0.98


def test_matches_sklearn():
    try:
        from sklearn.svm import SVC as SkSVC
    except ImportError:
        return
    X, y = make_two_armies(gap=2.6, spread=0.6)
    a = SVC(C=10, kernel="linear").fit(X, y)
    b = SkSVC(C=10, kernel="linear").fit(X, y)
    assert np.allclose(a.coef_, b.coef_[0], atol=0.02), (a.coef_, b.coef_)
    assert abs(a.intercept_ - b.intercept_[0]) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("通过", name)
