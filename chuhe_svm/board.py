"""棋盘：造数据 + 画水墨风的决策边界。"""
import numpy as np


def make_two_armies(n=40, gap=1.6, spread=0.9, seed=7):
    """隔河对峙的红黑两军（基本线性可分）。标签：红=+1，黑=-1。"""
    rng = np.random.default_rng(seed)
    red = rng.normal([0.0, -gap], spread, size=(n, 2))
    black = rng.normal([0.0, gap], spread, size=(n, 2))
    red[:, 0] *= 2.2
    black[:, 0] *= 2.2
    X = np.vstack([red, black])
    y = np.r_[np.ones(n), -np.ones(n)]
    return X, y


def make_siege(n=60, seed=3):
    """围城：红军被黑军团团围住（线性不可分）。"""
    rng = np.random.default_rng(seed)
    red = rng.normal(0, 0.55, size=(n, 2))
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.normal(2.6, 0.3, n)
    black = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    X = np.vstack([red, black])
    y = np.r_[np.ones(n), -np.ones(n)]
    return X, y


def draw_board(model, X, y, title, path, levels=(-1, 0, 1)):
    """把模型画成一张宣纸棋盘图并保存。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["KaiTi", "STKaiti", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    paper, ink, red = "#efe5cf", "#151515", "#b3261e"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130, facecolor=paper)
    ax.set_facecolor(paper)
    pad = 0.8
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=[-1, 1], colors=["#3c5560"], alpha=0.18)   # 河
    ax.contour(xx, yy, zz, levels=list(levels), colors=[ink, red, ink],
               linestyles=["--", "-", "--"], linewidths=[1.4, 2.6, 1.4])

    for lab, col, name in ((1, red, "红方"), (-1, ink, "黑方")):
        m = y == lab
        ax.scatter(X[m, 0], X[m, 1], s=150, facecolor="#f4ead4", edgecolor=col, linewidth=2.2, zorder=3, label=name)
        ax.scatter(X[m, 0], X[m, 1], s=18, color=col, zorder=4)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=420, facecolor="none", edgecolor=red, linewidth=1.6,
               linestyle=(0, (4, 3)), zorder=5, label=f"支持向量 ({len(sv)})")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(title, fontsize=20, color=ink, pad=12)
    ax.grid(color=ink, alpha=0.18, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(ink)
        s.set_alpha(0.5)
    ax.tick_params(colors=ink, labelsize=9)
    leg = ax.legend(loc="upper right", fontsize=11, facecolor=paper, edgecolor=ink)
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout()
    fig.savefig(path, facecolor=paper)
    plt.close(fig)
    return path
