# -*- coding: utf-8 -*-
"""Plotting helpers (matplotlib / seaborn).  All figures annotated with N."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 9


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmap(mat, marks, title, path, n=None, cbar_label="r",
            xlabel="brain metric", ylabel="organ", figsize=None, vmax=None):
    """mat: DataFrame (organs x metrics). marks: same-shape str DataFrame."""
    nrow, ncol = mat.shape
    figsize = figsize or (max(8, ncol * 0.28 + 3), max(4, nrow * 0.4 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    vmax = vmax or np.nanmax(np.abs(mat.values)) or 0.1
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=marks, fmt="", annot_kws={"size": 6}, ax=ax,
                cbar_kws={"label": cbar_label}, linewidths=0.3, linecolor="white")
    ax.set_title(title + (f"   (N≈{n})" if n else ""))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    save(fig, path)


def forest(df, est, lo, hi, label, title, path, xline=1.0, xlabel="OR (95% CI)",
           logx=False, color_col=None):
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.32 + 1)))
    y = np.arange(len(df))
    colors = ["#2c7fb8"] * len(df)
    if color_col and color_col in df:
        colors = ["#d95f02" if v else "#1b9e77" for v in df[color_col]]
    for i, row in df.iterrows():
        ax.plot([row[lo], row[hi]], [y[i], y[i]], color=colors[i], lw=1.5, zorder=1)
    ax.scatter(df[est], y, color=colors, zorder=2, s=28)
    ax.axvline(xline, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(df[label], fontsize=7)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.invert_yaxis()
    save(fig, path)


def bar(df, x, y, title, path, hue=None, xlabel=None, ylabel=None, rotate=90,
        color=None):
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4 + 2), 4))
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, color=color, palette=None if color else "viridis")
    ax.set_title(title); ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y)
    plt.setp(ax.get_xticklabels(), rotation=rotate, fontsize=7, ha="right")
    save(fig, path)


def dose_curve(group_means, title, path, ylabel="metric mean", n=None):
    """group_means: DataFrame indexed by group with columns mean,std,count."""
    fig, ax = plt.subplots(figsize=(5, 4))
    x = group_means.index.astype(float)
    ax.errorbar(x, group_means["mean"],
                yerr=group_means["std"] / np.sqrt(group_means["count"].clip(lower=1)),
                marker="o", capsize=3, color="#2c7fb8")
    ax.set_xlabel("cumulative extreme-aging burden (n organs)")
    ax.set_ylabel(ylabel + "  (mean ± SEM)")
    ax.set_title(title + (f"   (N={n})" if n else ""))
    save(fig, path)


def scatter(x, y, labels, title, path, xlabel, ylabel, rho=None, p=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(x, y, color="#2c7fb8", s=35)
    for xi, yi, l in zip(x, y, labels):
        ax.annotate(l, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    sub = title
    if rho is not None:
        sub += f"\nSpearman rho={rho:.2f}, p={p:.3g}"
    ax.set_title(sub); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    save(fig, path)


def violin(df, cols, title, path, ncol=6):
    n = len(cols); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cols):
        sns.violinplot(y=pd.to_numeric(df[c], errors="coerce").dropna(), ax=axes[i],
                       color="#8da0cb")
        axes[i].set_title(c, fontsize=7); axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    save(fig, path)


def missingness(df, cols, title, path):
    miss = df[cols].isna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, max(3, len(cols) * 0.2)))
    ax.barh(range(len(miss)), miss.values, color="#fc8d62")
    ax.set_yticks(range(len(miss))); ax.set_yticklabels(miss.index, fontsize=6)
    ax.set_xlabel("missing fraction"); ax.set_title(title)
    save(fig, path)


def grouped_compare(df, cat, val_m, val_f, title, path, ylabel):
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.5 + 2), 4))
    x = np.arange(len(df)); w = 0.4
    ax.bar(x - w / 2, df[val_m], w, label="male", color="#4575b4")
    ax.bar(x + w / 2, df[val_f], w, label="female", color="#d73027")
    ax.set_xticks(x); ax.set_xticklabels(df[cat], rotation=90, fontsize=6)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.axhline(0, color="grey", lw=0.6)
    save(fig, path)
