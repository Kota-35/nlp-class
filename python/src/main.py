"""Haskellの最適化結果を可視化するスクリプト."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

type MethodType = Literal["interior", "exterior", "lagrange", "lagrangian"]


def squared(x: NDArray[np.floating[Any]]) -> np.floating[Any]:
    """目的関数: sum(x^2)."""
    return np.sum(x**2)


def interior_penalty(
    x: NDArray[np.floating[Any]],
    rho: float,
) -> np.floating[Any]:
    """内点ペナルティ関数."""
    g_i = np.maximum(1, x)
    penalty = np.sum(1.0 / (g_i**2))
    return squared(x) + rho * penalty


def exterior_penalty(
    x: NDArray[np.floating[Any]],
    rho: float,
) -> np.floating[Any]:
    """外点ペナルティ関数."""
    alpha = 2.0
    beta = 2.0
    g = np.maximum(0, -x)
    h = np.abs(np.sum(x) - 2.0)
    penalty = np.sum(g**alpha) + h**beta
    return squared(x) + rho * penalty


def plot_1d_optimization(
    result_data: dict[str, Any],
    penalty_func: Callable[
        [NDArray[np.floating[Any]], float],
        np.floating[Any],
    ],
    rho_final: float,
    title: str,
) -> Figure:
    """1次元最適化問題の可視化."""
    solution = np.array(result_data["solution"])
    iterations = [np.array(it) for it in result_data["iterations"]]

    # プロット範囲
    xx = np.linspace(-4, 4, 200)
    yy_original = np.array([squared(np.array([x])) for x in xx])
    yy_penalty = np.array([penalty_func(np.array([x]), rho_final) for x in xx])

    # プロット作成
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 元の目的関数
    ax1.plot(xx, yy_original, "b-", label="Original f(x)", linewidth=2)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("Original f(x)", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    # 初期値
    if iterations:
        initial = iterations[0][0]
        ax1.plot(
            initial,
            squared(np.array([initial])),
            "ob",
            markersize=10,
            label="Initial point",
        )

    # 反復点
    iter_x = [it[0] for it in iterations[1:]]
    iter_y = [squared(np.array([x])) for x in iter_x]
    ax1.plot(iter_x, iter_y, "ok", markersize=6, label="Iterations")

    # 最終解
    ax1.plot(
        solution[0],
        squared(solution),
        "or",
        markersize=12,
        label="Solution",
    )

    # ペナルティ付き目的関数
    ax2 = ax1.twinx()
    ax2.plot(xx, yy_penalty, "r-", label="With penalty", linewidth=2, alpha=0.7)
    ax2.set_ylabel("With penalty", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")

    # タイトルと凡例
    plt.title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_convergence(result_data: dict[str, Any], title: str) -> Figure | None:
    """収束過程のプロット."""
    iterations = [np.array(it) for it in result_data["iterations"]]

    if not iterations:
        print("No iteration data available")
        return None

    # 各反復での目的関数値
    values = [squared(it) for it in iterations]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(values)), values, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective Value f(x)", fontsize=12)
    ax.set_title(f"{title} - Convergence", fontsize=14, fontweight="bold")
    ax.grid(visible=True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


def visualize_from_json(
    json_file: str | Path,
    method_type: MethodType,
) -> None:
    """JSONファイルから結果を読み込んで可視化."""
    json_path = Path(json_file)
    with json_path.open() as f:
        result_data = json.load(f)

    # imagesディレクトリを作成
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    # ペナルティ関数の選択
    if method_type == "interior":
        penalty_func: (
            Callable[
                [NDArray[np.floating[Any]], float],
                np.floating[Any],
            ]
            | None
        ) = interior_penalty
        rho_final = 10.0 * (5.0**19)  # 最終的なrho
        title = "Interior Penalty Method"
    elif method_type == "exterior":
        penalty_func = exterior_penalty
        rho_final = 0.1 * (5.0**19)
        title = "Exterior Penalty Method"
    elif method_type in {"lagrange", "lagrangian"}:
        penalty_func = None  # ラグランジュ法ではペナルティ関数を使わない
        rho_final = 0
        title = "Augmented Lagrangian Method"
    else:
        print(f"Unknown method type: {method_type}")
        return

    # 1次元問題の場合のみプロット
    solution = np.array(result_data["solution"])
    if len(solution) == 1 and penalty_func is not None:
        plot_1d_optimization(result_data, penalty_func, rho_final, title)
        output_path = images_dir / f"{method_type}_optimization.png"
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved {output_path}")

    # 収束過程のプロット
    fig2 = plot_convergence(result_data, title)
    if fig2:
        output_path = images_dir / f"{method_type}_convergence.png"
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved {output_path}")

    plt.show()


def main() -> None:
    """メイン関数."""
    min_args = 2
    method_type_arg_index = 2

    if len(sys.argv) < min_args:
        print("Usage: python main.py <json_file> [method_type]")
        print(
            "method_type: interior, exterior, lagrange, lagrangian (default: interior)",
        )
        sys.exit(1)

    json_file = sys.argv[1]
    method_type = (
        sys.argv[method_type_arg_index]
        if len(sys.argv) > min_args
        else "interior"
    )

    visualize_from_json(json_file, cast("MethodType", method_type))


if __name__ == "__main__":
    main()
