"""Haskellの最適化結果を可視化するスクリプト."""

import json
import re
import sys
from collections.abc import Callable
from glob import glob
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


def plot_rho_comparison(
    results: dict[float, dict[str, Any]],
    method_type: MethodType,  # noqa: ARG001
    title: str,
) -> Figure:
    """異なるrho初期値での収束過程を比較するプロット."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # カラーマップを取得
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.9, len(results)))

    # 収束過程の比較
    for (rho, result_data), color in zip(
        sorted(results.items()),
        colors,
        strict=True,
    ):
        iterations = [np.array(it) for it in result_data["iterations"]]
        if not iterations:
            continue

        values = [squared(it) for it in iterations]
        ax1.plot(
            range(len(values)),
            values,
            "-o",
            color=color,
            linewidth=2,
            markersize=4,
            label=f"rho_0 = {rho}",
        )

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Objective Value f(x)", fontsize=12)
    ax1.set_title(
        f"{title} - Convergence Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(visible=True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.legend(fontsize=10)

    # 最終解の比較
    rho_values = []
    final_values = []
    solution_values = []

    for rho, result_data in sorted(results.items()):
        rho_values.append(rho)
        final_values.append(result_data["finalValue"])
        solution = np.array(result_data["solution"])
        solution_values.append(
            solution[0] if len(solution) == 1 else np.linalg.norm(solution),
        )

    ax2_twin = ax2.twinx()

    line1 = ax2.plot(
        rho_values,
        final_values,
        "b-o",
        linewidth=2,
        markersize=8,
        label="Final Objective Value",
    )
    ax2.set_xlabel("Initial ρ₀", fontsize=12)
    ax2.set_ylabel("Final Objective Value", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.set_xscale("log")
    ax2.grid(visible=True, alpha=0.3)

    line2 = ax2_twin.plot(
        rho_values,
        solution_values,
        "r-s",
        linewidth=2,
        markersize=8,
        label="Solution Value",
    )
    ax2_twin.set_ylabel("Solution x", color="r", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="r")

    # 凡例を統合
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="best", fontsize=10)

    ax2.set_title(
        f"{title} - Final Results vs Initial rho",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def compare_rho_results(
    json_pattern: str | Path,
    method_type: MethodType,
) -> None:
    """複数のrho値の結果を比較して可視化."""
    # パターンに一致するファイルを探す
    json_files = sorted(glob(str(json_pattern)))

    if not json_files:
        print(f"No files matching pattern: {json_pattern}")
        return

    print(f"Found {len(json_files)} result files")

    # 各ファイルからrho値と結果を読み込む
    results: dict[float, dict[str, Any]] = {}

    for json_file in json_files:
        print(f"Loading {json_file}")
        with Path(json_file).open() as f:
            result_data = json.load(f)

        # ファイル名からrho値を抽出 (例: interior_rho10.0_result.json)

        match = re.search(r"rho([\d.]+)_result\.json", json_file)
        if match:
            rho = float(match.group(1))
            results[rho] = result_data
            print(f"  Loaded rho={rho}, solution={result_data['solution']}")

    if not results:
        print("No valid results found")
        return

    # imagesディレクトリを作成
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    # タイトル設定
    if method_type == "interior":
        title = "Interior Penalty Method"
    elif method_type == "exterior":
        title = "Exterior Penalty Method"
    else:
        title = f"{method_type.capitalize()} Method"

    # 比較プロットを作成
    fig = plot_rho_comparison(results, method_type, title)
    output_path = images_dir / f"{method_type}_rho_comparison.png"
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    print(f"\nSaved comparison plot to {output_path}")

    plt.show()


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
        print(
            "Usage: python main.py <json_file_or_pattern> [method_type] [--compare]",
        )
        print(
            "method_type: interior, exterior, lagrange, lagrangian (default: interior)",
        )
        print("\nOptions:")
        print("  --compare: Compare multiple results with different rho values")
        print("\nExamples:")
        print("  python main.py results/interior_result.json interior")
        print(
            '  python main.py "results/interior_rho*_result.json" interior --compare',
        )
        sys.exit(1)

    json_file = sys.argv[1]
    method_type = (
        sys.argv[method_type_arg_index]
        if len(sys.argv) > min_args
        else "interior"
    )

    # --compareオプションのチェック
    compare_mode = "--compare" in sys.argv

    if compare_mode:
        compare_rho_results(json_file, cast("MethodType", method_type))
    else:
        visualize_from_json(json_file, cast("MethodType", method_type))


if __name__ == "__main__":
    main()
