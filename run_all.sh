#!/bin/bash

# すべての最適化手法を実行し、結果を可視化するスクリプト

set -e  # エラーが発生したら終了

echo "======================================"
echo "最適化プログラムの実行と可視化"
echo "======================================"

# 1. Haskellプログラムのビルドと実行
echo ""
echo "1. Haskellプログラムを実行中..."
cd haskell
cabal run nlp-class-haskell
echo "✓ 最適化完了: 結果はhaskell/results/に保存されました"

# 2. Python可視化の実行
echo ""
echo "2. 結果を可視化中..."
cd ../python

echo "  - 内点法を可視化中..."
uv run python src/main.py ../haskell/results/interior_result.json interior

echo "  - 外点法を可視化中..."
uv run python src/main.py ../haskell/results/exterior_result.json exterior

echo "  - 拡張ラグランジュ法を可視化中..."
uv run python src/main.py ../haskell/results/lagrangian_result.json lagrange

echo ""
echo "3. rho初期値による比較を可視化中..."

echo "  - 内点法のrho比較..."
uv run python src/main.py "../haskell/results/interior_rho*_result.json" interior --compare

echo "  - 外点法のrho比較..."
uv run python src/main.py "../haskell/results/exterior_rho*_result.json" exterior --compare

echo ""
echo "======================================"
echo "✓ すべて完了しました！"
echo "======================================"
echo ""
echo "生成されたファイル:"
echo "  - JSON結果: haskell/results/*.json"
echo "  - 可視化画像: python/images/*.png"
echo "  - rho比較プロット: python/images/*_rho_comparison.png"
echo ""
