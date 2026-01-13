{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- HLINT ignore "Avoid lambda using `infix`" -}
{- HLINT ignore "Avoid lambda" -}
{- HLINT ignore "Parenthesize unary negation" -}
{- HLINT ignore "Redundant bracket" -}

module Optimization where

import Data.Aeson
import qualified Data.Vector.Storable as VS
import Numeric.LinearAlgebra

type VectorD = Vector Double
type MatrixD = Matrix Double

data OptimizationProblem = OptimizationProblem
    { object :: VectorD -> Double
    , constraints :: Maybe (VectorD -> VectorD) -- 等式成約
    , inequalities :: Maybe (VectorD -> VectorD) -- 不等式制約
    }

-- Newton法の設定
data NewtonConfig = NewtonConfig
    { maxIterations :: Int
    , tolerance :: Double
    , epsilon :: Double -- 数値微分の刻み幅
    }

-- ペナルティ法の結果
data OptimizationResult = OptimizationResult
    { solution :: VectorD
    , iterations :: [VectorD]
    , finalValue :: Double
    }
    deriving (Show)

-- JSON出力のためのインスタンス定義
instance ToJSON OptimizationResult where
    toJSON OptimizationResult{..} =
        Data.Aeson.object
            [ "solution" .= toList solution
            , "iterations" .= map toList iterations
            , "finalValue" .= finalValue
            ]

-- デフォルトの設定
defaultConfig :: NewtonConfig
defaultConfig =
    NewtonConfig
        { maxIterations = 20
        , tolerance = 1e-6
        , epsilon = 1e-5
        }

-------------------------------------------------------------------------------
-- 数値微分
-------------------------------------------------------------------------------

-- Jacobian(勾配)の計算
jacobian :: (VectorD -> Double -> Double) -> VectorD -> Double -> VectorD
jacobian f x rho =
    let n = size x
        eps = 1e-5
        gradient i =
            let x_plus = x + (konst 0 n) VS.// [(i, eps)]
                x_minus = x + (konst 0 n) VS.// [(i, -eps)]
             in (f x_plus rho - f x_minus rho) / (2 * eps)
     in vector [gradient i | i <- [0 .. n - 1]]

-- Hessian行列の計算
hessian :: (VectorD -> Double -> Double) -> VectorD -> Double -> MatrixD
hessian f x rho =
    let n = size x
        eps = 1e-5
        hessianRow i =
            let x_plus = x + (konst 0 n) VS.// [(i, eps)]
                x_minus = x + (konst 0 n) VS.// [(i, -eps)]
                j_plus = jacobian f x_plus rho
                j_minus = jacobian f x_minus rho
             in (j_plus - j_minus) / scalar (2 * eps)
     in fromRows [hessianRow i | i <- [0 .. n - 1]]

-- Newton法による最適化
newtonMethod ::
    NewtonConfig ->
    (VectorD -> Double -> VectorD) -> -- 勾配
    (VectorD -> Double -> MatrixD) -> -- Hessian
    VectorD -> -- 初期値
    Double -> -- rho
    (VectorD, [VectorD]) -- (解, 履歴)
newtonMethod NewtonConfig{..} df d2f x0 rho = go 0 x0 []
  where
    go iter xk history
        | iter >= maxIterations = (xk, reverse history)
        | otherwise =
            let j = df xk rho
                h = d2f xk rho
                dk = -(pinv h) #> j -- 探索方向
                stepNorm = norm_1 dk
             in if stepNorm <= tolerance
                    then (xk, reverse (xk : history))
                    else go (iter + 1) (xk + dk) (xk : history)

-- 内点ペナルティ関数
interiorPenalty :: VectorD -> Double -> Double
interiorPenalty x rho =
    let squared = VS.sum $ VS.map (** 2) (extract x)
        constraints = VS.map (\xi -> max 1.0 xi) (extract x)
        penalty = VS.sum $ VS.map (\gi -> gi ** 2) constraints
     in squared + rho * penalty

-- 内点法による最適化
solveInterior :: NewtonConfig -> VectorD -> Double -> Double -> Int -> OptimizationResult
solveInterior config x0 rho0 gamma maxOuter = go 0 x0 rho0 []
  where
    go iter xi rhoi allHistory
        | iter >= maxOuter = OptimizationResult xi allHistory (interiorPenalty xi rhoi)
        | otherwise =
            let (xStar, history) =
                    newtonMethod
                        config
                        (jacobian interiorPenalty)
                        (hessian interiorPenalty)
                        xi
                        rhoi
                stepNorm = norm_1 (xi - xStar)
             in if stepNorm <= tolerance config
                    then OptimizationResult xStar (allHistory ++ history) (interiorPenalty xStar rhoi)
                    else go (iter + 1) xStar (rhoi * gamma) (allHistory ++ history)

-------------------------------------------------------------------------------
-- 外点ペナルティ法
-------------------------------------------------------------------------------

-- 外点ペナルティ関数: sum(x^2) + rho * (sum(max(0,-x_i)^2) + |sum(x) - 2|^2)
exteriorPenalty :: VectorD -> Double -> Double
exteriorPenalty x rho =
    let squared = VS.sum $ VS.map (** 2) (extract x)
        alpha = 2.0 :: Double
        beta = 2.0 :: Double
        g = VS.map (\xi -> max 0 (-xi)) (extract x)
        gPenalty = VS.sum $ VS.map (** alpha) g
        h = abs (sumElements x - 2.0)
        hPenalty = h ** beta
     in squared + rho * (gPenalty + hPenalty)

-- 外点法による最適化
solveExterior :: NewtonConfig -> VectorD -> Double -> Double -> Int -> OptimizationResult
solveExterior config x0 rho0 gamma maxOuter = go 0 x0 rho0 []
  where
    go iter xi rhoi allHistory
        | iter >= maxOuter = OptimizationResult xi allHistory (exteriorPenalty xi rhoi)
        | otherwise =
            let (xStar, history) =
                    newtonMethod
                        config
                        (jacobian exteriorPenalty)
                        (hessian exteriorPenalty)
                        xi
                        rhoi
                stepNorm = norm_1 (xi - xStar)
             in if stepNorm <= tolerance config
                    then OptimizationResult xStar (allHistory ++ history) (exteriorPenalty xStar rhoi)
                    else go (iter + 1) xStar (rhoi * gamma) (allHistory ++ history)

-------------------------------------------------------------------------------
-- 拡張ラグランジュ法
-------------------------------------------------------------------------------

-- ラグランジュ関数: f(x) + mu'h(x) + (rho/2)||h(x)||^2
data AugmentedLagrangian = AugmentedLagrangian
    { objective :: VectorD -> Double
    , equalityConstraints :: VectorD -> VectorD
    , lagrangeMultipliers :: VectorD
    , penaltyParam :: Double
    }

augmentedLagrangianValue :: AugmentedLagrangian -> VectorD -> Double
augmentedLagrangianValue AugmentedLagrangian{..} x =
    let fx = objective x
        hx = equalityConstraints x
        linear = lagrangeMultipliers <.> hx
        quadratic = 0.5 * penaltyParam * (hx <.> hx)
     in fx + linear + quadratic

-- 拡張ラグランジュ法による最適化
solveAugmentedLagrangian :: NewtonConfig -> VectorD -> Int -> OptimizationResult
solveAugmentedLagrangian config x0 maxOuter =
    let problem =
            AugmentedLagrangian
                { objective = \x -> sumElements (cmap (** 2) x)
                , equalityConstraints = \x -> vector [x VS.! 0 - x VS.! 1 + 3]
                , lagrangeMultipliers = vector [0]
                , penaltyParam = 50.0
                }
     in go 0 x0 problem []
  where
    go iter xi alg allHistory
        | iter >= maxOuter =
            OptimizationResult xi allHistory (augmentedLagrangianValue alg xi)
        | otherwise =
            let f = augmentedLagrangianValue alg
                (xStar, history) =
                    newtonMethod
                        config
                        (\x _ -> jacobian (\y _ -> f y) x 0)
                        (\x _ -> hessian (\y _ -> f y) x 0)
                        xi
                        0
                hx = equalityConstraints alg xStar
                newMu = lagrangeMultipliers alg + scalar (penaltyParam alg) * hx
                newAlg = alg{lagrangeMultipliers = newMu}
                stepNorm = norm_1 (xi - xStar)
             in if stepNorm <= tolerance config
                    then OptimizationResult xStar (allHistory ++ history) (f xStar)
                    else go (iter + 1) xStar newAlg (allHistory ++ history)

-------------------------------------------------------------------------------
-- ヘルパー関数
-------------------------------------------------------------------------------

extract :: VectorD -> VS.Vector Double
extract = VS.convert . flatten . asColumn
