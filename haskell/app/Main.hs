module Main where

import Data.Aeson
import qualified Data.ByteString.Lazy.Char8 as BL
import Numeric.LinearAlgebra
import Optimization
import System.Directory (createDirectoryIfMissing)
import System.Random

-- 問題の定義
data ProblemType = Interior | Exterior | AugmentedLag
    deriving (Show, Read)

-- 実行設定
data RunConfig = RunConfig
    { problemType :: ProblemType
    , dimensions :: Int
    , initialRho :: Double
    , gamma :: Double
    , maxOuterIterations :: Int
    }

-- デフォルト設定
defaultRunConfig :: RunConfig
defaultRunConfig =
    RunConfig
        { problemType = Interior
        , dimensions = 1
        , initialRho = 10.0
        , gamma = 5.0
        , maxOuterIterations = 20
        }

-------------------------------------------------------------------------------
-- メイン関数
-------------------------------------------------------------------------------

main :: IO ()
main = do
    -- 乱数シード設定
    setStdGen (mkStdGen 777)

    -- resultsディレクトリを作成
    createDirectoryIfMissing True "results"

    -- 3つの問題をすべて解く
    putStrLn "=== Interior Penalty Method ==="
    resultInterior <- runInteriorProblem
    BL.writeFile "results/interior_result.json" $ encode resultInterior
    putStrLn "Saved to results/interior_result.json"

    putStrLn "\n=== Exterior Penalty Method ==="
    resultExterior <- runExteriorProblem
    BL.writeFile "results/exterior_result.json" $ encode resultExterior
    putStrLn "Saved to results/exterior_result.json"

    putStrLn "\n=== Augmented Lagrangian Method ==="
    resultLagrange <- runLagrangianProblem
    BL.writeFile "results/lagrangian_result.json" $ encode resultLagrange
    putStrLn "Saved to results/lagrangian_result.json"

    -- rhoの初期値を変えて複数の結果を生成
    putStrLn "\n=== Interior Penalty Method with Different Rho Values ==="
    runInteriorWithDifferentRhos [1.0, 10.0, 100.0]

    putStrLn "\n=== Exterior Penalty Method with Different Rho Values ==="
    runExteriorWithDifferentRhos [0.01, 0.1, 1.0]

-- 内点法の実行
runInteriorProblem :: IO OptimizationResult
runInteriorProblem = do
    gen <- getStdGen
    let dims = 1
        config = defaultConfig
        initialValue = vector [3 + x | x <- take dims (randoms gen :: [Double])]
        rho0 = 10.0
        gammaParam = 5.0
        maxOuter = 20

    putStrLn $ "Initial value: " ++ show initialValue
    let result = solveInterior config initialValue rho0 gammaParam maxOuter
    putStrLn $ "Solution: " ++ show (solution result)
    putStrLn $ "Final value: " ++ show (finalValue result)

    return result

-- 異なるrho初期値で内点法を実行
runInteriorWithDifferentRhos :: [Double] -> IO ()
runInteriorWithDifferentRhos rhos = do
    gen <- getStdGen
    let dims = 1
        config = defaultConfig
        initialValue = vector [3 + x | x <- take dims (randoms gen :: [Double])]
        gammaParam = 5.0
        maxOuter = 20

    putStrLn $ "Initial value: " ++ show initialValue

    mapM_
        ( \rho0 -> do
            putStrLn $ "\n--- Rho0 = " ++ show rho0 ++ " ---"
            let result = solveInterior config initialValue rho0 gammaParam maxOuter
                filename = "results/interior_rho" ++ show rho0 ++ "_result.json"
            BL.writeFile filename $ encode result
            putStrLn $ "Solution: " ++ show (solution result)
            putStrLn $ "Final value: " ++ show (finalValue result)
            putStrLn $ "Saved to " ++ filename
        )
        rhos

-- 外点法の実行
runExteriorProblem :: IO OptimizationResult
runExteriorProblem = do
    gen <- getStdGen
    let dims = 1
        config = defaultConfig
        initialValue = vector [-(3 + x) | x <- take dims (randoms gen :: [Double])]
        rho0 = 0.1
        gammaParam = 5.0
        maxOuter = 20

    putStrLn $ "Initial value: " ++ show initialValue
    let result = solveExterior config initialValue rho0 gammaParam maxOuter
    putStrLn $ "Solution: " ++ show (solution result)
    putStrLn $ "Final value: " ++ show (finalValue result)

    return result

-- 異なるrho初期値で外点法を実行
runExteriorWithDifferentRhos :: [Double] -> IO ()
runExteriorWithDifferentRhos rhos = do
    gen <- getStdGen
    let dims = 1
        config = defaultConfig
        initialValue = vector [-(3 + x) | x <- take dims (randoms gen :: [Double])]
        gammaParam = 5.0
        maxOuter = 20

    putStrLn $ "Initial value: " ++ show initialValue

    mapM_
        ( \rho0 -> do
            putStrLn $ "\n--- Rho0 = " ++ show rho0 ++ " ---"
            let result = solveExterior config initialValue rho0 gammaParam maxOuter
                filename = "results/exterior_rho" ++ show rho0 ++ "_result.json"
            BL.writeFile filename $ encode result
            putStrLn $ "Solution: " ++ show (solution result)
            putStrLn $ "Final value: " ++ show (finalValue result)
            putStrLn $ "Saved to " ++ filename
        )
        rhos

-- 拡張ラグランジュ法の実行
runLagrangianProblem :: IO OptimizationResult
runLagrangianProblem = do
    gen <- getStdGen
    let dims = 10
        config = defaultConfig{maxIterations = 200}
        initialValue = vector $ take dims (randoms gen :: [Double])

    putStrLn $ "Initial value: " ++ show initialValue
    let result = solveAugmentedLagrangian config initialValue 20
    putStrLn $ "Solution: " ++ show (solution result)
    putStrLn $ "Final value: " ++ show (finalValue result)

    return result

-- スタンドアロンで特定の問題を解く
solveSpecificProblem :: ProblemType -> Int -> IO ()
solveSpecificProblem Interior dims = do
    gen <- getStdGen
    let config = defaultConfig
        initialValue = vector [3 + x | x <- take dims (randoms gen :: [Double])]
        result = solveInterior config initialValue 10.0 5.0 20
    BL.putStrLn $ encode result
solveSpecificProblem Exterior dims = do
    gen <- getStdGen
    let config = defaultConfig
        initialValue = vector [-(3 + x) | x <- take dims (randoms gen :: [Double])]
        result = solveExterior config initialValue 0.1 5.0 20
    BL.putStrLn $ encode result
solveSpecificProblem AugmentedLag dims = do
    gen <- getStdGen
    let config = defaultConfig{maxIterations = 200}
        initialValue = vector $ take dims (randoms gen :: [Double])
        result = solveAugmentedLagrangian config initialValue 20
    BL.putStrLn $ encode result
