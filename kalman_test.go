package kalman

import (
    "fmt"
    "gonum.org/v1/gonum/mat"
    "testing"
)

func matPrint(X mat.Matrix) {
 fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
 fmt.Printf("%v\n", fa)
}

func TestNewFilter(t *testing.T) {
    A := mat.NewDense(2, 2, []float64{-1, 1, 0, -3})
    B := mat.NewDense(2, 1, []float64{0, 1})
    C := mat.NewDense(1, 2, []float64{1, 0})
    D := mat.NewDense(1, 1, []float64{4})
    H := mat.NewDense(1, 2, []float64{-1, 1})
    G := mat.NewDense(2, 1, []float64{1, 0})
    Q := mat.NewDense(1, 1, []float64{2})
    R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

    aPriErrCovInit := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
    aPriStateEstInit := mat.NewVecDense(2, []float64{0,0})
    input := mat.NewVecDense(1, []float64{1})
    outputInit := mat.NewVecDense(1, []float64{3})

    f := NewFilter(A, B, C, D, H, G, R, Q, aPriErrCovInit, aPriStateEstInit, input, outputInit)


    aPostStateEst := f.APostStateEst(0)
    fmt.Println("The matrix")
    matPrint(aPostStateEst)
    fmt.Println("should match")
    matPrint(mat.NewVecDense(2, []float64{-0.5, 0}))

    aPriStateEst:= f.APriStateEst(1)
    fmt.Println("The matrix")
    matPrint(aPriStateEst)
    fmt.Println("should match")
    matPrint(mat.NewVecDense(2, []float64{0.5, 1}))
}

func TestNewFilter2(t *testing.T) {
    A := mat.NewDense(4, 4, []float64{1, 0, 0, 0.1, 0, 1, 0.1, 0, 0, 0, 1, 0, 0, 0, 0, 1})
    B := mat.NewDense(4, 1, []float64{0, 0.005, 0.1, 0})
    C := mat.NewDense(2, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0})
    D := mat.NewDense(2, 1, []float64{0, 0})
    G := mat.NewDiagonal(4, []float64{0.2, 0.2, 0.1, 0.1})
    H := mat.NewDense(2, 2, []float64{0.1, 0.1, 0.2, 0.2})
    R := mat.NewDiagonal(2, []float64{0.2, 0.2})
    Q := mat.NewDiagonal(4, []float64{0.2, 0.2, 0.1, 0.1})

    aPriErrCovInit := mat.NewDense(4, 4, []float64{1, 0, 2, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 2, 0, 1})
    aPriStateEstInit := mat.NewVecDense(4, []float64{300, 200, 20, 20})
    input := mat.NewVecDense(1, []float64{-0.5})
    outputInit := mat.NewVecDense(2, []float64{300, 200})

    f := NewFilter(A, B, C, D, H, G, R, Q, aPriErrCovInit, aPriStateEstInit, input, outputInit)

    aPostStateEst := f.APostStateEst(20)
    matPrint(aPostStateEst)

    f.AddOutput(mat.NewVecDense(2, []float64{303, 202}))

    aPostStateEst = f.APostStateEst(20)
    matPrint(aPostStateEst)
}
