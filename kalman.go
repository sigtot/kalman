package kalman

import (
    "gonum.org/v1/gonum/mat"
)

const MAX_OUTPUTS = 200

type Filter struct {
    // System matrices
    a mat.Matrix
    b mat.Matrix
    c mat.Matrix
    d mat.Matrix
    g mat.Matrix
    h mat.Matrix

    // Tuning matrices
    r mat.Matrix
    q mat.Matrix

    // Kalman filter matrices
    aPriErrCovs  []mat.Matrix
    aPostErrCovs []mat.Matrix
    kalmanGains  []mat.Matrix

    // State, input and output vectors
    aPriStateEsts  []mat.Vector
    aPostStateEsts []mat.Vector
    aPriOutputEsts []mat.Vector
    input          mat.Vector // This implementation is limited to only constant inputs
    outputs        []mat.Vector
}

func NewFilter(A mat.Matrix,
               B mat.Matrix,
               C mat.Matrix,
               D mat.Matrix,
               H mat.Matrix,
               G mat.Matrix,
               R mat.Matrix,
               Q mat.Matrix,
               aPriErrorCovInit mat.Matrix,
               aPriStateEstInit mat.Vector,
               input mat.Vector,
               outputInit mat.Vector) Filter {
    f := Filter{}

    f.a = A
    f.b = B
    f.c = C
    f.d = D
    f.h = H
    f.g = G
    f.r = R
    f.q = Q

    f.aPriErrCovs = append(f.aPriErrCovs, aPriErrorCovInit)
    f.aPriStateEsts = append(f.aPriStateEsts, aPriStateEstInit)
    f.input = input
    f.outputs = append(f.outputs, outputInit)

    return f
}

func (f *Filter) APriErrCov(k int) mat.Matrix {
    if k < len(f.aPriErrCovs) && k < len(f.outputs) {
        return f.aPriErrCovs[k]
    }

    var a mat.Dense
    a.Product(f.a, f.APostErrCov(k-1), f.a.T())

    var b mat.Dense
    b.Product(f.g, f.q, f.g.T())

    var aPriErrCov mat.Dense
    aPriErrCov.Add(&a, &b)

    if k < len(f.outputs) {
        f.aPriErrCovs = append(f.aPriErrCovs, &aPriErrCov)
    }
    return &aPriErrCov
}

func (f *Filter) APostErrCov(k int) mat.Matrix {
    if k < len(f.aPostErrCovs) && k < len(f.outputs) {
        return f.aPostErrCovs[k]
    }

    var KkC mat.Dense
    KkC.Mul(f.KalmanGain(k), f.c)

    dim, _ := KkC.Dims()
    I := mat.NewDiagonal(dim, nil)
    for i := 0; i < dim; i++ {
        I.SetDiag(i, 1)
    }

    var a mat.Dense
    a.Sub(I, &KkC)

    var b mat.Dense
    b.Product(&a, f.APriErrCov(k), a.T())

    var c mat.Dense
    c.Product(f.KalmanGain(k), f.h, f.r, f.h.T(), f.KalmanGain(k).T())

    var aPostErrCov mat.Dense
    aPostErrCov.Add(&b, &c)

    if k < len(f.outputs) {
        f.aPostErrCovs = append(f.aPostErrCovs, &aPostErrCov)
    }
    return &aPostErrCov
}

func (f *Filter) KalmanGain(k int) mat.Matrix {
    if k < len(f.kalmanGains) && k < len(f.outputs) {
        return f.kalmanGains[k]
    }

    var PkCT mat.Dense
    PkCT.Mul(f.APriErrCov(k), f.c.T())

    var CPkCT mat.Dense
    CPkCT.Mul(f.c, &PkCT)

    var HRHT mat.Dense
    HRHT.Product(f.h, f.r, f.h.T())

    var a mat.Dense
    a.Add(&CPkCT, &HRHT)

    var aInv mat.Dense
    aInv.Inverse(&a)

    var kalmanGain mat.Dense
    kalmanGain.Mul(&PkCT, &aInv)

    if k < len(f.outputs) {
        f.kalmanGains = append(f.kalmanGains, &kalmanGain)
    }
    return &kalmanGain
}

func (f *Filter) APriStateEst(k int) mat.Vector {
    if k < len(f.aPriStateEsts) && k < len(f.outputs) {
        return f.aPriStateEsts[k]
    }

    var a mat.VecDense
    a.MulVec(f.a, f.APostStateEst(k-1))

    var b mat.VecDense
    b.MulVec(f.b, f.input)

    var aPriStateEst mat.VecDense
    aPriStateEst.AddVec(&a, &b)

    if k < len(f.outputs) {
        f.aPriStateEsts = append(f.aPriStateEsts, &aPriStateEst)
    }
    return &aPriStateEst
}

func (f *Filter) APostStateEst(k int) mat.Vector {
    if k < len(f.aPostStateEsts) && k < len(f.outputs) {
        return f.aPostStateEsts[k]
    }

    if k >= len(f.outputs) {
        return f.APriStateEst(k)
    }

    if f.outputs[k] == nil {
        f.aPostStateEsts = append(f.aPostStateEsts, f.APriStateEst(k))
        return f.APriStateEst(k)
    }

    var estErr mat.VecDense
    estErr.SubVec(f.outputs[k], f.APriOutputEst(k))

    var b mat.VecDense
    b.MulVec(f.KalmanGain(k), &estErr)

    var aPostStateEst mat.VecDense
    aPostStateEst.AddVec(f.APriStateEst(k), &b)

    if k < len(f.outputs) {
        f.aPostStateEsts = append(f.aPostStateEsts, &aPostStateEst)
    }
    return &aPostStateEst
}

func (f *Filter) APriOutputEst(k int) mat.Vector {
    if k < len(f.aPostStateEsts) && k < len(f.outputs) {
        return f.aPriOutputEsts[k]
    }

    var a mat.VecDense
    a.MulVec(f.c, f.APriStateEst(k))

    var b mat.VecDense
    b.MulVec(f.d, f.input)

    var aPriOutputEst mat.VecDense
    aPriOutputEst.AddVec(&a, &b)

    if k < len(f.outputs) {
        f.aPriOutputEsts = append(f.aPriOutputEsts, &aPriOutputEst)
    }
    return &aPriOutputEst
}

func (f *Filter) AddOutput(output mat.Vector) {
    f.outputs = append(f.outputs, output)
    if len(f.outputs) > MAX_OUTPUTS {
        f.outputs = f.outputs[1:]
        f.aPriErrCovs = f.aPriErrCovs[1:]
        f.aPostErrCovs = f.aPostErrCovs[1:]
        f.kalmanGains = f.kalmanGains[1:]
        f.aPriStateEsts = f.aPriStateEsts[1:]
        f.aPostStateEsts = f.aPostStateEsts[1:]
        f.aPriOutputEsts = f.aPriOutputEsts[1:]
    }
}

func (f *Filter) CurrentK() int {
    return len(f.outputs) - 1
}
