package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class Adam {
    // Adam, m v t
    double betal1 = 0.9;
    double betal2 = 0.999;
    Matrix eps;
    int t = 0;
    Matrix m, v, mt, vt;

    public Adam(Matrix weight) {
        int rows = weight.getRowDimension();
        int cols = weight.getColumnDimension();
        this.m = new Matrix(rows, cols);
        this.v = new Matrix(rows, cols);
        this.eps = new Matrix(rows, cols, 0.00000001);
    }

    public Matrix update(Matrix dw){
        Matrix new_dw;
        // t++
        this.t = t + 1;

        // betal1*m + (1-betal1)*grad
        this.m = m.times(betal1).plus(dw.times(1-betal1));
        this.v = v.times(betal2).plus(dw.arrayTimes(dw).times(1-betal2));
        this.mt = m.times(1 / (1 - Math.pow(betal1, t)));
        this.vt = v.times(1 / (1 - Math.pow(betal2, t)));

        // mt/(vt^0.5 + eps)
        new_dw = mt.arrayRightDivide(vt.sqrt().plus(eps));
        return new_dw;
    }

}
