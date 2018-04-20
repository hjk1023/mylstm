package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class Weight{
    Weight_whwx whwx_i;
    Weight_whwx whwx_f;
    Weight_whwx whwx_o;
    Weight_whwx whwx_a;
    Weight_wy wy;

    public Weight(int data_dim, int hidden_dim) {
        this.whwx_i = new Weight_whwx(data_dim, hidden_dim);
        this.whwx_f = new Weight_whwx(data_dim, hidden_dim);
        this.whwx_o = new Weight_whwx(data_dim, hidden_dim);
        this.whwx_a = new Weight_whwx(data_dim, hidden_dim);
        this.wy = new Weight_wy(data_dim, hidden_dim);
    }
}


class Weight_whwx {
    Matrix wh, wx, b;


    public Weight_whwx(int data_dim, int hidden_dim) {
        this.wh = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), hidden_dim, hidden_dim);
        this.wx = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, data_dim);
        this.b = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, 1);


    }
}


class Weight_wy{
    Matrix w, b;

    public Weight_wy(int data_dim, int hidden_dim) {
        this.w = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, hidden_dim);
        this.b = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, 1);
    }
}