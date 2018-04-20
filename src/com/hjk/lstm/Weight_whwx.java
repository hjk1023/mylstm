package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class Weight_whwx {
    Matrix wh, wx, b;


    public Weight_whwx(int data_dim, int hidden_dim) {
        this.wh = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), hidden_dim, hidden_dim);
        this.wx = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, data_dim);
        this.b = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, 1);


    }
}


class Weight_wy{
    Matrix wy, by;

    public Weight_wy(int data_dim, int hidden_dim) {
        this.wy = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, hidden_dim);
        this.by = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, 1);
    }
}