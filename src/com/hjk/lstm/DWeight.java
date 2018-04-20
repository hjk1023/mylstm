package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class DWeight{
    DWeight_whwx dwhwx_i;
    DWeight_whwx dwhwx_f;
    DWeight_whwx dwhwx_o;
    DWeight_whwx dwhwx_a;
    DWeight_wy dwy;

    public DWeight(int data_dim, int hidden_dim) {
        this.dwhwx_i = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_f = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_o = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_a = new DWeight_whwx(data_dim, hidden_dim);
        this.dwy = new DWeight_wy(data_dim, hidden_dim);
    }
}


class DWeight_whwx {
    Matrix dwh, dwx, db;


    public DWeight_whwx(int data_dim, int hidden_dim) {
        this.dwh = new Matrix(hidden_dim, hidden_dim);
        this.dwx = new Matrix(hidden_dim, data_dim);
        this.db = new Matrix(hidden_dim, 1);


    }
}


class DWeight_wy{
    Matrix dw, db;

    public DWeight_wy(int data_dim, int hidden_dim) {
        this.dw = new Matrix(data_dim, hidden_dim);
        this.db = new Matrix(data_dim, 1);
    }
}
