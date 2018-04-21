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

    // 更新各权重矩阵的偏导数
    public void update(Matrix di, Matrix df, Matrix da, Matrix doo, Matrix hss, Matrix x){
        this.dwhwx_i.cal_grad(di, hss, x);
        this.dwhwx_f.cal_grad(df, hss, x);
        this.dwhwx_a.cal_grad(da, hss, x);
        this.dwhwx_o.cal_grad(doo, hss, x);
    }

}


class DWeight_whwx {
    Matrix dwh, dwx, db;

    public DWeight_whwx(int data_dim, int hidden_dim) {
        this.dwh = new Matrix(hidden_dim, hidden_dim);
        this.dwx = new Matrix(hidden_dim, data_dim);
        this.db = new Matrix(hidden_dim, 1);
    }

    public void cal_grad(Matrix delta, Matrix ht_pre, Matrix x){
        this.dwh.plusEquals(delta.times(ht_pre.transpose()));
        this.dwx.plusEquals(delta.times(x.transpose()));
        this.db.plusEquals(delta);
    }
}


class DWeight_wy{
    Matrix dw, db;

    public DWeight_wy(int data_dim, int hidden_dim) {
        this.dw = new Matrix(data_dim, hidden_dim);
        this.db = new Matrix(data_dim, 1);
    }
}
