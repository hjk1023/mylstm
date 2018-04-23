package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class DWeight{
    int data_dim;
    int hidden_dim;
    DWeight_whwx dwhwx_i;
    DWeight_whwx dwhwx_f;
    DWeight_whwx dwhwx_o;
    DWeight_whwx dwhwx_a;
    DWeight_wy dwy;

    // 初始化权重偏导
    public DWeight(int data_dim, int hidden_dim) {
        this.data_dim = data_dim;
        this.hidden_dim = hidden_dim;

        this.dwhwx_i = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_f = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_o = new DWeight_whwx(data_dim, hidden_dim);
        this.dwhwx_a = new DWeight_whwx(data_dim, hidden_dim);
        this.dwy = new DWeight_wy(data_dim, hidden_dim);
    }

    // 计算各权重矩阵的偏导数
    public void update(Matrix di, Matrix df, Matrix da, Matrix doo, Matrix hss, Matrix x){
        this.dwhwx_i.cal_grad(di, hss, x);
        this.dwhwx_f.cal_grad(df, hss, x);
        this.dwhwx_a.cal_grad(da, hss, x);
        this.dwhwx_o.cal_grad(doo, hss, x);
    }

    // 加法
    public DWeight add(DWeight dWeight2){
        DWeight dWeight_add = new DWeight(this.data_dim, this.hidden_dim);
        dWeight_add.dwhwx_i = this.dwhwx_i.add(dWeight2.dwhwx_i);
        dWeight_add.dwhwx_f = this.dwhwx_f.add(dWeight2.dwhwx_f);
        dWeight_add.dwhwx_o = this.dwhwx_o.add(dWeight2.dwhwx_o);
        dWeight_add.dwhwx_a = this.dwhwx_a.add(dWeight2.dwhwx_a);
        dWeight_add.dwy = this.dwy.add(dWeight2.dwy);
        return dWeight_add;
    }

}


class DWeight_whwx {
    int data_dim;
    int hidden_dim;
    Matrix dwh, dwx, db;


    public DWeight_whwx(int data_dim, int hidden_dim) {
        this.data_dim = data_dim;
        this.hidden_dim = hidden_dim;
        this.dwh = new Matrix(hidden_dim, hidden_dim, 0.0);
        this.dwx = new Matrix(hidden_dim, data_dim, 0.0);
        this.db = new Matrix(hidden_dim, 1, 0.0);
    }

    public void cal_grad(Matrix delta, Matrix ht_pre, Matrix x){
        this.dwh.plusEquals(delta.times(ht_pre.transpose()));
        this.dwx.plusEquals(delta.times(x.transpose()));
        this.db.plusEquals(delta);
    }

    // add
    public DWeight_whwx add(DWeight_whwx dWeight_whwx2){
        DWeight_whwx dWeight_whwx_add = new DWeight_whwx(this.data_dim ,this.hidden_dim);
        dWeight_whwx_add.dwh = this.dwh.plus(dWeight_whwx2.dwh);
        dWeight_whwx_add.dwx = this.dwx.plus(dWeight_whwx2.dwx);
        dWeight_whwx_add.db = this.db.plus(dWeight_whwx2.db);
        return dWeight_whwx_add;
    }
}


class DWeight_wy{
    int data_dim;
    int hidden_dim;
    Matrix dw, db;

    public DWeight_wy(int data_dim, int hidden_dim) {
        this.data_dim = data_dim;
        this.hidden_dim = hidden_dim;
        this.dw = new Matrix(data_dim, hidden_dim, 0.0);
        this.db = new Matrix(data_dim, 1, 0.0);
    }

    // add
    public DWeight_wy add(DWeight_wy dWeight_wy2){
        DWeight_wy dWeight_wy_add = new DWeight_wy(this.data_dim, this.hidden_dim);
        dWeight_wy_add.dw = this.dw.plus(dWeight_wy2.dw);
        dWeight_wy_add.db = this.db.plus(dWeight_wy2.db);
        return dWeight_wy_add;
    }
}
