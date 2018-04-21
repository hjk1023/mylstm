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

    public void update_hx(DWeight dWeight, double lr){
        this.whwx_i.update(dWeight.dwhwx_i, lr);
        this.whwx_f.update(dWeight.dwhwx_f, lr);
        this.whwx_a.update(dWeight.dwhwx_a, lr);
        this.whwx_o.update(dWeight.dwhwx_o, lr);
    }
    public void update_y(DWeight dWeight, double lr){
        this.wy.update(dWeight.dwy, lr);
    }
}


class Weight_whwx {
    Matrix wh, wx, b;

    public Weight_whwx(int data_dim, int hidden_dim) {
        this.wh = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), hidden_dim, hidden_dim);
        this.wx = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, data_dim);
        this.b = Matrix.uniform(-Math.sqrt(1.0/data_dim), Math.sqrt(1.0/data_dim), hidden_dim, 1);


    }

    public void update(DWeight_whwx dWeight_whwx, double lr){
        this.wh.minusEquals(dWeight_whwx.dwh.times(lr));
        this.wx.minusEquals(dWeight_whwx.dwx.times(lr));
        this.b.minusEquals(dWeight_whwx.db.times(lr));
    }
}


class Weight_wy{
    Matrix w, b;

    public Weight_wy(int data_dim, int hidden_dim) {
        this.w = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, hidden_dim);
        this.b = Matrix.uniform(-Math.sqrt(1.0/hidden_dim), Math.sqrt(1.0/hidden_dim), data_dim, 1);
    }

    public void update(DWeight_wy dWeight_wy, double lr){
        this.w.minusEquals(dWeight_wy.dw.times(lr));
        this.b.minusEquals(dWeight_wy.db.times(lr));
    }
}