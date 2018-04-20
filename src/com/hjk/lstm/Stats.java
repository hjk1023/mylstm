package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class Stats {

    Matrix[] iss, fss, oss, ass, hss, css, ys;
    int step, current_step;

    public Stats(int step, int data_dim, int hidden_dim) {
        this.step = step;
        this.current_step = 1;
        // input gate
        this.iss = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            iss[i] = new Matrix(hidden_dim, 1);
        }
        // forget gate
        this.fss = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            fss[i] = new Matrix(hidden_dim, 1);
        }
        // output gate
        this.oss = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            oss[i] = new Matrix(hidden_dim, 1);
        }
        // current input state
        this.ass = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            ass[i] = new Matrix(hidden_dim, 1);
        }
        // hidden gate
        this.hss = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            hss[i] = new Matrix(hidden_dim, 1);
        }
        // cell gate
        this.css = new Matrix[step + 1];
        for (int i = 0; i < step + 1; i++){
            css[i] = new Matrix(hidden_dim, 1);
        }
        // output value
        this.ys = new Matrix[step];
        for (int i = 0; i < step; i++){
            ys[i] = new Matrix(data_dim, 1);
        }



    }

    public void update(Weight weight, Matrix x){
        // 前一时刻隐藏状态
        Matrix ht_pre = this.hss[current_step - 1];

        // input gate
        this.iss[current_step] = _cal_gate(weight.whwx_i, ht_pre, x, "sigmoid");
        // forget gate
        this.fss[current_step] = _cal_gate(weight.whwx_f, ht_pre, x, "sigmoid");
        // output gate
        this.oss[current_step] = _cal_gate(weight.whwx_o, ht_pre, x, "sigmoid");
        // current input state
        this.ass[current_step] = _cal_gate(weight.whwx_a, ht_pre, x, "tanh");
        // cell state, ct = ft * ct_pre + it * at
        Matrix fc = fss[current_step].arrayTimes(css[current_step - 1]);
        Matrix ia = iss[current_step].arrayTimes(ass[current_step]);
        this.css[current_step] = fc.plus(ia);
        // hidden state, ht = ot * tanh(ct)
        Matrix tc = Matrix.tanh(css[current_step]);
        this.hss[current_step] = oss[current_step].arrayTimes(tc);
        // output value, yt = sigmoid(wy.dot(ht) + by)
        Matrix yz = weight.wy.w.times(hss[current_step]).plus(weight.wy.b);
        ys[current_step-1] = Matrix.sigmoid(yz);
        current_step ++;
    }

    private Matrix _cal_gate(Weight_whwx  whwx, Matrix ht_pre, Matrix x, String activation){
        Matrix zh = whwx.wh.times(ht_pre);
        Matrix zx = whwx.wx.times(x);
        Matrix z = zh.plus(zx).plus(whwx.b);
        Matrix a = z;
        if (activation == "sigmoid"){
            a = Matrix.sigmoid(z);
        }
        else if (activation == "tanh"){
            a = Matrix.tanh(z);
        }
        else {
            throw new IllegalArgumentException("activation fun not implemented!");
        }
        return a;
    }

}

