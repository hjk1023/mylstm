package com.hjk.lstm;

import com.hjk.matrix.Matrix;

public class Stats {
    Matrix[] iss, fss, oss, ass, hss, css, ys;

    public Stats(int step, int data_dim, int hidden_dim) {
        // input gate
        this.iss = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            iss[i] = new Matrix(hidden_dim, 1);
        }
        // forget gate
        this.fss = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            fss[i] = new Matrix(hidden_dim, 1);
        }
        // output gate
        this.oss = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            oss[i] = new Matrix(hidden_dim, 1);
        }
        // current input state
        this.ass = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            ass[i] = new Matrix(hidden_dim, 1);
        }
        // hidden gate
        this.hss = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            hss[i] = new Matrix(hidden_dim, 1);
        }
        // cell gate
        this.css = new Matrix[step + 1];
        for (int i = 0; i < step; i++){
            css[i] = new Matrix(hidden_dim, 1);
        }
        // output value
        this.ys = new Matrix[step];
        for (int i = 0; i < step; i++){
            ys[i] = new Matrix(data_dim, 1);
        }



    }
}
