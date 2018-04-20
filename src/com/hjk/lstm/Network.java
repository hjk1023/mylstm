package com.hjk.lstm;

import com.hjk.matrix.Matrix;
import com.hjk.lstm.Weight;

public class Network {
    private int data_dim;
    private int hidden_dim;
    Weight weight;


    // constructor
    public Network(int data_dim, int hidden_dim) {
        // data_dim: 特征向量维度; hidden_dim: 隐单元维度
        this.data_dim = data_dim;
        this.hidden_dim = hidden_dim;

        // 初始化权重向量
        weight = new Weight(data_dim, hidden_dim);

    }

    // 前向传播， 单个训练样本[x1、x2...xt]
    private Stats _forward(Matrix[] x){
        int step = x.length;
        Stats stats = new Stats(step, this.data_dim, this.hidden_dim);
        for (int t = 0;t < step; t++){
            stats.update(weight, x[t]);

        }
        return stats;
    }

    // 预测
    public Matrix predict(Matrix[] x){
        Stats stats = this._forward(x);
        Matrix pre_y = stats.ys[x.length - 1];
        return pre_y;
    }

    // 计算损失， 多个样本Matrix[nb_samples][step]
    public double loss(Matrix[][] x, Matrix[] y){
        double cost = 0.0;
        for (int i = 0; i < x.length; i++){
            Matrix pre_y = this.predict(x[i]);
            cost += (pre_y.minus(y[i])).norm_mse();
        }
        return  cost/x.length;
    }

    public DWeight _bptt(Matrix[] x, Matrix y){
        int step = x.length;
        DWeight dWeight = new DWeight(this.data_dim, this.hidden_dim);
        // 初始化 delta_ct，后向传播过程中，此值需要累加
        Matrix delta_ct = new Matrix(this.hidden_dim, 1);
        Stats stats = this._forward(x);
        Matrix pre_y = this.predict(x);
        // 目标函数对 z 的偏导数 y = sigmoid(z)
        Matrix[] delta_yz = new Matrix[step];
        for (int i = 0; i < step; i++){
            delta_yz[i] = new Matrix(this.data_dim, 1);
        }
        // y*(1-y)
        Matrix sigmod_d = _sigmoid_derivate(pre_y);
        // (pre_y-y)*pre_y*(1-pre_y)
        delta_yz[step-1] = pre_y.minus(y).arrayTimes(sigmod_d);

        for (int t = step - 1; t >= 0; t-- ){
            // 输出层wy, by的偏导数 z = wy*h + by
            // todo check here
            dWeight.dwy.dw.plusEquals(delta_yz[t].times(stats.hss[t].transpose()));
            dWeight.dwy.db.plusEquals(delta_yz[t]);

            // 目标函数对隐藏状态的偏导数
            Matrix delta_ht = this.weight.wy.w.transpose().times(delta_yz[t]);





        }
        return dWeight;
    }


    // 工具函数
    // y = sigmoid(z), y' = y(1-y)
    private Matrix _sigmoid_derivate(Matrix y){
        Matrix one = new Matrix(y.getRowDimension(), y.getColumnDimension(), 1.0);
        return y.arrayTimes(one.minus(y));
    }

    // y = tanh(z) , y' = 1 - y^2
    private  Matrix _tanh_derivate(Matrix y){
        Matrix one = new Matrix(y.getRowDimension(), y.getColumnDimension(), 1.0);
        return one.minus(y.arrayTimes(y));
    }

}
