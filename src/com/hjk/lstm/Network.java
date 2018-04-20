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

}
