package com.hjk.lstm;

import com.hjk.matrix.Matrix;



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

    // back propagate through time， using one sample
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
            // t_state, t for state
            int t_state = t + 1;
            dWeight.dwy.dw.plusEquals(delta_yz[t].times(stats.hss[t_state].transpose()));
            dWeight.dwy.db.plusEquals(delta_yz[t]);

            // 目标函数对隐藏状态的偏导数
            Matrix delta_ht = this.weight.wy.w.transpose().times(delta_yz[t]);
            // 各个门及状态单元的偏导
            Matrix delta_ot = delta_ht.arrayTimes(stats.css[t_state]);
            // delta_ct += delta_ht * ot * (1 - tanh(ct)^2)
            delta_ct.plusEquals(delta_ht.arrayTimes(stats.oss[t_state]).arrayTimes(_tanh_derivate(stats.css[t_state])));
            Matrix delta_it = delta_ct.arrayTimes(stats.ass[t_state]);
            Matrix delta_ft = delta_ct.arrayTimes(stats.css[t_state-1]);
            Matrix delta_at = delta_ct.arrayTimes(stats.iss[t_state]);

            Matrix delta_at_net = delta_at.arrayTimes(_tanh_derivate(stats.ass[t_state]));
            Matrix delta_it_net = delta_it.arrayTimes(_sigmoid_derivate(stats.iss[t_state]));
            Matrix delta_ft_net = delta_ft.arrayTimes(_sigmoid_derivate(stats.fss[t_state]));
            Matrix delta_ot_net = delta_ot.arrayTimes(stats.oss[t_state]);

            dWeight.update(delta_it_net, delta_ft_net, delta_at_net, delta_ot_net, stats.hss[t_state-1], x[t]);
        }
        return dWeight;
    }

    public void sgd_step(Matrix[] x, Matrix y, double lr){
        DWeight dWeight = this._bptt(x, y);
        this.weight.update_hx(dWeight, lr);
        this.weight.update_y(dWeight, lr);
    }

    public void train(Matrix[][] X, Matrix[] Y, double lr, int epochs){
        double[] losses = new double[epochs];
        long startime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++){
            for (int j = 0; j < X.length; j++){
                this.sgd_step(X[j], Y[j], lr);
            }
            losses[i] = this.loss(X, Y);
            if (i > 0 && losses[i] > losses[i-1]){
                lr *= 0.7;
                System.out.printf("lr decrease to: %f\n", lr);
            }
            System.out.printf("epoch %d: ; loss = %f\n", i,losses[i]*1000);
        }
        long endtime = System.currentTimeMillis();
        System.out.printf("training time: %.2f s\n", (endtime-startime)/1000.0);
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
