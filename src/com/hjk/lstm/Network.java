package com.hjk.lstm;

public class Network {
    private int data_dim;
    private int hidden_dim;

    public Network(int data_dim, int hidden_dim) {
        // data_dim: 特征向量维度; hidden_dim: 隐单元维度
        this.data_dim = data_dim;
        this.hidden_dim = hidden_dim;
        // 初始化权重向量
        Weight_whwx whwx_i = new Weight_whwx(data_dim, hidden_dim);
        Weight_whwx whwx_f = new Weight_whwx(data_dim, hidden_dim);
        Weight_whwx whwx_o = new Weight_whwx(data_dim, hidden_dim);
        Weight_whwx whwx_a = new Weight_whwx(data_dim, hidden_dim);
        Weight_wy wy = new Weight_wy(data_dim, hidden_dim);

        // 初始化各个状态向量
        Stats stats;




    }
}
