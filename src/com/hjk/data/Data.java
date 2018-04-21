package com.hjk.data;

import com.hjk.matrix.Matrix;

import java.util.Arrays;
import java.util.Collections;

public class Data {
    private Matrix[][] train_x;
    private Matrix[] train_y;

    private Matrix[] pre_x;

    // 虚拟机数量除以100
    private double threshold = 40;





    // 构造函数
    public Data(int[] data, int step) {
        double[] data_norm = this.normalize(data);
        this.create_pre_x(data_norm, step);
        this.create_sample(data_norm, step);
    }

    public Matrix[][] getTrain_x() {
        return train_x;
    }

    public Matrix[] getTrain_y() {
        return train_y;
    }

    public Matrix[] getPre_x() {
        return pre_x;
    }

    public Matrix[] nextPre_x(Matrix pre_y){
        int len = this.pre_x.length;
        for (int i = 0; i < len - 1; i++){
            this.pre_x[i] = this.pre_x[i + 1];
        }
        this.pre_x[len-1] = pre_y;
        return this.pre_x;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    private void create_pre_x(double[] data, int step){
        this.pre_x = new Matrix[step];
        for (int i = 0; i < step; i++){
            double[][] e = {{data[data.length - step + i]}};
            Matrix me = new Matrix(e);
            //me.print(6,4);
            this.pre_x[i] = me;
        }
    }

    private void create_sample(double[] data, int step){
        int nb_sample = data.length - step;
        this.train_x = new Matrix[nb_sample][step];
        this.train_y = new Matrix[nb_sample];
        for (int i = 0; i < nb_sample; i++){
            for (int j = 0; j < step; j++) {
                double[][] ex = {{data[i + j]}};
                //System.out.printf("%f ", data[i+j]);
                Matrix mex = new Matrix(ex);
                this.train_x[i][j] = mex;
            }
            double[][] ey = {{data[i + step]}};
            Matrix mey = new Matrix(ey);
            this.train_y[i] = mey;
            //System.out.println();
            //System.out.println(data[i + step]);
        }
    }

    public double[] normalize(int[] data){
        double[] data_norm = new double[data.length];
        for (int i = 0; i < data.length; i++){
            data_norm[i] = data[i] / this.threshold;
        }
        return data_norm;
    }

    public double normalize_re(double d){
        return d*this.threshold;
    }



}
