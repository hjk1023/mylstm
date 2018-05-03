import com.hjk.data.Data;
import com.hjk.lstm.Network;
import com.hjk.matrix.Matrix;

import java.util.Arrays;

public class Predictor {

    public static void run()
    {
        // 一个时间点：Matrix, shape = (data_dim,1)
        // 一个样本：Matrix[time_steps]
        // 全部样本：Matrix[nb_samples][time_steps]


        int[][] train, test;
        int result[] = new int[18];
        int result_true[] = new int[18];
        String path = "C:\\Users\\huangjk\\Desktop\\testdata\\train.csv";
        String tpath = "C:\\Users\\huangjk\\Desktop\\testdata\\test.csv";
        train = IO.read(path);
        test = IO.read(tpath);
        train = Denoise.deleteAbnormalData(train);

        // 间隔天数
        int day_interval = 1;
        // 预测天数
        int day_pre = 7;


        int N = train.length;
        double err = 0;
        double y_square = 0;
        double yhat_square = 0;
        double score;
        for (int i = 0; i < train.length; i++) {
            int[] data = train[i];
            int[] tdata = test[i];
            int pre_vm = predictor_lstm(data, day_interval, day_pre);
            result[i] = pre_vm;
            int pre_true = 0;
            for (int j = 0; j < tdata.length; j++){
                pre_true += tdata[j];
            }
            result_true[i] = pre_true;
            err += Math.pow((pre_true-pre_vm),2);
            y_square += Math.pow(pre_true,2 );
            yhat_square += Math.pow(pre_vm, 2);

        }
        err  = Math.sqrt(err / N);
        y_square = Math.sqrt(y_square / N);
        yhat_square = Math.sqrt(yhat_square / N);
        score = 1 - err / (y_square + yhat_square);


        System.out.println(Arrays.toString(result));
        System.out.println(Arrays.toString(result_true));
        System.out.println(score);

    }



    public static int predictor_lstm(int[] data, int day_interval, int day_pre){
        /**
         * input: data:         一种flavor的历史请求数量
         *        day_interval: 间隔天数
         *        day_pre:      预测天数
         * output:pre_vm:       预测期间的虚拟机数量
         */

        // 构造训练样本， data: 为一种flavor的历史请求数量
        int step = day_interval + day_pre;
        Data train = new Data(data, step);
        Matrix[][] train_x = train.getTrain_x();
        Matrix[] train_y = train.getTrain_y();

        // 初始化LSTM
        int epochs;
        if (data.length > 70){
            epochs =  50;
        }
        else {
            epochs = 60;
        }
        Network lstm = new Network(1, 8);

        // 训练LSTM， lr：学习速率 epochs: 迭代次数
        lstm.train(train_x, train_y, 8, 0.1, epochs);




        // 滚动预测
        Matrix[] pre_arr = new Matrix[day_interval + day_pre];
        Matrix[] pre_x = train.getPre_x();
        Matrix pre_y = lstm.predict(pre_x);
        pre_arr[0] = pre_y;
        for (int i = 1; i < day_interval + day_pre; i++) {
            pre_x = train.nextPre_x(pre_y);
            pre_y = lstm.predict(pre_x);
            pre_arr[i] = pre_y;
        }

        // pre_vm， 预测区间的虚拟机数量
        int pre_vm = 0;
        double sum = 0;
        for (int i = day_interval; i < day_interval + day_pre; i++){
            double value = pre_arr[i].get(0,0);
            value = train.normalize_re(value);
            //System.out.println(value);
            //value = Math.round(value);

            if (value < 0){
                value = 0;
            }
            sum += value;
        }
        sum = Math.round(sum);
        pre_vm = (int) sum;
        //System.out.printf("predict vm numbers: %d\n", pre_vm);
        return pre_vm;
    }
}
