import com.hjk.lstm.Network;
import com.hjk.matrix.Matrix;
import com.hjk.data.Data;


//import com.hjk.lstm.Lstm;
public class Demo {


    public static void main(String[] args)
    {
        // 一个时间点：Matrix, shape = (data_dim,1)
        // 一个样本：Matrix[time_steps]
        // 全部样本：Matrix[nb_samples][time_steps]


        // 虚构数据
        int[] data0 = new int[60];
        for (int i = 0; i < 60; i++){
            data0[i] = i + 1;
        }

        int[] data = {3,  9,  3,  0,  0, 11,  0, 22,  2,  3,  0,  0,  4, 17,  0,  4,  8,
                1,  2,  0,  6, 11,  2, 12, 20,  8,  0, 8,  2,  0, 17};

        // 间隔天数
        int day_interval = 7;
        // 预测天数
        int day_pre = 7;

        int pre_vm = predictor_lstm(data, day_interval, day_pre);




    }


    public static int predictor_lstm(int[] data, int day_interval, int day_pre){
        /**
         * input: data:         一种flavor的历史请求数量
         *        day_interval: 间隔天数
         *        day_pre:      预测天数
         * output:pre_vm:       预测期间的虚拟机数量
         */

        // 构造训练样本， data: 为一种flavor的历史请求数量
        Data train = new Data(data, 14);
        Matrix[][] train_x = train.getTrain_x();
        Matrix[] train_y = train.getTrain_y();

        // 初始化LSTM
        Network lstm = new Network(1, 4);

        // 训练LSTM， lr：学习速率 epochs: 迭代次数
        lstm.train(train_x, train_y, 3, 0.3, 100);


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
            System.out.println(value);
            //value = Math.round(value);

            if (value < 0){
                value = 0;
            }
            sum += value;
        }
        sum = Math.round(sum);
        pre_vm = (int) sum;
        System.out.printf("predict vm numbers: %d", pre_vm);
        return pre_vm;
    }

}

