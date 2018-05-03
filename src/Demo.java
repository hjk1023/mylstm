import com.hjk.lstm.Network;
import com.hjk.matrix.Matrix;
import com.hjk.data.Data;

import java.util.Arrays;


//import com.hjk.lstm.Lstm;
public class Demo {


    public static void main(String[] args){
        int step = 7;
        int[] data = new int[600];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }

        Data train = new Data(data, step);
        Matrix[][] train_x = train.getTrain_x();
        Matrix[] train_y = train.getTrain_y();

        // Initialize LSTM
        int epochs = 30;
        Network lstm = new Network(1, 8);

        // 训练LSTM， lr：学习速率 epochs: 迭代次数
        lstm.train(train_x, train_y, 64, 0.05, epochs);
    }


}

