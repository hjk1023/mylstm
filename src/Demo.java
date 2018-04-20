import com.hjk.lstm.Network;
import com.hjk.matrix.Matrix;

//import com.hjk.lstm.Lstm;
public class Demo {


    public static void main(String[] args)
    {
        double[][] array = {{1},{3}};
        Matrix m1 = new Matrix(array);
        m1.print(3,2);
        Network lstm = new Network(2, 4);
        Matrix[] x = {m1, m1, m1};
        Matrix[][] X = {x, x, x, x};
        double[][] y_array = {{0.5},{0.5}};
        Matrix y = new Matrix(y_array);
        Matrix[] Y = {y,y,y,y};
        Matrix pre_y = lstm.predict(x);
        pre_y.print(6, 4);
        System.out.println(lstm.loss(X,Y));
        lstm._bptt(x,y);


    }





}
