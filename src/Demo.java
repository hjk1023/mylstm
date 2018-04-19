import com.hjk.matrix.Matrix;

//import com.hjk.lstm.Lstm;
public class Demo {
    public static void main(String[] args)
    {
        double[][] array = {{1,2,3},{4,5,6}};
        Matrix m1 = new Matrix(array);
        Matrix m2;
        m1.print(3,2);
        //Lstm lstm = new Lstm();
        m2 = Matrix.exp(m1);
        m2.print(10, 8);

        //m1 = lstm.tanh(m1);
        m1.print(10,8);




    }
}
