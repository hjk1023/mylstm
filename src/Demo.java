import Jama.Matrix;

public class Demo {
    public static void main(String[] args)
    {
        double[][] array = {{1,2,3},{4,5,6}};
        Matrix m1 = new Matrix(array);
        m1.print(3,2);
        Matrix m2 = m1.plus(m1);
        m2.print(6,4);
        Matrix m3 = new Matrix(3,3);
        m3.print(6,2);

    }
}
