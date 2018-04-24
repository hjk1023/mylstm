import java.math.BigDecimal;
import java.math.RoundingMode;

public class Denoise {
    public static int[][] deleteAbnormalData(int[][] data) {
        //平均数(不包括0的数据)
        double[] avg = computeAvgExcludeZero(data);
        //标准差(不包括0的数据)
        double[] sdv = computeSdvExcludeZero(data, avg);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                //只考虑大于平均值的异常点
                if (sdv[i] != 0 && data[i][j] - avg[i] >= 3 * sdv[i] ) {
                    //data[i][j] = 0;
                    //BigDecimal b = new  BigDecimal(avg[i]);
                    //data[i][j] = b.setScale(0, RoundingMode.HALF_UP).intValue();
                    if (j > 0 && j < data[i].length - 1){
                        data[i][j] = (int) (data[i][j-1] + data[i][j+1])/2;
                    }
                    if (j == 0){
                        data[i][j] = data[i][j+1];
                    }
                    if (j == data.length - 1){
                        data[i][j] = data[i][j-1];
                    }
                    //System.out.println(i + " " + j + " " + data[i][j]);
                }
            }
        }
        return data;
    }

    /**
     * 计算平均数(不包括0的数据)

     */
    public static double[] computeAvgExcludeZero(int[][] data) {
        int numFlavor = data.length;
        double[] avg = new double[numFlavor];
        for (int i = 0; i < numFlavor; i++) {
            int days = 0;
            for (int j = 0; j < data[i].length; j++) {
                if (data[i][j] != 0) {
                    avg[i] += data[i][j];
                    days++;
                }
            }
            if (days != 0) {
                avg[i] /= days;
            }
        }
        return avg;
    }

    /**
     * 计算标准差(不包括0元素)
     * @param data
     * @param avg
     * @return
     */
    public static double[] computeSdvExcludeZero(int[][] data, double[] avg) {
        int numFlavor = data.length;
        double[] sdv = new double[numFlavor];
        for (int i = 0; i < numFlavor; i++) {
            if (avg[i] == 0) {
                sdv[i] = 0;
            }else {
                int days = 0;
                for (int j = 0; j < data[i].length; j++) {
                    if (data[i][j] != 0) {
                        sdv[i] += ((data[i][j] - avg[i]) * (data[i][j] - avg[i]));
                        days++;
                    }
                }
                sdv[i] = Math.sqrt(sdv[i] / days);
            }
        }
        return sdv;
    }

}
