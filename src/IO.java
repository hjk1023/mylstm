import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class IO {
    public static int[][]  read(String path)
    {
        int[][] data = new int[18][];
        try {
            BufferedReader br =
                    new BufferedReader(new FileReader(path));

            String line = br.readLine();


            int index = 0;
            while (line != null) {
                String[] strings = line.split(",");
                int len = strings.length - 1;
                int[] x = new int[len];
                for (int i = 0; i < len; i++){
                    x[i] = Integer.parseInt(strings[i + 1]);
                }
                data[index++] = x;
                //System.out.println(Arrays.toString(x));
                line = br.readLine();
            }
            br.close();
        }
        catch(IOException e) {
            System.out.println(e.getMessage());
        }
        return data;

    }

}



