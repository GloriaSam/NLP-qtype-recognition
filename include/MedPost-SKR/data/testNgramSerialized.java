import java.io.FileNotFoundException;

import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class testNgramSerialized
{
    private static final int num_tags = 60;
    private static final int MAX_TAGS = 60;
    private static double count1[] = new double[MAX_TAGS];
    private static double count2[] = new double[MAX_TAGS * MAX_TAGS];
    private static double pr_tag[] = new double[MAX_TAGS];

    // PUBLIC METHODS  ------------------------------------------------

    // Our default constructor does nothing
    public testNgramSerialized() {}


    public static void main(String args[])
    {
        scangram();
    } // main


    // PRIVATE METHODS  ------------------------------------------------

    private static void scangram()
    {
        try
        {
           ObjectInputStream ois = 
                new ObjectInputStream(new FileInputStream("./ngramOne.serial"));
           double big_array[] = (double [])ois.readObject();
           ois.close();

           int pos = 0;
           for(int i = 0; i < num_tags; i++)
             count1[i] = big_array[pos++];

           for(int i = 0; i < (num_tags * num_tags); i++)
             count2[i] = big_array[pos++];

           for(int i = 0; i < num_tags; i++)
             pr_tag[i] = big_array[pos++];

           for(int i = 0; i < num_tags; i++)
              System.out.println("count1[" + i + "]: " + count1[i]);

           for(int i = 0; i < (num_tags * num_tags); i++)
              System.out.println("count2[" + i + "]: " + count2[i]);

           for(int i = 0; i < num_tags; i++)
              System.out.println("pr_tag[" + i + "]: " + pr_tag[i]);

        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e)  {
            System.err.println("IO Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } catch (ClassNotFoundException e)  {
            System.err.println("Class Not Found Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } // catch
    } // scangram
} // class testNgramSerialized
