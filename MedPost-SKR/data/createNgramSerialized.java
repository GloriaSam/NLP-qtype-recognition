import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class createNgramSerialized
{
    private static final String ngram_file = 
      System.getProperty("ngram_file", "./ngrams.cur");

    private static final int MAX_TAGS = 60;
    private static int num_tags = 0;
    private static double count0;
    private static double count1[] = new double[MAX_TAGS];
    private static double count2[] = new double[MAX_TAGS * MAX_TAGS];
    private static String tag_str[] = new String[MAX_TAGS];
    private static double pr_tag[] = new double[MAX_TAGS];
    private static double big_array[] = new double[MAX_TAGS + (MAX_TAGS * MAX_TAGS) + MAX_TAGS];

    // PUBLIC METHODS  ------------------------------------------------

    // Our default constructor does nothing
    public createNgramSerialized() {}


    public static void main(String args[])
    {
        scangram();
    } // main


    // PRIVATE METHODS  ------------------------------------------------

    private static void scangram()
    {
        String line = null;
        boolean done;

        /* ngram file is formatted as follows:

              Number of Tags (integer)
              Tags -- Number of Tags lines one line for each tag
              count0
              count1 values -- Number space # space text  "33 # ''"
                   Number of Tags lines one line for each value
              count2 values -- Number space # space text  "33 # ''"
                   (Number of Tags * Number of Tags) lines one for each
        */

        try
        {
          FileInputStream fis = new FileInputStream(ngram_file);
          BufferedReader br = new BufferedReader(new InputStreamReader(fis));

          // Retrieve the num_tags value - first line of file

          line = br.readLine();
          num_tags = Integer.parseInt(line);

System.out.println("num_tags: " + num_tags);

          // Retrieve the ngram tags (num_tags worth) */

          done = false;
          for(int i = 0; !done && (i < num_tags); i++)
          {
             if((line = br.readLine()) == null)
               done = true;
             else
               tag_str[i] = line;
System.out.println("tag_str[" + i + "]: " + tag_str[i]);
          } // for

          if(!done)
          {
             if((line = br.readLine()) == null)
               done = true;
             else
               count0 = Double.parseDouble(line);
          } /* fi */

System.out.println("count0: " + count0);

          // Retrieve the count1 values (num_tags worth)

          for(int i = 0; !done && (i < num_tags); i++)
          {
             if((line = br.readLine()) == null)
               done = true;
             else
               count1[i] = Double.parseDouble(line.substring(0, 
                                                          line.indexOf(' ')));
System.out.println("count1[" + i + "]: " + count1[i]);
          } // for

          // Retrieve the count2 values (num_tags * num_tags worth)

          for(int i = 0; !done && (i < (num_tags * num_tags)); i++)
          {
             if((line = br.readLine()) == null)
               done = true;
             else
             {
                String tmp = line.substring(0, line.indexOf(' '));
                if(tmp.length() > 0)
                  count2[i] = Double.parseDouble(tmp);
                else
                  count2[i] = 0;
             } // else
System.out.println("count2[" + i + "]: " + count2[i]);
          } // for

          br.close(); fis.close();

          // Setup the pr_tag array

          for(int i = 0; i < num_tags; i++)
            pr_tag[i] = count1[i] / count0;

          for(int i = 0; i < num_tags; i++)
System.out.println("pr_tag[" + i + "]: " + pr_tag[i]);

        int pos = 0;
        for(int i = 0; i < num_tags; i++)
           big_array[pos++] = count1[i];

        for(int i = 0; i < (num_tags * num_tags); i++)
           big_array[pos++] = count2[i];

        for(int i = 0; i < num_tags; i++)
           big_array[pos++] = pr_tag[i];


        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./ngramOne.serial"));
        oos.writeObject(big_array);
        oos.close();

        File f = new File("./ngramTags.txt");
        PrintWriter out = new PrintWriter(new FileWriter(f));

        out.println("num_tags = " + num_tags);
        out.println("\ntag_str[] = {");
        for(int i = 0; i < num_tags; i++)
          out.print("\"" + tag_str[i] + "\", ");
        out.println("\n};");

        out.close();

        } catch (FileNotFoundException e) {
            System.err.println(ngram_file + " does not exist.");
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e)  {
            System.err.println("IO Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } // catch
    } // scangram
} // class createNgramSerialized
