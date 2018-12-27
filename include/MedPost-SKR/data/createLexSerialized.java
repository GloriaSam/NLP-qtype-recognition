import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.Map;
import java.util.HashMap;
import java.util.StringTokenizer;

public class createLexSerialized
{
    private static final String lex_file = 
      System.getProperty("lex_file", "./lex.cur");

    // PUBLIC METHODS  ------------------------------------------------

    // Our default constructor does nothing
    public createLexSerialized() {}


    public static void main(String args[])
    {
        loadLex();
    } // main


    // PRIVATE METHODS  ------------------------------------------------

    private static void loadLex()
    {
        String line = null;

        Map lexDB = new HashMap(14800); // Rounded up 11,129 entries div 0.75

        try
        {
          FileInputStream fis = new FileInputStream(lex_file);
          BufferedReader br = new BufferedReader(new InputStreamReader(fis));
          while((line=br.readLine()) != null)
          {
             int pos = line.indexOf('|');

             if(pos > -1)
             {
                String key = line.substring(0, pos);
                String tmp = line.substring(pos + 1);
                double Val = computeVal(tmp);
                int degreesOfAmbig = computeDegreesOfAmbiguity(tmp);
                if(key.equals("^-_") || key.equals("_") || key.equals("^:_"))
                  degreesOfAmbig = 1;
                String data = tmp + "|" + Val + "|" + degreesOfAmbig;
System.out.println("key: #" + key + "#  data: #" + data + "#");

                lexDB.put(key, data);
             } // fi
          } // while

        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./lexDB.serial"));
        oos.writeObject(lexDB);
        oos.close();
        } catch (FileNotFoundException e) {
            System.err.println(lex_file + " does not exist.");
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e)  {
            System.err.println("IO Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } // catch
    } // loadLex


    private static double computeVal(String inBuf)
    {
        double rtn = 0.0;

        if(inBuf.length() > 0)
        {
           StringTokenizer st = new StringTokenizer(inBuf, "_");
           while(st.hasMoreTokens())
           {
              double val = 1000.0;
              String wordT = st.nextToken();
              int pos = wordT.indexOf(':', 1);
              if(pos > -1)
                val = Double.parseDouble(wordT.substring(pos + 1));

              rtn += val;
           } // while
        } // fi

        return(rtn);
    } // computeVal


    private static int computeDegreesOfAmbiguity(String inBuf)
    {
        int rtn = 0;

        if(inBuf.length() > 0)
        {
           StringTokenizer st = new StringTokenizer(inBuf, "_");
           while(st.hasMoreTokens())
           {
              String wordT = st.nextToken();
              rtn++;
           } // while
        } // fi

        return(rtn);
    } // computeDegreesOfAmbiguity
} // class createLexSerialized
