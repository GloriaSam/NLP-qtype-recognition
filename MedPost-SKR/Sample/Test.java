import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import gov.nih.nlm.nls.mps.*;

public class Test
{
    private static String inFile = "";
    private static String outFile = "";
    private static PrintStream out = System.out;
    private static InputStream in = System.in;
    private static BufferedReader br = null;

    private static Tagger tagger = null;

    // PUBLIC METHODS  ------------------------------------------------

    public static void main(String args[])
    {
        parseArgs(args);
        tagger = new Tagger();
        getText();
    } // main



    // Our constructor does nothing
    public Test() {}

    // PRIVATE METHODS  ------------------------------------------------


    // Parse the arguments list
    private static void parseArgs(String args[])
    {
        int nArgs = args.length;
        for(int i = 0; i < args.length; ++i)
        {
            if(!args[i].startsWith("-")) // input/output filenames
            {
                if(inFile.length() == 0)
                  inFile = args[i];
                else
                  outFile = args[i];
            } // fi
        } // for

        if(outFile.length() > 0) // Default is System.out
        {
            try 
            {
               FileOutputStream f = new FileOutputStream(outFile);
               out = new PrintStream(f);
            } catch (FileNotFoundException e) {
               System.err.println(outFile + " not able to create.");
               System.exit(-1);
            } //
        } // fi

        try
        {
           if(inFile.length() > 0) // Default is System.in
           {
              FileInputStream fis = new FileInputStream(inFile);
              br = new BufferedReader(new InputStreamReader(fis));
           } // fi

           else // stdin
              br = new BufferedReader(new InputStreamReader(System.in));
        } catch (FileNotFoundException e) {
            System.err.println(inFile + " does not exist.");
            e.printStackTrace();
            System.exit(-1);
        } // try-catch
    } // parseArgs


    private static void getText()
    {
        String line = "";
        boolean done, eoc;

        try
        {
          done = false;
          if((line = br.readLine()) == null)
            done = true;

          // Metamap sends "syn|prolog" as first line - ignore

          if(!done && line.startsWith("syn|prolog"))
          {
             if((line = br.readLine()) == null)
               done = true;
          } // fi

          while(!done)
          {
             eoc = false;
             StringBuffer buf = new StringBuffer();
             if(line.startsWith("EOF") || line.startsWith("^THE_END^"))
             {
                eoc = true;
                done = true;
             } // fi

             else if(line.length() < 2)
               eoc = true;

             while(!done && !eoc)
             {
                if(line.length() < 2)
                  eoc = true;
                else
                {
                    if(buf.length() > 0)
                      buf.append(" ");
                    buf.append(line.trim());
                } // else

                if(!eoc)
                  if((line = br.readLine()) == null)
                   done = true;
             } // while !eoc

             if(buf.length() > 0)
             {
                String result = tagger.tagText(buf.toString(), true, false);
                out.print(result);
             } // fi

             if(!done)
               if((line = br.readLine()) == null)
                 done = true;
          } // while !done

          if(br != null) br.close();
        } catch (IOException e)  {
            System.err.println("IO Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } // catch
    } // getText
} // class Test
