/*
===========================================================================
*
*                            PUBLIC DOMAIN NOTICE                          
*               National Center for Biotechnology Information
*         Lister Hill National Center for Biomedical Communications
*                                                                          
*  This software/database is a "United States Government Work" under the   
*  terms of the United States Copyright Act.  It was written as part of    
*  the authors' official duties as a United States Government employee and 
*  thus cannot be copyrighted.  This software/database is freely available 
*  to the public for use. The National Library of Medicine and the U.S.    
*  Government have not placed any restriction on its use or reproduction.  
*                                                                          
*  Although all reasonable efforts have been taken to ensure the accuracy  
*  and reliability of the software and data, the NLM and the U.S.          
*  Government do not and cannot warrant the performance or results that    
*  may be obtained by using this software or data. The NLM and the U.S.    
*  Government disclaim all warranties, express or implied, including       
*  warranties of performance, merchantability or fitness for any particular
*  purpose.                                                                
*                                                                          
*  Please cite the authors in any work or product based on this material.   
*
===========================================================================
*/

package gov.nih.nlm.nls.mps;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.lang.Character;

import gov.nih.nlm.nls.mps.Lex;
import gov.nih.nlm.nls.mps.Ngram;
import gov.nih.nlm.nls.mps.Sentence;

/**
 * Java representation of the MedPost/SKR Part of Speech Tagger for
 * BioMedical Text.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	3.0, January 23, 2006
 * @since	1.0
**/

/* History:
 *    3.0, January 23, 2006
 *      - Added support for also displaying Penn Treebank or MedPost tags<br>
 *
 *    2.0, June 9, 2005
 *      - Cleaned up MAX_WORDS to dynamically size<br>
 *      - Added support for prologfull and upcaret output formats<br>
 *
 *    1.0, August 18, 2004
 *      - Epoch
*/

public class Tagger
{
    static Lex lex = null;
    static Ngram ngram = null;

    /**
     * Static Initializer to load both the Ngram and Lexical data only at
     * startup.
    **/

    static
    {
       ngram = new Ngram();
       lex = new Lex();
    };

    public static final int MAX_TAGS = 60;
    public static final int MAX_STATES = 60;
    public static final int MAX_LLEN = 10000;
    public static final int BUF_SIZE = 1000;

    private static String inFile = "";
    private static String outFile = "";
    private static PrintStream out = System.out;
    private static InputStream in = System.in;
    private static BufferedReader br = null;

    private boolean printed_prolog = false;
    private static boolean prologOutput = false;
    private static boolean prologFullOutput = false;
    private static boolean tagPennTreebank = false;
    private static boolean tagMedPost = false;
    private static boolean tagSpecialist = true;

    // PUBLIC METHODS  ------------------------------------------------

    /**
     * The main method for Tagger 
     *
     * @param args Array of arguments to the Tagger
    **/

    public static void main(String args[])
    {
        // Default is prolog output with Specialist Lexicon tags

        prologOutput = true;
        prologFullOutput = false;
        tagSpecialist = true;
        tagPennTreebank = false;
        tagMedPost = false;

        // Parse any arguments from the user

        parseArgs(args);
        Tagger edp = new Tagger();

        try
        {
            edp.run();
        } catch (Exception e)
        {
            System.out.println("Exception: " + e.toString());
            e.printStackTrace();
        } // catch
    } // main


    /**
     * Tags the incoming text and returns a prolog or "^" list of tagged text. 
     *
     * @param    inDoc Free text to be tagged by the tagger
     * @param    isPrologOutput Flag for Prolog or upCaret formatted output
     * @param    isPrologFullOutput Flag for including Degrees of Ambiguity
     * @return   Prolog inspired string of text and tags;
     *           NOTE: "^THE_END^" on a line by itself terminator.
    **/

    public String tagText(String inDoc, boolean isPrologOutput, 
                          boolean isPrologFullOutput)
    {
        StringBuffer result = new StringBuffer();

        printed_prolog = false;
        prologOutput = isPrologOutput;
        prologFullOutput = isPrologFullOutput;

        if(isPrologOutput)
        {
            result.append("[\n");
            result.append(processText(inDoc));
            if(printed_prolog)
              result.append("\n");
            result.append("].\n");
        } // fi

        else
          result.append(processText(inDoc));

        result.append("^THE_END^\n");
        return(result.toString());
    } // tagText


    /**
     * Thread-safe routine that tags the incoming text and returns a prolog
     * or "^" list of tagged text. 
     *
     * @param    inDoc Free text to be tagged by the tagger
     * @param    isPrologOutput Flag for Prolog or upCaret formatted output
     * @param    isPrologFullOutput Flag for including Degrees of Ambiguity
     * @return   Prolog inspired string of text and tags;
     *           NOTE: "^THE_END^" on a line by itself terminator.
    **/

    public synchronized String tagTextSynchronized(String inDoc,
                 boolean isPrologOutput, boolean isPrologFullOutput,
                 boolean usePennTreebankTags, boolean useMedPostTags,
                 boolean useSpecialistTags)
    {
        StringBuffer result = new StringBuffer();

        printed_prolog = false;
        prologOutput = isPrologOutput;
        prologFullOutput = isPrologFullOutput;
        tagPennTreebank = usePennTreebankTags;
        tagMedPost = useMedPostTags;
        tagSpecialist = useSpecialistTags;

        if(isPrologOutput)
        {
            result.append("[\n");
            result.append(processText(inDoc));
            if(printed_prolog)
              result.append("\n");
            result.append("].\n");
        } // fi

        else
          result.append(processText(inDoc));

        result.append("^THE_END^\n");
        return(result.toString());
    } // tagTextSynchronized


    /**
     * Default constructor does nothing. 
    **/

    // Our constructor does nothing
    public Tagger() {}


    /** 
     * Use prologFullOutput?
     *
     * @return prologFullOutput
    **/

    public static boolean usePrologFull()
    {
        return(prologFullOutput);
    } //usePrologFull


    /** 
     * Use prologOutput?
     *
     * @return prologOutput
    **/

    public static boolean usePrologOutput()
    {
        return(prologOutput);
    } //usePrologOutput


    /** 
     * Use Penn Treebank tags?
     *
     * @return tagPennTreebank
    **/

    public static boolean usePennTreebankTags()
    {
        return(tagPennTreebank);
    } //usePennTreebankTags


    /** 
     * Use orignal MedPost tags?
     *
     * @return tagMedPost
    **/

    public static boolean useMedPostTags()
    {
        return(tagMedPost);
    } //useMedPostTags


    /** 
     * Use Specialist Lexicon tags?
     *
     * @return tagSpecialist
    **/

    public static boolean useSpecialistTags()
    {
        return(tagSpecialist);
    } //useSpecialistTags

    // PRIVATE METHODS  ------------------------------------------------


    /**
     * Prints usage information. 
    **/

    private static void usage()
    {
        System.out.println("");
        System.out.println("java -cp lib/mps.jar gov.nih.nlm.nls.mps.Tagger \\");
        System.out.println("    infile outfile [outFormat] [tagFormat]");

        System.out.println("");
        System.out.println("  where outFormat (optional, default is -prolog):");
        System.out.println("     -upCaret     - \"word^tag\" format");
        System.out.println("     -prolog      - \"['word', 'tag'],\" format");
        System.out.println("     -prologFull  - \"['word', 'tag/degreeOfAmbiguity'],\" format");
        System.out.println("");

        System.out.println("  where tagFormat (optional, default is -Specialist):");
        System.out.println("    -Specialist  - SPECIALIST Lexicon tags");
        System.out.println("     -Penn        - Penn Treebank tags");
        System.out.println("     -MedPost     - Original MedPost tags");
        System.exit(-1);
    } // usage


    /**
     * Process document called via main. 
    **/

    private void run()
    {
        // process

        getText();

        if(outFile.length() > 0)
          out.close();
    } // run


    /**
     * Parse program arguments and initialize variables based on args.
     *
     * @param   args Array of arguments to the Tagger
    **/

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

            else
            {
                if(args[i].compareToIgnoreCase("-upCaret") == 0)
                {
                    prologOutput = false;
                    prologFullOutput = false;
                } // fi

                else if(args[i].compareToIgnoreCase("-prolog") == 0)
                {
                    prologOutput = true;
                    prologFullOutput = false;
                } // else fi

                else if(args[i].compareToIgnoreCase("-prologFull") == 0)
                {
                    prologOutput = true;
                    prologFullOutput = true;
                } // else fi

                else if(args[i].compareToIgnoreCase("-Penn") == 0)
                {
                    tagPennTreebank = true;
                    tagMedPost = false;
                    tagSpecialist = false;
                } // else fi

                else if(args[i].compareToIgnoreCase("-MedPost") == 0)
                {
                    tagPennTreebank = false;
                    tagMedPost = true;
                    tagSpecialist = false;
                } // else fi

                else if(args[i].compareToIgnoreCase("-Specialist") == 0)
                {
                    tagPennTreebank = false;
                    tagMedPost = false;
                    tagSpecialist = true;
                } // else fi

                else if(args[i].compareToIgnoreCase("-help") == 0)
                  usage();

                else
                {
                    System.out.println("");
                    System.out.println("ERROR: Argument not recognized: #" +
                           args[i] + "# ignored");
                    usage();
                } // else
            } // else
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


    /**
     * Read through file chopping into units delimited by blank lines and
     * then calling for further processing.  Only called via main when we
     * are running stand-alone.
    **/

    private void getText()
    {
        String line = "";
        boolean done, eoc;

        StringBuffer result = new StringBuffer();
        try
        {
          done = false;
          if((line = br.readLine()) == null)
            done = true;

          // Possible first lines include: "syn|prolog", "syn|prologfull" or
          // "syn|upcaret" - just use to determine output format.

          if(!done && line.startsWith("syn|"))
          {
              if(line.startsWith("syn|prologfull"))
                 prologFullOutput = true;

              else if(line.startsWith("syn|upcaret"))
                 prologOutput = false;

              if(line.indexOf("|penn") > -1)
              {
                    tagPennTreebank = true;
                    tagMedPost = false;
                    tagSpecialist = false;
              } // fi

              else if(line.indexOf("|medpost") > -1)
              {
                    tagPennTreebank = false;
                    tagMedPost = true;
                    tagSpecialist = false;
              } // else fi

              else // DEFAULT - Specialist
              {
                    tagPennTreebank = false;
                    tagMedPost = false;
                    tagSpecialist = true;
              } // else fi

              if((line = br.readLine()) == null)
                done = true;
          } // fi

          // Now, loop through input text until we either run out of
          // data (EOF), we see "EOF" text on a line by itself, or
          // "^THE_END^" text on a line by itself.

          while(!done)
          {
             eoc = false;
             StringBuffer buf = new StringBuffer();
             if(line.startsWith("EOF") || line.startsWith("^THE_END^"))
             {
                eoc = true;
                done = true;
             } // fi

             // If we have a blank line, we are at the end of a unit of text.

             else if(line.length() < 2)
               eoc = true;

             while(!done && !eoc)
             {
                // If we have a blank line, we are at the end of a unit of text.

                if(line.length() < 2)
                  eoc = true;

                // Else, continue accumulating this unit's text

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

             // If this unit of text contains anything, process further.

             if(buf.length() > 0)
                result.append(tagText(buf.toString(), prologOutput,
                                                  prologFullOutput));

             if(!done)
               if((line = br.readLine()) == null)
                 done = true;
          } // while !done

          if(br != null) br.close();

          // Print out the result to the designated output media.

          out.print(result);
          out.flush();
        } catch (IOException e)  {
            System.err.println("IO Exception: " + e.toString());
            e.printStackTrace();
            System.exit(-1);
        } // catch
    } // getText


    /**
     * Tokenize text, break into sentences and call for tagging.
     *
     * @param    inText Free text to be tagged by the tagger
     * @return   Prolog inspired string of text and tags for inText
    **/

    private String processText(String inText)
    {
        char ch;
        // Space @ beginning of new string;
        StringBuffer newL = new StringBuffer().append(' ');

        // Tokenize the incoming text.  Basically breaking up everything
        // isn't a cohesive string of alphanumerics.  We modify single
        // quotes so that they work with prolog.  We also modify back single
        // quote to a normal double-quote for prolog.

        int i = 0;
        int len = inText.length();

        while(i < len)
        {
           ch = inText.charAt(i);
           if(ch == '\'')
              newL.append(" '' ");  // Proper # for prolog

           else if((ch == ' ') || Character.isLetterOrDigit(ch))
             newL.append(ch);

           else
           {
              newL.append(' ');
              newL.append(ch);
              newL.append(' ');
           } // else

           i++;
        } // while

        // Now strip off the beginning spaces, compress multiple spaces into
        // a single space, and add in the "UNTAGGED" tags wherever we 
        // encounter a space and at the very end of the text.

        StringBuffer newF = new StringBuffer(newL.toString().trim().replaceAll("  ", " ").replaceAll("  ", " ").replaceAll(" ", "_UNTAGGED ")).append("_UNTAGGED");

        // Now we breakdown the text into sentences where possible

        len = newF.length();
        int lpos, rpos1, rpos2, rpos3, rposF;
        lpos = 0;
        StringBuffer result = new StringBuffer();

        while(lpos < len)
        {
           rpos1 = newF.indexOf(" !_", lpos);
           rpos2 = newF.indexOf(" ?_", lpos);
           rpos3 = newF.indexOf(" ._", lpos);
   
           rposF = 1000000;
           if((rpos1 < rposF) && (rpos1 > -1))
             rposF = rpos1;

           if((rpos2 < rposF) && (rpos2 > -1))
             rposF = rpos2;

           if((rpos3 < rposF) && (rpos3 > -1))
             rposF = rpos3;

           if(rposF == 1000000)
             rposF = -1;

           StringBuffer sent = new StringBuffer();
           if(rposF > -1)  // need a substring
           {
              sent.append(newF.substring(lpos, rposF + 3)).append('.');
              lpos = rposF + 12;
           } // fi

           else
           {
              sent.append(newF.substring(lpos));
              lpos = len + 1;
           } // else

           // Generate an estimate on the max number of words in this
           // sentence.  Saves time versus allocating 10,000 for each 
           // sentence that I was doing.

           String[] Jresult = (sent.toString()).split("_");
           int tmpNumWords = Jresult.length * 3;

           // Now tag this sentence

           Sentence sentence = new Sentence(sent.toString(), tmpNumWords);
           String tmp = sentence.printToString(printed_prolog);
           if(tmp.length() > 0)
             printed_prolog = true;
           result.append(tmp);
        } // while

        return(result.toString());
    } // processText
} // class Tagger
