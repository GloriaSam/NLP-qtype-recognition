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
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.lang.Character;
import java.util.Map;
import java.util.HashMap;
import java.util.StringTokenizer;

import gov.nih.nlm.nls.mps.Tags;


/**
 * Load, access, and search the trained Lexicon data.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	2.0, June 9, 2005
 * @since	1.0
**/

/* History:
 *    2.0, June 9, 2005
 *      - Added degree of ambiguity support (searchLexDB, lastAmbiguityDegree)
 *
 *    1.0, August 18, 2004
 *      - Epoch
**/

public class Lex
{
    private static final String lexFile = 
      System.getProperty("lexFile", "/usr/local/MedPost-SKR/data/lexDB.serial");

    private static Map lexDB = null;
    private static final int ntail = 30;

    /**
     * Static Initializer to initialize the Lexicon Serialized HashMap.
    **/

    static {
        init();
    };

    private int num_tags = 0;
    private double lastCount = 0.0;
    private int lastAmbiguityDegree = 0;

    // PUBLIC METHODS  ------------------------------------------------

    /**
     * The main method for Lex (stand-alone use)
    **/

    public static void main()
    {
        Lex edp = new Lex();
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
     * The default constructor which sets local variable 
    **/

    public Lex()
    {
        num_tags = Tags.num_tags;
    } // Lex


    /** 
     * Get the lexical entry for this word if it exists.
     *
     * @param  word Word to look for in the Lexicon
     * @return Lexical entry if it exists
    **/

    public String getLexEntry(String word)
    {
        boolean found = false;
        String buff = "";
        String rtn = "";

        lastCount = 0.0;
        if(word.length() > 0)
        {
           buff = compactNumbers(word);
           rtn = searchLexDb(buff, true);
           if(rtn.length() > 0)
              found = true;

           else  // try the lowercase version
           {
               buff = buff.toLowerCase();
               rtn = searchLexDb(buff, true);
               if(rtn.length() > 0)
                  found = true;
           } // else
        } // fi

        // If not found, get the search string corresponding to the word.

        if(!found)
        {
           buff = searchString(word);

           while(!found && (buff.length() > 0))
           {
              rtn = searchLexDb(buff, false);
              if(rtn.length() > 0)
                found = true;
              else
                buff = buff.substring(1);
           } // while
        } // fi

        if(!found)
          System.out.println("word: #" + word + "# lex entry not found");

        return(rtn.toString());
    } //getLexEntry


    /** 
     * Does word exist in the Lexicon?
     *
     * @param  word Word to look for in the Lexicon
     * @return true if word exists, false if not
    **/

    public boolean exists(String word)
    {
        boolean rtn = false;

        if(word.length() > 0)
        {
           StringBuffer buff = new StringBuffer().append("^");
           buff.append(word.toLowerCase()).append("_");

           String tmp = searchLexDb(buff.toString(), true);
           if(tmp.length() > 0)
              rtn = true;
        } // fi

        return(rtn);
    } //exists


    /** 
     * Get lastCount assigned via countLex.
     *
     * @return count0 value
    **/

    public double getLastCount()
    {
        return(lastCount);
    } // getLastCount


    /** 
     * Get lastAmbiguityDegree assigned via countLex.
     *
     * @return count0 value
    **/

    public int getLastAmbiguityDegree()
    {
        return(lastAmbiguityDegree);
    } // getLastAmbiguityDegree


    /** 
     * Search through the lexical string (inBuf) looking for inTag and
     * then return the count - either the number following "inTag:" or
     * 1,000.0 if no colon.
     * Example: Parse through the lex string:
     *   Something like the following for "to" "_TO_II_CC:1"
     *     For each tag without a colon, add 1,000 "_TO", "_II"
     *     For each tag with a colon, add in the # following colon "_CC:1"
     *   Thus, the example equals 2,001.
     *
     * @param  inBuf Lexical entry string to scan
     * @param  inTag Tag to scan for within inBuf
     * @return final count of inTags possibilities
    **/

    public double scanLex(String inBuf, String inTag)
    {
        double rtn = 0.0;
 
        if(inBuf.length() > 0)
        {
             StringBuffer tmp = new StringBuffer("_").append(inTag);
             int pos = inBuf.indexOf(tmp.toString());
             if(pos >= 0)
             {
                 boolean found = false;
                 String s = inBuf.substring(pos);

                 while(!found && (s.length() > 0) && (s.charAt(0) == '_'))
                 {
                    String tmp2 = "";
                    int pos2 = s.indexOf('_', 1);
                    if(pos2 > -1)
                    {
                       tmp2 = s.substring(1, pos2);
                       s = s.substring(pos2);
                    } // fi

                    else
                    {
                       tmp2 = s.substring(1);
                       s = "";
                    } // else

                    // Find the count (compensate for "::1000").

                    double v = 1000.0;
                    int pos3 = tmp2.indexOf(':', 1);

                    if(pos3 > -1)
                    {
                       String tmp3 = tmp2.substring(pos3 + 1);
                       tmp2 = tmp2.substring(0, pos3);

                       if(tmp3.length() > 0)
                         v = Double.parseDouble(tmp3);
                       
                    } // fi

                    if(inTag.equals(tmp2))
                    {
                       found = true;
                       rtn = v;
                    } // fi
                 } // while
             } // fi pos
        } // fi length

        return(rtn);
    } // scanLex

    // PRIVATE METHODS  ------------------------------------------------


    /**
     * Prints usage information. 
    **/

    private static void usage()
    {
        System.out.println("Lex");
        System.exit(-1);
    } // usage


    /**
     * Test run called via main. 
    **/

    private void run()
    {
        searchLexDb("^providing_", true);
    } // run

    /**
     * Initialize the HashMap via serialized data file. 
    **/

    private static void init()
    {
        try
        {
           ObjectInputStream ois = 
                   new ObjectInputStream(new FileInputStream(lexFile));
           lexDB = (HashMap)ois.readObject();
           ois.close();
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
    } // init


    /** 
     * Develop an alternative ending to look for based on structure of word.
     *
     * @param  inWord Word to create the alternative for
     * @return Alternative variation of the word to look for
    **/

    private String searchString(String inWord)
    {
        String line = null;
        StringBuffer rtn = new StringBuffer();
        int len = inWord.length();
        boolean found = false;

        if(len == 1)
        {
           if(Character.isDigit(inWord.charAt(0)))
           {
              rtn.append(inWord).append("I");
              found = true;
           } // fi

           else if(Character.isLetter(inWord.charAt(0)))
           {
              rtn.append(inWord.toLowerCase()).append("S");
              found = true;
           } // else fi
        } // fi

        if(!found)
        {
            int	num_A = 0;   // Upper case
            int	num_L = 0;   // Lower case
            int	num_N = 0;   // Numbers
            int	num_P = 0;   // Punctuation (numeric)
            int	num_Q = 0;   // Quotes
            int	num_O = 0;   // Other
            char last_ch = ' ';

            for(int i = 0; i < len; i++)
            {
                if((i == 0) && (inWord.charAt(i) == '-'))
                  num_L++;

                else if(Character.isLowerCase(inWord.charAt(i)))
                  num_L++;

                else if(Character.isUpperCase(inWord.charAt(i)))
                  num_A++;

                else if(Character.isDigit(inWord.charAt(i)))
                  num_N++;

                else if((inWord.charAt(i) == '=')||(inWord.charAt(i) == ':') ||
                        (inWord.charAt(i) == '+')||(inWord.charAt(i) == '.') ||
                        (inWord.charAt(i) == ','))
                  num_P++;

                else if((inWord.charAt(i) == '\'') ||
                        (inWord.charAt(i) == '`') || (inWord.charAt(i) == '"'))
                  num_Q++;

                else
                  num_O++;

                last_ch = inWord.charAt(i);
            } // for

            int pos = 0;
            if((len - ntail) > 0)
              pos = len - ntail;

            rtn.append(inWord.substring(pos).toLowerCase());

            if((num_L + num_Q) == len)
              rtn.append("");

            else if((num_A + num_Q) == len)
              rtn.append("A");

            else if((num_N + num_P + num_Q) == len)
              rtn.append("N");

            else if((num_L + num_A + num_Q) == len)
              rtn.append("B");

            else if((num_A + num_N + num_P + num_Q) == len)
              rtn.append("C");

            else if((num_L + num_N + num_P + num_Q) == len)
              rtn.append("E");

            else if((num_A + num_L + num_N + num_P + num_Q) == len)
              rtn.append("D");

            else if((num_O == 0) && (last_ch == '+'))
              rtn.append("F");

            else
              rtn.append("O");
        } // else

        rtn.append("_");

        return(rtn.toString());
    } // searchString


    /** 
     * Search Lexicon for inKey.
     *
     * If requested (flag = true), we also track the precomputed
     * scanLex value for this Lexical item.  The data has two elements:
     * Potential Tag(s) for inKey|Precomputed scanLex Value
     *
     * @param  inKey Key to search for in the Lexicon
     * @return Lexical entry for this key
    **/

    private String searchLexDb(String inKey, boolean flag)
    {
        String rtn = "";

        Object o = lexDB.get(inKey);
        if(o != null)
        {
           String tmp = o.toString();
           int pos = tmp.indexOf("|");
           int pos2 = tmp.indexOf("|", pos + 1);

           rtn = tmp.substring(0, pos);

           // If flag = true, track this scanLex value

           if(flag)
             lastCount = Double.parseDouble(tmp.substring(pos + 1, pos2));

           // Pull the last field (Degrees of Ambiguity)

           if(Tagger.usePrologFull())
              lastAmbiguityDegree = Integer.parseInt(tmp.substring(pos2 + 1));
        } // fi

        return(rtn);
    } // searchLexDb


    /** 
     * Force numbers and multiple numbers to single "1".
     *
     * @param  inWord Word to compact
     * @return Processed word string compacted where possible
    **/

    private String compactNumbers(String inWord)
    {
        StringBuffer rtn = new StringBuffer().append("^");
        boolean last_digit = false;

        for(int i = 0; i < inWord.length(); i++)
        {
           char ch = inWord.charAt(i);
           if(Character.isDigit(ch))
           {
               if(!last_digit)
               {
                  last_digit = true;
                  rtn.append('1');
               } // fi
           } // fi

           else
           {
              last_digit = false;
              rtn.append(ch);
           } // else
        } // for
        
        rtn.append("_");

        return(rtn.toString());
    } // compactNumbers
} // class Lex
