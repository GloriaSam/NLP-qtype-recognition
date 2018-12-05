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

import java.lang.Character;


/**
 * Determine proper tag for each word.
 * <p>
 * Determine an absolute match when possible, otherwise calculate the
 * probabilities of the various tags for each word.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	2.0, June 9, 2005
 * @since	1.0
**/


/* History:
 *    2.0, June 9, 2005
 *      - Cleaned up MAX_WORDS to be sent in at creation
 *
 *    1.0, August 18, 2004
 *      - Epoch
**/

public class Tags
{
    public static final int num_tags = 60;
    private static final String tag_str[] = {
       "''", "(", ")", ",", ".", ":", "CC", "CS", "CSN", "CST", "DB", "DD",
       "EX", "GE", "II", "JJ", "JJR", "JJT", "MC", "NN", "NNP", "NNS", "PN",
       "PND", "PNG", "PNR", "RR", "RRR", "RRT", "SYM", "TO", "VBB", "VBD",
       "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ",
       "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM", "VVB", "VVD", "VVG",
       "VVGJ", "VVGN", "VVI", "VVN", "VVNJ", "VVZ", "``"};

    public static final int DSQ = 0;   // Positional location of '' in tag list
    public static final int LP = 1;    // Positional location of ( in tag list
    public static final int RP = 2;    // Positional location of ) in tag list
    public static final int CM = 3;    // Positional location of , in tag list
    public static final int PD = 4;    // Positional location of . in tag list
    public static final int CN = 5;    // Positional location of : in tag list
    public static final int MC = 18;   // Positional location of MC in tag list
    public static final int NN = 19;   // Positional location of NN in tag list
    public static final int NNP = 20;  // Positional location of NNP in tag list
    public static final int NNS = 21;  // Positional location of NNS in tag list
    public static final int LSQ = 59;  // Positional location of `` in tag list
    private static final double INSIGNIF = 10e-10;

    // d_w(t) for each word and tag
    private double dw[][];


    // PUBLIC METHODS  ------------------------------------------------

    // Our default constructor does nothing
    public Tags() {
        dw = new double[1000][Tagger.MAX_TAGS];
    }

    // Our default constructor does nothing
    public Tags(int numWords) {
        dw = new double[numWords * 2][Tagger.MAX_TAGS];
    }


    /** 
     * Get tag_str at position pos.
     *
     * @param   pos Position within tag_str requested
     * @return  Tag at requested position
    **/

    public static String getTagStrAt(int pos)
    {
      return(tag_str[pos]);
    } // getTagStrAt


    /** 
     * Get tag location.
     *
     * @param  tag Tag to look for within tag_str
     * @return Position within tag_str of tag
    **/

    public int getTagPos(String tag)
    {
        int rtn = -1;
        boolean found = false;

        for(int i = 0; !found && (i < num_tags); i++)
        {
            if(tag_str[i].equals(tag))
            {
                found = true;
                rtn = i;
            } // fi
        } // for

        return(rtn);
    } // getTagPos


    /** 
     * Get dw at position pos1, pos2.
     *
     * @param   pos1 Position one within dw requested
     * @param   pos2 Position two within dw requested
     * @return  Dw element at requested position
    **/

    public double getDwAt(int pos1, int pos2)
    {
        return(dw[pos1][pos2]);
    } // getDwAt


    /** 
     * Get full dw array.
     *
     * @return  Dw array
    **/

    public double[][] getDw()
    {
        return(dw);
    } // getDw


    /** 
     * Set dw at position pos1, pos2.
     *
     * @param   pos1 Position one within dw requested
     * @param   pos2 Position two within dw requested
     * @param   val  Value to be inserted at pos1, pos2
    **/

    public void setDwAt(int pos1, int pos2, double val)
    {
        dw[pos1][pos2] = val;
    } // setDwAt


    /** 
     * Normalize dw array.
     *
     * @param   numWords    Number of words in current sentence
     * @param   lex_backoff Array of Lex backoff values for each tag
     * @param   count       Array of Lexical occurrences counts
    **/

    public void normalize(int numWords, double lex_backoff[], double count[])
    {
        double N, T, Z, d, sum;
        boolean known;

        for(int i = 0; i < numWords; i++)
        {
            N = 0.0;
            T = 0.0;
            Z = 0.0;
            for(int t = 0; t < num_tags; t++)
            {
                N += dw[i][t];
                if(dw[i][t] > 0.0)
                  T++;
                else
                  Z += lex_backoff[t];
            } // for num_tags

            d = N / (N + T);

            // If the word was seen enough (or if it was manually tagged)
            // then don't back off, but average with an insignificant amount
            // of the backoff in case it is needed to break ties.

            known = false;
            if((count[i] > 999.0) || (N > 999.0))
              known = true;

            if(known)
              d = 1.0 - INSIGNIF;

            // Witten-Bell smoothing with uniform backoff priors

            sum = 0.0;
            for(int t = 0; t < num_tags; t++)
            {
                if(known)
                {
                    if((N > 0.0) && (dw[i][t] > 0.0))
                      dw[i][t] = (d * dw[i][t]) / N;
                    else
                      dw[i][t] = 0.0;

                    if((Z > 0.0) && (dw[i][t] > 0.0))
                      dw[i][t] += ((1.0 - d) * lex_backoff[t]) / Z;
                } // fi known

                else
                {
                    if((N > 0.0) && (dw[i][t] > 0.0))
                      dw[i][t] = (d * dw[i][t]) / N;

                    else if(Z > 0.0)
                      dw[i][t] += ((1.0 - d) * lex_backoff[t]) / Z;

                    else
                      dw[i][t] = 0.0;
                } // else

                sum += dw[i][t];
            } // for num_tags

            if(sum == 0.0)
            {
                System.out.println("Cannot continue because word " + 
                         i + " has no possible tags.");
                System.exit(999);
            } // fi
        } // for numWords
    } // normalize


    /** 
     * Update dw array based on actual text/tags.
     *
     * @param   wordPos Position of this word in the dw array
     * @param   tagPos  Which tag we are affecting the value for
     * @param   cond    Ensuring (true) tag or Removing (false) tag probability
    **/

    public void tagSet(int wordPos, int tagPos, boolean cond)
    {
       if(cond)  // We know for sure that this is the proper tag.
       {
           for(int i = 0; i < num_tags; i++)
              dw[wordPos][i] = 0.0;
           dw[wordPos][tagPos] = 1.0;
       } // fi

       else  // We know for sure that this is NOT the proper tag.
       {
           dw[wordPos][tagPos] = 0.0;

           double m = 0.0;
           for(int i = 0; i < num_tags; i++)
              m += dw[wordPos][i];

           for(int i = 0; i < num_tags; i++)
           {
              if(m == 0.0)
                dw[wordPos][i] = 0.0;
              else
                dw[wordPos][i] /= m;
           } // for
       } // else
    } // tagSet


    /** 
     * Attempt to narrow down or pinpoint actual tag for the given word.
     *
     * @param   word     Word to check on
     * @param   i        Position of this word
     * @param   wordLen  Length of the word to check
    **/

    public void checkTags(String word, int i, int wordLen)
    {
         if(isAllDigit(word))
           tagSet(i, MC, true);

         else
         {
            tagSet(i, MC, false);

            // Handle punctuation first.  Consider ? and ! as periods.

            if((wordLen == 1) &&
               !Character.isLetterOrDigit(word.charAt(0)))
            {
                tagSet(i, DSQ, word.charAt(0) == 39);

                tagSet(i, LP, (word.charAt(0) == '(') ||
                   (word.charAt(0) == '[') || (word.charAt(0) == '{'));

                tagSet(i, RP, (word.charAt(0) == ')') ||
                   (word.charAt(0) == ']') || (word.charAt(0) == '}'));

                tagSet(i, CM, word.charAt(0) == ',');

                tagSet(i, PD, (word.charAt(0) == '.') ||
                   (word.charAt(0) == '!') || (word.charAt(0) == '?'));

                tagSet(i, CN, (word.charAt(0) == '-') ||
                   (word.charAt(0) == ':') || (word.charAt(0) == ';'));

                tagSet(i, LSQ, word.charAt(0) == '`');
            } // fi

            else if((wordLen == 2) &&
                !Character.isLetterOrDigit(word.charAt(0)))
            {
                tagSet(i, DSQ, word.equals("''"));
                tagSet(i, LSQ, word.equals("``"));
                tagSet(i, CN, word.equals("--"));
            } // else fi

            // If word has at least one letter, and has an embedded cap
            // or number, make it an NN or NNS.
            // NNP is possible but we accept this error.

            boolean a = false;
            boolean b = false;

            for(int j = 0; j < wordLen; j++)
            {
                if(!a)
                {
                   if(Character.isLetter(word.charAt(j)))
                     a = true;
                } // fi

                if(!b)
                {
                   if(Character.isDigit(word.charAt(j)))
                     b = true;
                   else if((j > 0) &&
                           Character.isUpperCase(word.charAt(j)))
                     b = true;
                } // fi
            } // for

            if(a && b)
            {
                if((wordLen > 0) && 
                   (word.charAt(wordLen - 1) == 's'))
                  tagSet(i, NNS, true);
                else
                  tagSet(i, NN, true);
            } // fi

            // Require any NNP to be capitalized

            if(!Character.isUpperCase(word.charAt(0)))
              tagSet(i, NNP, false);
         } // else !num
    } // checkTags


    // PRIVATE METHODS  ------------------------------------------------


    /** 
     * Is the string made up entirely of digits?
     *
     * @param   inBuf  String to check
     * @return  true if all digits, false if not
    **/

    private static boolean isAllDigit(String inBuf)
    {
        boolean rtn = true;
        for(int i = 0; rtn && (i < inBuf.length()); i++)
           rtn = Character.isDigit(inBuf.charAt(i));

        return(rtn);
    } // isAllDigit
} // class Tags
