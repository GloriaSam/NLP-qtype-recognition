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
import java.lang.Math;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.io.PrintStream;

import gov.nih.nlm.nls.mps.Lex;
import gov.nih.nlm.nls.mps.Tags;
import gov.nih.nlm.nls.mps.Predict;
import gov.nih.nlm.nls.mps.PennTranslator;
import gov.nih.nlm.nls.mps.MedPostTranslator;
import gov.nih.nlm.nls.mps.MedPostSKRTranslator;


/**
 * Sentence processing.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	3.0, January 23, 2006
 * @since	1.0
 **/

/* History:
 *    3.0, January 23, 2006
 *      - Added support also allowing display of Penn Treebank and MedPost tags
 *
 *    2.0, June 9, 2005
 *      - Cleaned up MAX_WORDS to be sent in during creation
 *      - Added support for prologfull and upcaret output formats
 *
 *    1.0, August 18, 2004
 *      - Epoch
**/

public class Sentence
{
    private static double lex_backoff[] = new double[Tagger.MAX_TAGS];
    private static int num_tags = 0;    // Set from Tags
    private static int num_states = 0;  // Set from Tags
    static {
        num_tags = Tags.num_tags;
        num_states = num_tags;
        String entry = "";
        entry = Tagger.lex.getLexEntry("");
	for(int i = 0; i < num_tags; i++)
	  lex_backoff[i] = Tagger.lex.scanLex(entry, Tags.getTagStrAt(i));
    };

    private static final int MAX_WLEN = 200;

    private int numWords;           // # words in sentence

    private int comp_tag[];         // Computed tags

    // Words from sentence
    private String words[];

    // Length of each word so we don't have to recompute
    private int wordsLens[];

    // # words (multi-words)
    private int wordsCnts[];

    // Count of occurrences (only if whole word was in lexicon)
    private double count[];

    // Degrees of Ambiguity for word (only if whole word was in lexicon)
    private int degreesOfAmbig[];

    // The current tag probability estimates
    private double pr[][];

    // Temporary array
    private double tmpD[] = new double[Tagger.MAX_TAGS];

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * The default constructor does nothing 
    **/

    public Sentence() {
        numWords = 0;
        comp_tag = new int[1000];
        words = new String[1000];
        wordsLens = new int[1000];
        wordsCnts = new int[1000];
        count = new double[1000];
        degreesOfAmbig = new int[1000];
        pr = new double[1000][60];
    }


    /** 
     * Main constructor for Sentence handles parsing into words and processing.
     *
     * @param  inSent Sentence to be processed.
     * @param  inNumWords Estimated max number of words (words * 3).
    **/

    public Sentence(String inSent, int inNumWords)
    {
         numWords = 0;
         comp_tag = new int[inNumWords];
         words = new String[inNumWords];
         wordsLens = new int[inNumWords];
         wordsCnts = new int[inNumWords];
         count = new double[inNumWords];
         degreesOfAmbig = new int[inNumWords];
         pr = new double[inNumWords][Tagger.MAX_TAGS];

         parseWords(inSent);
         processSentence();
    } // Sentence


    /** 
     * Get number of words for this sentence.
     *
     * @return Number of words in sentence
    **/

    public int getNumWords()
    {
        return(numWords);
    } // getNumWords


    /** 
     * Get occurrences count for word at requested position.
     *
     * @param  pos Position within count requesting
     * @return Value of count at the requested position
    **/

    public double getCountAt(int pos)
    {
        return(count[pos]);
    } // getCountAt


    /** 
     * Set occurrences count for word at requested position.
     *
     * @param  pos Position within count requesting
     * @param  val Value to set count[pos]
    **/

    public void setCountAt(int pos, double val)
    {
        count[pos] = val;
    } // setCountAt


    /** 
     * Get pr value (tag probability) at position pos1, pos2.
     *
     * @param  pos1 Position within pr requesting (word)
     * @param  pos2 Position within pr requesting (tag)
     * @return Value from pr array at the requested positions
    **/

    public double getPrAt(int pos1, int pos2)
    {
        return(pr[pos1][pos2]);
    } // getPrAt


    /** 
     * Get pr value (tag probability) at position pos1, pos2.
     *
     * @param  pos1 Position within pr requesting (word)
     * @param  pos2 Position within pr requesting (tag)
     * @param  val Value to set pr[pos1][pos2]
    **/

    public void setPrAt(int pos1, int pos2, double val)
    {
        pr[pos1][pos2] = val;
    } // setPrAt


    /** 
     * Get word at position pos.
     *
     * @param  pos Position within words array requesting
     * @return Word at the requested position
    **/

    public String getWordAt(int pos)
    {
        return(words[pos]);
    } // getWordAt


    /** 
     * Get word length at position pos.
     *
     * @param  pos Position within words array requesting
     * @return Word length at the requested position
    **/

    public int getWordLenAt(int pos)
    {
        return(wordsLens[pos]);
    } // getWordLenAt


    /** 
     * Get number of words at position pos (may have multi-word).
     *
     * @param  pos Position within words array requesting
     * @return Number of words at the requested position
    **/

    public int getNumWordWords(int pos)
    {
        return(wordsCnts[pos]);
    } // getNumWordWords


    /** 
     * Print tagging for sentence to designated output stream.
     *
     * @param  printed_prolog Toggle flag shows if already printed info
     * @param  out            Output stream to print info
     * @return Value of printed_prolog after routine runs
    **/

    public boolean print(boolean printed_prolog, PrintStream out)
    {
       boolean rtn = printed_prolog;

       for(int i = 0; i < numWords; i++)
       {
 
          String tmp_tag = "UNTAGGED";
          if(comp_tag[i] > -1)
            tmp_tag = Tags.getTagStrAt(comp_tag[i]);
          String results[][] = 
               MedPostSKRTranslator.translate(words[i], tmp_tag, wordsCnts[i]);

          for(int w = 0; w < results.length; w++)
          {
             if(rtn)
               out.println(",");
             out.print(" ['" + results[w][0] + "', '" + results[w][1] + "']");
             rtn = true;
          } // for w
       } // for

       return(rtn);
    } // print


    /** 
     * Print tagging for sentence to string.
     *
     * @param  printed_prolog Toggle flag shows if already printed info
     * @return Output results from this sentence
    **/

    public String printToString(boolean printed_prolog)
    {
       StringBuffer tmp = new StringBuffer();

       for(int i = 0; i < numWords; i++)
       {
 
          String tmp_tag = "UNTAGGED";
          if(comp_tag[i] > -1)
            tmp_tag = Tags.getTagStrAt(comp_tag[i]);

          String lres[][];

          if(Tagger.usePennTreebankTags())
            lres = PennTranslator.translate(words[i], tmp_tag, wordsCnts[i]);

          else if(Tagger.useMedPostTags())
            lres = MedPostTranslator.translate(words[i], tmp_tag, wordsCnts[i]);

          else
            lres = MedPostSKRTranslator.translate(words[i], tmp_tag,
                                                   wordsCnts[i]);

          for(int w = 0; w < lres.length; w++)
          {
             if(Tagger.usePrologOutput())
             {
                if(printed_prolog || (tmp.length() > 0))
                  tmp.append(",\n");
                tmp.append(" ['");
                tmp.append(lres[w][0]);
                tmp.append("', '");
                tmp.append(lres[w][1]);

                if(Tagger.usePrologFull() && (degreesOfAmbig[i] > 1))
                {
                   tmp.append("/");
                   tmp.append(degreesOfAmbig[i]);
                } // fi
                tmp.append("']");
             } /* fi */

             else
             {
                tmp.append(lres[w][0]);
                tmp.append("^");
                tmp.append(lres[w][1]);
                tmp.append("\n");
             } /* else */
          } // for w
       } // for

       return(tmp.toString());
    } // printToString


    // PRIVATE METHODS  ------------------------------------------------


    /** 
     * Parse sentence into individual words and multi-word words.
     *
     * @param  inSent Sentence to be parsed
    **/

    private void parseWords(String inSent)
    {
         int len = inSent.length();
         int lpos, rpos, tpos;
         String rest = "";

         numWords = 0;
         lpos = 0;
         rpos = 0;

         // Break at whitespace until we reach the end of the sentence.

         while(lpos < len)
         {
             while((lpos < len) && Character.isWhitespace(inSent.charAt(lpos)))
                lpos++;

             if(lpos < (len - 1))
             {
                rpos = lpos;
                while((rpos < len) && 
                      !Character.isWhitespace(inSent.charAt(rpos)))
                   rpos++;

                String tmp1 = inSent.substring(lpos, rpos);

                // If there is a tag separator, get the tag

                if(tmp1.charAt(0) == '_')
                  tpos = tmp1.indexOf("_UNTAGGED");
                else
                  tpos = tmp1.indexOf('_');

                wordsLens[numWords] = tpos;
                wordsCnts[numWords] = 1;
                String tmp = tmp1.substring(0, tpos);
                rest = tmp1.substring(tpos + 1);

                if(tpos > 0)
                {
                    // This part of the code is walking through the words
                    // in words trying to match up the words to multi-words
                    // in the Lexicon.  For example, "up" "to" maps to "up to".
                    // If there is a match, make the beginning word the entire
                    // idiom and then move the remaining words to the left the
                    // appropriate number of locations and reset length and
                    // number of words as appropriate.

                    boolean found = false;
                    if((numWords > 0) && (tpos < 12) &&
                       Character.isLetter(tmp.charAt(0)) &&
                       (wordsLens[numWords - 1] > 0) &&
                       (wordsLens[numWords - 1] < 12) &&
                       Character.isLetter(words[numWords - 1].charAt(0)))
                    {
                        StringBuffer idiom = new StringBuffer();
                        idiom.append(words[numWords - 1]).append(" ");
                        idiom.append(tmp);
                        String sIdiom = idiom.toString();
                        if(Tagger.lex.exists(sIdiom))
                        {
                           words[numWords - 1] = sIdiom;
                           wordsLens[numWords - 1] += tpos + 1;
                           wordsCnts[numWords - 1]++;
                           numWords--;
                           found = true;
                        } // fi
                    } // fi dual-words

                    if(!found && (numWords > 1) && (tpos < 12) &&
                       Character.isLetter(tmp.charAt(0)) &&
                       (wordsLens[numWords - 1] > 0) &&
                       (wordsLens[numWords - 1] < 12) &&
                       Character.isLetter(words[numWords - 1].charAt(0)) &&
                       (wordsLens[numWords - 2] > 0) &&
                       (wordsLens[numWords - 2] < 12) &&
                       Character.isLetter(words[numWords - 2].charAt(0)))
                    {
                        StringBuffer idiom = new StringBuffer();
                        idiom.append(words[numWords - 2]).append(" ");
                        idiom.append(words[numWords - 1]).append(" ");
                        idiom.append(tmp);
                        String sIdiom = idiom.toString();
                        if(Tagger.lex.exists(sIdiom))
                        {
                           words[numWords - 2] = sIdiom;
                           wordsLens[numWords - 2] += 
                                      wordsLens[numWords - 1] + tpos + 2;
                           wordsCnts[numWords - 2] += 
                                                wordsCnts[numWords - 1] + 1;
                           numWords -= 2;
                           found = true;
                        } // fi
                    } // fi tri-words

                    if(!found && (numWords > 2) && (tpos < 12) &&
                       Character.isLetter(tmp.charAt(0)) &&
                       (wordsLens[numWords - 1] > 0) &&
                       (wordsLens[numWords - 1] < 12) &&
                       Character.isLetter(words[numWords - 1].charAt(0)) &&
                       (wordsLens[numWords - 2] > 0) &&
                       (wordsLens[numWords - 2] < 12) &&
                       Character.isLetter(words[numWords - 2].charAt(0)) &&
                       (wordsLens[numWords - 3] > 0) &&
                       (wordsLens[numWords - 3] < 12) &&
                       Character.isLetter(words[numWords - 3].charAt(0)))
                    {
                        StringBuffer idiom = new StringBuffer();
                        idiom.append(words[numWords - 3]).append(" ");
                        idiom.append(words[numWords - 2]).append(" ");
                        idiom.append(words[numWords - 1]).append(" ");
                        idiom.append(tmp);
                        String sIdiom = idiom.toString();
                        if(Tagger.lex.exists(sIdiom))
                        {
                           words[numWords - 3] = sIdiom;
                           wordsLens[numWords - 3] += 
                                      wordsLens[numWords - 2] +
                                      wordsLens[numWords - 1] + tpos + 3;
                           wordsCnts[numWords - 3] += 
                                  wordsCnts[numWords - 2] + 
                                            wordsCnts[numWords - 1] + 1;
                           numWords -= 3;
                           found = true;
                        } // fi
                    } // fi quad-words

                    if(!found)
                      words[numWords] = tmp;

                    // End sentence at period-tag

                    if(rest.charAt(0) == '.')
                       rpos = len + 1;

                    numWords++;
                } // fi tpos > 0

                lpos = rpos;
             } // fi lpos
         } // while
    } // parseWords


    /** 
     * Process the sentence to derive tags for the parsed words.
    **/

    private void processSentence()
    {
        Tags tags = new Tags(numWords);

        for(int i = 0; i < numWords; i++)
        {
           String word = getWordAt(i);

           // Lowercase the first letter of first word if remaining letters
           // of the first word are all lowercase.

           if(i == 0)
           {
               boolean ok = true;
               for(int j = 1; ok && (j < word.length()); j++)
                 if(Character.isUpperCase(word.charAt(j)))
                   ok = false;

               if(ok)
                 word = word.toLowerCase();
           } // fi first word

           // Get the lexical entry information for this word and setup
           // preliminary probabilities.

           String entry = Tagger.lex.getLexEntry(word);
           setCountAt(i, Tagger.lex.getLastCount());
           degreesOfAmbig[i] = Tagger.lex.getLastAmbiguityDegree();
           fillTmpD(entry, tags);

           for(int t = 0; t < num_tags; t++)
              tags.setDwAt(i, t, tmpD[t]);
        } // for each word

        tags.normalize(numWords, lex_backoff, count);

        // Check post-lexicon constraints

        for(int i = 0; i < numWords; i++)
        {
           // Scores of 1,000 or higher mean the lexicon entry is definite

           if(getCountAt(i) < 999.0)
             tags.checkTags(getWordAt(i), i, getWordLenAt(i));
        } // for each word

        for(int i = 0; i < numWords; i++)
           for(int j = 0; j < num_tags; j++)
              setPrAt(i, j, tags.getDwAt(i, j));

        // Forward-backward algorithms to generate tag probabilities

        Predict predictor = new Predict(numWords);

        predictor.compute(Tagger.ngram.getPrTag(), tags.getDw(), numWords);

        // Compute maximum likelihood for each tag, and normalize!

        for(int r = 0; r < numWords; r++)
        {
           Arrays.fill(pr[r], 0.0); // Initializing pr[r][num_tags] to 0.0
           double v = 0.0;
           for(int i = 0; i < num_states; i++)
           {
               int pos = (r * num_states) + i;
               pr[r][i] += predictor.getAlphaAt(pos) * predictor.getBetaAt(pos);
               v += pr[r][i];
           } // for

           for(int i = 0; i < num_tags; i++)
              pr[r][i] /= v;
        } // for r

        // Compute the maximum likelihood tag based on whatever is in
        // the pr field.

        Arrays.fill(comp_tag, -1);

        for(int i = 0; i < numWords; i++)
        {
           double m = 0.0;
           for(int j = 0; j < num_tags; j++)
           {
              double pr = getPrAt(i,j);
              if(pr > m)
              {
                  m = pr;
                  comp_tag[i] = j;
              } // fi
           } // for j
        } // for i
    } // processSentence


    /** 
     * Fill/Initialize the temporary probabilities array.
     *
     * @param  entry Lexical entry for the currently processing word(s)
     * @param  tags  Link to the tags instance
    **/

    private void fillTmpD(String entry, Tags tags)
    {
       // Initialize tmpD array to 0.0

       Arrays.fill(tmpD, 0.0);

       // Go through entry and only work on items from entry

       StringTokenizer st = new StringTokenizer(entry, "_");
       while(st.hasMoreTokens())
       {
          int posT = -1;
          double val = 1000.0;

          String wordT = st.nextToken();
          int pos = wordT.indexOf(':', 1);
          if(pos == -1)
            posT = tags.getTagPos(wordT);
          else
          {
             String tmp = wordT.substring(0, pos);
             val = Double.parseDouble(wordT.substring(pos + 1));
             posT = tags.getTagPos(tmp);
          } // else

          if(posT > -1)
            tmpD[posT] = val;
       } // while
    } // fillTmpD
} // class Sentence
