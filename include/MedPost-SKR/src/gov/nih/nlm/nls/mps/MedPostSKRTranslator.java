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

import java.util.Map;
import java.util.HashMap;


/**
 * Translates MedPost tags into SPECIALIST Lexicon tags.
 * <p>
 * This translator is solely concerned with translating the MedPost assigned
 * tags into tags from the SPECIALIST Lexicon tag set.  The translator
 * handles both the one-to-one matchup as well as the multi-word matchup
 * using a series of HashMaps to identify the translation mappings.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	3.0, January 23, 2006
 * @since	1.0
**/

/* History:
 *    3.0, January 23, 2006
 *      - Changed from Translator to MedPostSKRTranslator to differentiate
 *        between the other PennTreebankTranslator and MedPostTranslator.
 *
 *    2.0, November 29, 2005
 *      - Added "have been" and "have used" to multi-word translation.
 *
 *    1.0, August 18, 2004
 *      - Epoch
**/

public class MedPostSKRTranslator
{
    private static String rtn[][];

    private static Map multiWordTrans;  // Map multi-word elements
    private static Map punctTrans;      // Map punctuation elements
    private static Map tagTrans;        // Map remaining tag elements

    // NOTE: I rounded up the number of entries and divided by default
    //     load factor of .75 to get the initial capacity.

    static {
       multiWordTrans = new HashMap(100);
       multiWordTrans.put("a priori",          "prep noun");
       multiWordTrans.put("according to",      "verb prep");
       multiWordTrans.put("ad libitum",        "prep noun");
       multiWordTrans.put("ahead of",          "adv prep");
       multiWordTrans.put("apart from",        "adv prep");
       multiWordTrans.put("as if",             "prep prep");
       multiWordTrans.put("as far as",         "prep adv prep");
       multiWordTrans.put("as for",            "prep prep");
       multiWordTrans.put("as per",            "prep prep");
       multiWordTrans.put("as long as",        "prep adv prep");
       multiWordTrans.put("as of",             "prep prep");
       multiWordTrans.put("as soon as",        "prep adv prep");
       multiWordTrans.put("as though",         "prep prep");
       multiWordTrans.put("as to",             "prep prep");
       multiWordTrans.put("as well",           "prep adj");
       multiWordTrans.put("as well as",        "prep adv prep");
       multiWordTrans.put("as yet",            "prep adv");
       multiWordTrans.put("aside from",        "adv prep");
       multiWordTrans.put("assuming that",     "verb compl");
       multiWordTrans.put("at least",          "prep adj");
       multiWordTrans.put("at most",           "prep adv");
       multiWordTrans.put("away from",         "adv prep");
       multiWordTrans.put("because of",        "prep prep");
       multiWordTrans.put("considering that",  "verb compl");
       multiWordTrans.put("contrary to",       "adj prep");
       multiWordTrans.put("de novo",           "prep noun");
       multiWordTrans.put("devoid of",         "adj prep");
       multiWordTrans.put("due to",            "adj prep");
       multiWordTrans.put("except for",        "verb prep");
       multiWordTrans.put("except that",       "verb compl");
       multiWordTrans.put("excepting that",    "verb compl");
       multiWordTrans.put("exclusive of",      "adj prep");
       multiWordTrans.put("given that",        "verb compl");
       multiWordTrans.put("granted that",      "verb compl");
       multiWordTrans.put("granting that",     "verb compl");
       multiWordTrans.put("has been",          "aux aux");
       multiWordTrans.put("has used",          "aux verb");
       multiWordTrans.put("in as much as",     "prep prep adv prep");
       multiWordTrans.put("in gel",            "prep noun");
       multiWordTrans.put("in order that",     "prep noun compl");
       multiWordTrans.put("in the event that", "prep det noun compl");
       multiWordTrans.put("in situ",           "prep noun");
       multiWordTrans.put("in utero",          "prep noun");
       multiWordTrans.put("in vitro",          "prep noun");
       multiWordTrans.put("in vivo",           "prep noun");
       multiWordTrans.put("inasmuch as",       "conj prep");
       multiWordTrans.put("inside of",         "noun prep");
       multiWordTrans.put("insofar as",        "conj prep");
       multiWordTrans.put("instead of",        "adv prep");
       multiWordTrans.put("irrespective of",   "adv prep");
       multiWordTrans.put("just as",           "adv prep");
       multiWordTrans.put("next to",           "adv prep");
       multiWordTrans.put("other than",        "adj prep");
       multiWordTrans.put("out of",            "adv prep");
       multiWordTrans.put("outside of",        "adv prep");
       multiWordTrans.put("owing to",          "verb prep");
       multiWordTrans.put("per cell",          "prep noun");
       multiWordTrans.put("per se",            "prep noun");
       multiWordTrans.put("preparatory to",    "adj prep");
       multiWordTrans.put("previous to",       "adj prep");
       multiWordTrans.put("prior to",          "adv prep");
       multiWordTrans.put("provided that",     "verb compl");
       multiWordTrans.put("providing that",    "verb compl");
       multiWordTrans.put("pursuant to",       "adj prep");
       multiWordTrans.put("rather than",       "adv prep");
       multiWordTrans.put("regardless of",     "adv prep");
       multiWordTrans.put("seeing that",       "verb compl");
       multiWordTrans.put("so that",           "prep compl");
       multiWordTrans.put("subsequent to",     "adj prep");
       multiWordTrans.put("such as",           "adj prep");
       multiWordTrans.put("such that",         "adj compl");
       multiWordTrans.put("supposing that",    "verb compl");
       multiWordTrans.put("up to",             "prep prep");
       multiWordTrans.put("vice versa",        "adv adv");

       punctTrans = new HashMap(40);
       punctTrans.put(".",    "pd");
       punctTrans.put("-",    "hy");
       punctTrans.put("--",   "hy");
       punctTrans.put("'",    "ap");
       punctTrans.put("_",    "un");
       punctTrans.put("$",    "dl");
       punctTrans.put("~",    "tl");
       punctTrans.put("|",    "ba");
       punctTrans.put("#",    "nm");
       punctTrans.put("@",    "at");
       punctTrans.put("!",    "ex");
       punctTrans.put(",",    "cm");
       punctTrans.put(")",    "rp");
       punctTrans.put("(",    "lp");
       punctTrans.put("%",    "pc");
       punctTrans.put("+",    "pl");
       punctTrans.put(":",    "cl");
       punctTrans.put(";",    "sc");
       punctTrans.put("]",    "rk");
       punctTrans.put("[",    "lk");
       punctTrans.put("<",    "ls");
       punctTrans.put(">",    "gr");
       punctTrans.put("=",    "eq");
       punctTrans.put("&",    "am");
       punctTrans.put("?",    "qu");
       punctTrans.put("*",    "ax");
       punctTrans.put("\"",   "dq");
       punctTrans.put("/",    "sl");
       punctTrans.put("`",    "bq");

       tagTrans = new HashMap(80);
       tagTrans.put("NN",   "noun");
       tagTrans.put("NNS",  "noun");
       tagTrans.put("NNP",  "noun");
       tagTrans.put("VVGN", "noun");
       tagTrans.put("II",   "prep");
       tagTrans.put("CSN",  "prep");
       tagTrans.put("DD",   "det");
       tagTrans.put("DB",   "det");
       tagTrans.put("CC",   "conj");
       tagTrans.put("CS",   "conj");
       tagTrans.put("CST",  "compl");
       tagTrans.put("MC",   "num");
       tagTrans.put("VVNJ", "adj");
       tagTrans.put("VVGJ", "adj");
       tagTrans.put("JJ",   "adj");
       tagTrans.put("JJT",  "adj");
       tagTrans.put("JJR",  "adj");
       tagTrans.put("RR",   "adv");
       tagTrans.put("RRR",  "adv");
       tagTrans.put("RRT",  "adv");
       tagTrans.put("PN",   "pron");
       tagTrans.put("PND",  "pron");
       tagTrans.put("PNR",  "pron");
       tagTrans.put("PNG",  "pron");
       tagTrans.put("VM",   "modal");
       tagTrans.put("VBB",  "aux");
       tagTrans.put("VBD",  "aux");
       tagTrans.put("VBG",  "aux");
       tagTrans.put("VBI",  "aux");
       tagTrans.put("VBN",  "aux");
       tagTrans.put("VBZ",  "aux");
       tagTrans.put("VDB",  "aux");
       tagTrans.put("VDD",  "aux");
       tagTrans.put("VDG",  "aux");
       tagTrans.put("VDI",  "aux");
       tagTrans.put("VDN",  "aux");
       tagTrans.put("VDZ",  "aux");
       tagTrans.put("VHB",  "aux");
       tagTrans.put("VHD",  "aux");
       tagTrans.put("VHG",  "aux");
       tagTrans.put("VHI",  "aux");
       tagTrans.put("VHN",  "aux");
       tagTrans.put("VHZ",  "aux");
       tagTrans.put("VVB",  "verb");
       tagTrans.put("VVD",  "verb");
       tagTrans.put("VVG",  "verb");
       tagTrans.put("VVI",  "verb");
       tagTrans.put("VVN",  "verb");
       tagTrans.put("VVZ",  "verb");
       tagTrans.put("TO",   "adv");
       tagTrans.put("EX",   "adv");
       tagTrans.put("GE",   "noun");
       tagTrans.put("SYM",  "noun");
       tagTrans.put("''",   "ap");
    };

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * Default constructor does nothing. 
    **/

    public MedPostSKRTranslator() {}


    /**
     * Coordinate translation from MedPost tagging to Specialist tagging.
     *
     * @param    inWordStr Word(s) to be tagged
     * @param    inTag     Initial recommended tag(s)
     * @param    wordCnt   Number of words (single or multi-word)
     * @return   Two dimensional array with text/tag matches
    **/

    public static String[][] translate(String inWordStr, String inTag,
                                       int wordCnt)
    {
        rtn = new String[wordCnt][wordCnt + 1];

        // Is this a multi-word?

        if(wordCnt > 1)
          checkMultiWords(inWordStr, inTag, wordCnt);

        // Must be a single word

        else
        {
            rtn[0][0] = inWordStr;
            if(inTag.equals("UNTAGGED"))  // Shouldn't happen
               rtn[0][1] = "UNTAGGED";
            else
               rtn[0][1] = findTrans(inWordStr, inTag);
        } // else

        return(rtn);
    } // MedPostSKRTranslator


    // PRIVATE METHODS  ------------------------------------------------


    /**
     * Lookup the single inWord in the punctTrans HashMap or lookup the inTag
     * in the tagTrans HashMap to find the the translation.
     *
     * @param    inWord Word to be tagged
     * @param    inTag  Tag to be translated
     * @return   The matching tag if found in the lookup, otherwise nothing
    **/

    // This routine handles the translation from MedPost to Specialist

    private static String findTrans(String inWord, String inTag)
    {
        String localRtn = inTag;

        // First look in the punctation HashMap to see if we have a match.

        Object o = punctTrans.get(inWord);

        if(o == null)
        {
           char ch = inWord.charAt(0);

           // Make sure we don't make a non-number word a number.

           if(!Character.isDigit(ch) && inTag.equals("MC"))
              localRtn = "noun";

           // Else, lookup the tag in the normal tag HashMap

           else
           {
              Object o2 = tagTrans.get(inTag);
              if(o2 != null)
                localRtn = o2.toString();
           } // else
        } // fi

        // Found a match in the punctuation HashMap

        else
          localRtn = o.toString();

        return(localRtn);
    } // findTrans


    /**
     * Lookup the inWord in the multiWordTrans HashMap to find the
     * the translation.
     *
     * @param    inWord Word(s) to be tagged
     * @return   The matching tag if found in the lookup, otherwise nothing
    **/

    private static String findMultiTrans(String inWord)
    {
        String localRtn = "";
        Object o = multiWordTrans.get(inWord.toLowerCase());

        if(o != null)
          localRtn = o.toString();

        return(localRtn);
    } // findMultiTrans


    /**
     * Translate multi-words into individual words and tags.
     *
     * @param    inWordStr Word(s) to be tagged
     * @param    inTag     Initial recommended tag(s)
     * @param    wordCnt   Number of words (single or multi-word)
    **/

    private static void checkMultiWords(String inWordStr, String inTag,
                                                                    int wordCnt)
    {
        String tags = findMultiTrans(inWordStr);

        // We want the translated tag if we didn't have a match. If we
        // don't have a match in the lookup table, in theory, we shouldn't
        // have a multi-word.  If we don't have a lookup match, use the tag
        // that was sent to the routine.

        String tmp_tags = inTag;

        if(tags.length() > 0)
          tmp_tags = tags;

        String tmp = inWordStr;
        int pos = 0;

        // Now, we need to go through both the word string and the tag string
        // and separate out the individual words/tags since the Specialist
        // Lexicon tag set doesn't recognize multi-words.  If we run out of
        // tags before we run out of words, just keep using the last tag for
        // subsequent words.

        while(tmp.length() > 0)
        {
            // Word list first

            String t1 = "";
            String rest = "";
            int p = tmp.indexOf(' ');  // look for spaces between words
            if(p > -1)
            {
                t1 = tmp.substring(0, p);
                rest = tmp.substring(p + 1);
            } // fi
            else
             t1 = tmp;

            // Tag list next

            String t2 = "";
            String rest2 = "";
            int p2 = tmp_tags.indexOf(' ');  // look for spaces between words
            if(p2 > -1)
            {
                t2 = tmp_tags.substring(0, p2);
                rest2 = tmp_tags.substring(p2 + 1);
            } // fi
            else
             t2 = tmp_tags;

            if(rest2.length() > 0)  // If we don't > 1, keep last one
              tmp_tags = rest2;

            // Put resultant word and tag into the global rtn array.

            rtn[pos][0] = t1;
            rtn[pos][1] = t2;
            pos++;

            tmp = rest;
        } // while
    } // checkMultiWords
} // class MedPost_SKRTranslator
