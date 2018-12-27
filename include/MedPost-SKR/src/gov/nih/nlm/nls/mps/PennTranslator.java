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
 * Translates MedPost tags into Penn Treebank tags.
 * <p>
 * The Translator is solely concerned with translating the MedPost assigned
 * tags into tags from the Penn Treebank tag set.  The Translator
 * handles both the one-to-one matchup as well as the multi-word matchup
 * using a series of HashMaps to identify the translation mappings.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	1.0, January 23, 2006
 * @since	1.0
**/

public class PennTranslator
{
    private static String rtn[][];

    private static Map singleWordTrans; // Map single-word elements
    private static Map multiWordTrans;  // Map multi-word elements
    private static Map punctTrans;      // Map punctuation elements
    private static Map tagTrans;        // Map remaining tag elements

    // NOTE: I rounded up the number of entries and divided by default
    //     load factor of ~ .75 to get the initial capacity.

    static {
       singleWordTrans = new HashMap(14);
       singleWordTrans.put("to",        "TO");
       singleWordTrans.put("who",       "WP");
       singleWordTrans.put("what",      "WP");
       singleWordTrans.put("whom",      "WP");
       singleWordTrans.put("how",       "WRB");
       singleWordTrans.put("however",   "WRB");
       singleWordTrans.put("when",      "WRB");
       singleWordTrans.put("whenever",  "WRB");
       singleWordTrans.put("where",     "WRB");
       singleWordTrans.put("whereby",   "WRB");
       singleWordTrans.put("why",       "WRB");

       multiWordTrans = new HashMap(100);
       multiWordTrans.put("a priori",          "FW FW");
       multiWordTrans.put("according to",      "VBG TO");
       multiWordTrans.put("ad libitum",        "FW FW");
       multiWordTrans.put("ahead of",          "RB IN");
       multiWordTrans.put("apart from",        "RB IN");
       multiWordTrans.put("as if",             "IN IN");
       multiWordTrans.put("as far as",         "RB RB IN");
       multiWordTrans.put("as for",            "IN IN");
       multiWordTrans.put("as per",            "IN IN");
       multiWordTrans.put("as long as",        "RB RB IN");
       multiWordTrans.put("as of",             "IN IN");
       multiWordTrans.put("as soon as",        "RB RB IN");
       multiWordTrans.put("as though",         "IN IN");
       multiWordTrans.put("as to",             "IN TO");
       multiWordTrans.put("as well",           "RB RB");
       multiWordTrans.put("as well as",        "RB RB IN");
       multiWordTrans.put("as yet",            "RB RB");
       multiWordTrans.put("aside from",        "RB IN");
       multiWordTrans.put("assuming that",     "VBG IN");
       multiWordTrans.put("at least",          "IN JJS");
       multiWordTrans.put("at most",           "IN RBS");
       multiWordTrans.put("away from",         "RB IN");
       multiWordTrans.put("because of",        "IN IN");
       multiWordTrans.put("considering that",  "VBG IN");
       multiWordTrans.put("contrary to",       "JJ TO");
       multiWordTrans.put("de novo",           "FW FW");
       multiWordTrans.put("devoid of",         "JJ IN");
       multiWordTrans.put("due to",            "JJ TO");
       multiWordTrans.put("except for",        "IN IN");
       multiWordTrans.put("except that",       "IN IN");
       multiWordTrans.put("excepting that",    "VBG IN");
       multiWordTrans.put("exclusive of",      "JJ IN");
       multiWordTrans.put("given that",        "VBN IN");
       multiWordTrans.put("granted that",      "VBN IN");
       multiWordTrans.put("granting that",     "VBG IN");
       multiWordTrans.put("has been",          "VBZ VBN");
       multiWordTrans.put("has used",          "VBZ VBN");
       multiWordTrans.put("in as much as",     "IN RB RB IN");
       multiWordTrans.put("in gel",            "IN NN");
       multiWordTrans.put("in order that",     "IN NN IN");
       multiWordTrans.put("in the event that", "IN DT NN IN");
       multiWordTrans.put("in situ",           "FW FW");
       multiWordTrans.put("in utero",          "FW FW");
       multiWordTrans.put("in vitro",          "FW FW");
       multiWordTrans.put("in vivo",           "FW FW");
       multiWordTrans.put("inasmuch as",       "RB IN");
       multiWordTrans.put("inside of",         "NN IN");
       multiWordTrans.put("insofar as",        "RB IN");
       multiWordTrans.put("instead of",        "RB IN");
       multiWordTrans.put("irrespective of",   "RB IN");
       multiWordTrans.put("just as",           "RB IN");
       multiWordTrans.put("next to",           "JJ TO");
       multiWordTrans.put("other than",        "JJ IN");
       multiWordTrans.put("out of",            "IN IN");
       multiWordTrans.put("outside of",        "IN IN");
       multiWordTrans.put("owing to",          "JJ TO");
       multiWordTrans.put("per cell",          "IN NN");
       multiWordTrans.put("per se",            "FW FW");
       multiWordTrans.put("preparatory to",    "JJ TO");
       multiWordTrans.put("previous to",       "JJ TO");
       multiWordTrans.put("prior to",          "RB TO");
       multiWordTrans.put("provided that",     "VBN IN");
       multiWordTrans.put("providing that",    "VBG IN");
       multiWordTrans.put("pursuant to",       "JJ TO");
       multiWordTrans.put("rather than",       "RB IN");
       multiWordTrans.put("regardless of",     "RB IN");
       multiWordTrans.put("seeing that",       "VBG IN");
       multiWordTrans.put("so that",           "IN IN");
       multiWordTrans.put("subsequent to",     "JJ TO");
       multiWordTrans.put("such as",           "JJ IN");
       multiWordTrans.put("such that",         "JJ IN");
       multiWordTrans.put("supposing that",    "VBG IN");
       multiWordTrans.put("up to",             "IN TO");
       multiWordTrans.put("vice versa",        "RB RB");

       // NOT A TRUE MAPPING TO PENN TREEBANK WITH PUNCTUATION/SYMBOLS

       punctTrans = new HashMap(40);
       punctTrans.put(".",    ".");
       punctTrans.put("-",    "-");
       punctTrans.put("--",   "-");
       punctTrans.put("'",    "'");
       punctTrans.put("_",    "_");
       punctTrans.put("$",    "$");
       punctTrans.put("~",    "~");
       punctTrans.put("|",    "|");
       punctTrans.put("#",    "#");
       punctTrans.put("@",    "@");
       punctTrans.put("!",    "!");
       punctTrans.put(",",    ",");
       punctTrans.put(")",    ")");
       punctTrans.put("(",    "(");
       punctTrans.put("%",    "%");
       punctTrans.put("+",    "+");
       punctTrans.put(":",    ":");
       punctTrans.put(";",    ";");
       punctTrans.put("]",    "]");
       punctTrans.put("[",    "[");
       punctTrans.put("<",    "<");
       punctTrans.put(">",    ">");
       punctTrans.put("=",    "=");
       punctTrans.put("&",    "&");
       punctTrans.put("?",    ".");
       punctTrans.put("*",    "*");
       punctTrans.put("\"",   "\"");
       punctTrans.put("/",    "/");
       punctTrans.put("`",    "`");

       tagTrans = new HashMap(80);
       tagTrans.put("NN",   "NN");
       tagTrans.put("NNS",  "NNS");
       tagTrans.put("NNP",  "NNP");
       tagTrans.put("VVGN", "NN");
       tagTrans.put("II",   "IN");
       tagTrans.put("CSN",  "IN");
       tagTrans.put("DD",   "DT");
       tagTrans.put("DB",   "PDT");
       tagTrans.put("CC",   "CC");
       tagTrans.put("CS",   "IN");
       tagTrans.put("CST",  "IN");
       tagTrans.put("MC",   "CD");
       tagTrans.put("VVNJ", "JJ");
       tagTrans.put("VVGJ", "JJ");
       tagTrans.put("JJ",   "JJ");
       tagTrans.put("JJT",  "JJS");
       tagTrans.put("JJR",  "JJR");
       tagTrans.put("RR",   "RB");
       tagTrans.put("RRR",  "RBR");
       tagTrans.put("RRT",  "RBS");
       tagTrans.put("PN",   "PRP");
       tagTrans.put("PND",  "PRP");
       tagTrans.put("PNR",  "WDT");
       tagTrans.put("PNG",  "PRP$");
       tagTrans.put("VM",   "MD");
       tagTrans.put("VBB",  "VBP");
       tagTrans.put("VBD",  "VBD");
       tagTrans.put("VBG",  "VBG");
       tagTrans.put("VBI",  "VB");
       tagTrans.put("VBN",  "VBN");
       tagTrans.put("VBZ",  "VBZ");
       tagTrans.put("VDB",  "VBP");
       tagTrans.put("VDD",  "VBD");
       tagTrans.put("VDG",  "VBG");
       tagTrans.put("VDI",  "VB");
       tagTrans.put("VDN",  "VBN");
       tagTrans.put("VDZ",  "VBZ");
       tagTrans.put("VHB",  "VBP");
       tagTrans.put("VHD",  "VBD");
       tagTrans.put("VHG",  "VBG");
       tagTrans.put("VHI",  "VB");
       tagTrans.put("VHN",  "VBN");
       tagTrans.put("VHZ",  "VBZ");
       tagTrans.put("VVB",  "VBP");
       tagTrans.put("VVD",  "VBD");
       tagTrans.put("VVG",  "VBG");
       tagTrans.put("VVI",  "VB");
       tagTrans.put("VVN",  "VBN");
       tagTrans.put("VVZ",  "VBZ");
       tagTrans.put("TO",   "TO");
       tagTrans.put("EX",   "EX");
       tagTrans.put("GE",   "POS");
       tagTrans.put("SYM",  "NN");
       tagTrans.put("''",   "''");
    };

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * Default constructor does nothing. 
    **/

    public PennTranslator() {}


    /**
     * Coordinate translation from MedPost tagging to Penn Treebank tagging.
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
    } // Translate


    // PRIVATE METHODS  ------------------------------------------------


    /**
     * Lookup the single inWord in the punctTrans HashMap or lookup the inTag
     * in the tagTrans HashMap to find the the translation.
     *
     * @param    inWord Word to be tagged
     * @param    inTag  Tag to be translated
     * @return   The matching tag if found in the lookup, otherwise nothing
    **/

    // This routine handles the translation of single words from MedPost
    // to Penn Treebank.

    private static String findTrans(String inWord, String inTag)
    {
        String localRtn = inTag;

        // First look in the punctation HashMap to see if we have a match.

        Object o = punctTrans.get(inWord);

        if(o == null)
        {
           Object w = singleWordTrans.get(inWord);

           if(w == null)
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
           } // fi singleWordTrans

           // Found a match in the singleWordTrans HashMap

           else
             localRtn = w.toString();
        } // fi punctTrans

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
} // class PennTranslator
