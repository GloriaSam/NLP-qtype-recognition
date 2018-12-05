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
 * Cleans up the MedPost tags.
 * <p>
 * This Translator is solely concerned with putting the MedPost tags into
 * a format that the print routines can handle along with the other translators.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	1.0, January 23, 2006
 * @since	1.0
**/

public class MedPostTranslator
{
    private static String rtn[][];

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * Default constructor does nothing. 
    **/

    public MedPostTranslator() {}


    /**
     * Coordinate translation from MedPost tagging to MedPost tagging.
     *
     * @param    inWordStr Word(s) to be tagged
     * @param    inTag     Initial recommended tag(s)
     * @param    wordCnt   Number of words (single or multi-word)
     * @return   Two dimensional array with text/tag matches
    **/

    public static String[][] translate(String inWordStr, String inTag,
                                       int wordCnt)
    {
        rtn = new String[1][2];

        rtn[0][0] = inWordStr;
        rtn[0][1] = inTag;

        return(rtn);
    } // Translate


    // PRIVATE METHODS  ------------------------------------------------

} // class MedPostTranslator
