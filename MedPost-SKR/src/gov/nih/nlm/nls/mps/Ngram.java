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

import gov.nih.nlm.nls.mps.Tags;

import java.io.File;
import java.io.BufferedReader;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.IOException;


/**
 * Load and access the trained Ngram frequencies.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	2.0, June 9, 2005
 * @since	1.0
**/

/* History:
 *    2.0, June 9, 2005
 *      - Updated count0 definition to correspond to new Ngrams.cur file
 *
 *    1.0, August 18, 2004
 *      - Epoch
**/

public class Ngram
{
    private static final String ngramOne = 
      System.getProperty("ngramOne",
                         "/usr/local/MedPost-SKR/data/ngramOne.serial");


    private static int num_tags = Tags.num_tags;
    private static final double count0 = 101214;

    private double count1[] = new double[Tagger.MAX_TAGS];
    private double count2[] = new double[Tagger.MAX_TAGS * Tagger.MAX_TAGS];
    private double prTag[] = new double[Tagger.MAX_TAGS];

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * The default constructor which loads in the Ngram data 
    **/

    public Ngram()
    {
        scangram();
    } // Ngram


    /** 
     * Get prTag array.
     *
     * @return Array containing prTag.
    **/

    public double[] getPrTag()
    {
      return(prTag);
    } // getPrTag


    /** 
     * Get Ngram count0.
     *
     * @return count0 value
    **/

    public double getCount0()
    {
      return(count0);
    } // getCount0


    /** 
     * Get Ngram count1 at position pos.
     *
     * @param  pos Position within count1 requesting
     * @return Value of count1 at the requested position
    **/

    public double getCount1At(int pos)
    {
      return(count1[pos]);
    } // getCount1At


    /** 
     * Get Ngram count2 at position pos.
     *
     * @param  pos Position within count2 requesting
     * @return Value of count2 at the requested position
    **/

    public double getCount2At(int pos)
    {
      return(count2[pos]);
    } // getCount2At


    // PRIVATE METHODS  ------------------------------------------------


    /** 
     * Load Ngram data into the various data arrays.
     *
    **/

    private void scangram()
    {
        try
        {
           // Read in the serialized big_array which contains the three
           // arrays we are trying to initialize (count1, count2, and prTag).

           ObjectInputStream ois = 
                new ObjectInputStream(new FileInputStream(ngramOne));
           double big_array[] = (double [])ois.readObject();
           ois.close();

           // Now split out the separate array values.

           int pos = 0;
           for(int i = 0; i < num_tags; i++)
             count1[i] = big_array[pos++];

           for(int i = 0; i < (num_tags * num_tags); i++)
             count2[i] = big_array[pos++];

           for(int i = 0; i < num_tags; i++)
             prTag[i] = big_array[pos++];

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
    } // scangram
} // class Ngram
