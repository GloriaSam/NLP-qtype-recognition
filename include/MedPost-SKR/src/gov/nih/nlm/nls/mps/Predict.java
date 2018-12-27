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


/**
 * Predict probabilities for tags.
 *
 * @author	Larry Smith (original MedPost)
 * @author	Jim Mork (MedPost/SKR)
 * @version	1.0, August 18, 2004
 * @since	1.0
**/

public class Predict
{
    private static int num_states = 0;
    private static int num_tags = 0;
    private static double count0 = 0.0;
    private static double prState[] = new double[Tagger.MAX_STATES];
    private static double prTrans[][] = 
                              new double[Tagger.MAX_STATES][Tagger.MAX_STATES];

    /**
     * Static Initializer to initialize and smooth prState and prTrans values.
    **/

    static {
        init();
        smooth();
    };

    private double alphaArray[] = null;
    private double betaArray[] = null;

    // PUBLIC METHODS  ------------------------------------------------


    /**
     * Constructor allocates appropriate space for the alpha & beta arrays. 
     *
     * @param  numWords Number of words we are currently working on.
    **/

    public Predict(int numWords)
    {
         alphaArray = new double[numWords * Tagger.MAX_STATES];
         betaArray = new double[numWords * Tagger.MAX_STATES];
    } // Predict


    /** 
     * Get value at position pos in alphaArray.
     *
     * @param  pos Position within alphaArray requesting
     * @return Value of alphaArray at the requested position
    **/

    public double getAlphaAt(int pos)
    {
        return(alphaArray[pos]);
    } // getAlphaAt


    /** 
     * Get value at position pos in betaArray.
     *
     * @param  pos Position within betaArray requesting
     * @return Value of betaArray at the requested position
    **/

    public double getBetaAt(int pos)
    {
        return(betaArray[pos]);
    } // getBetaAt


    /** 
     * Compute probabilities for the word(s) - Forward-backward algorithms.
     *
     * @param  prTag     prTag array
     * @param  dw        dw array
     * @param  numWords  Number of words working on
    **/

    public void compute(double prTag[], double dw[][], int numWords)
    {
       double scale = 0.0;

       // Compute alpha (forward)

       for(int r = 0; r < numWords; r++)
       {
           scale = 0.0;
           for(int i = 0; i < num_states; i++)
           {
               int pos = (r * num_states) + i;
               alphaArray[pos] = 0.0;

               if(dw[r][i] > 0.0)
               {
                  if(r == 0)
                    alphaArray[pos] = (prState[i] * dw[r][i]) / prTag[i];
                  else
                  {
                     for(int k = 0; k < num_tags; k++)
                     {
                        alphaArray[pos] += 
                              alphaArray[((r - 1) * num_states) + k] * 
                                       (prTrans[k][i] * dw[r][i]) / prTag[i];
                     } // for
                  } // else

                  scale += alphaArray[pos];
               } // fi
           } // for

           for(int i = 0; i < num_states; i++)
             alphaArray[(r * num_states) + i] /= scale;
       } // for alpha

       // Compute beta (backwards)

       for(int r = (numWords - 1); r >= 0; r--)
       {
           scale = 0.0;
           for(int i = 0; i < num_states; i++)
           {
              int pos = (r * num_states) + i;
              betaArray[pos] = 0.0;

              if(r == (numWords - 1))
                 betaArray[pos] = 1.0;

              else
              {
                  for(int k = 0; k < num_tags; k++)
                  {
                      betaArray[pos] += 
                          ((prTrans[i][k] * dw[r + 1][k]) / prTag[k]) *
                                 betaArray[(r + 1) * num_states + k];
                  } // for k
              } // else

              scale += betaArray[pos];
           } // for i

           for(int i = 0; i < num_states; i++)
             betaArray[r * num_states + i] /= scale;
        } // for beta
    } // compute

    // PRIVATE METHODS  ------------------------------------------------


    /** 
     * Initialize the prState and prTrans arrays based on Ngram data.
    **/

    private static void init()
    {
        int i, j;

        num_tags = Tags.num_tags;
        num_states = num_tags;
        count0 = Tagger.ngram.getCount0();
        for(i = 0; i < num_states; i++)
        {
            Integer I = new Integer(num_states);
            double d = I.doubleValue();
            double count1 = Tagger.ngram.getCount1At(i);

            if(count0 > 0.0)
              prState[i] = count1 / count0;
            else
              prState[i] = 1.0 / d;

            for(j = 0; j < num_states; j++)
            {
               if(count1 > 0.0)
                 prTrans[i][j] = 
                    Tagger.ngram.getCount2At((i * num_tags) + j) / count1;
               else
                 prTrans[i][j] = 1.0 / d;
            } // for
        } // for
    } // init


    /** 
     * Smooth probability values.
    **/

    private static void smooth()
    {
        double p1[] = new double[Tagger.MAX_TAGS];
        double p2[] = new double[Tagger.MAX_TAGS * Tagger.MAX_TAGS];
        double N, T, Z, d, a, b, m, p;
        int k;

        N = 0.0;
        T = 0.0;
        Z = 0.0;
        for(int i = 0; i < num_tags; i++)
        {
            double count1 = Tagger.ngram.getCount1At(i);
            N += count1;
            if(count1 > 0.0)
              T++;
            else
              Z++;
        } // for
        d = N / (N + T);

        // The discounted probability for the 1-grams is not backed-off

        for(int i = 0; i < num_tags; i++)
        {
            double count1 = Tagger.ngram.getCount1At(i);
            if(count1 > 0.0)
              p1[i] = d * (count1 / count0);
            else
              p1[i] = (1.0 - d) / Z;
        } // for

        // The 2-gram probabilities

        for(int i = 0; i < num_tags; i++)
        {
            N = 0.0;
            T = 0.0;
            Z = 0.0;
            for(int j = 0; j < num_tags; j++)
            {
               k = (i * num_tags) + j;
               double count2 = Tagger.ngram.getCount2At(k);
               N += count2;
               if(count2 > 0.0)
                 T++;
               else
                 Z++;
            } // for
            if(N > 0.0)
              d = N / (N + T);
            else
              d = 0.0;

            a = 0.0;
            b = 0.0;
            for(int j = 0; j < num_tags; j++)
            {
               k = (i * num_tags) + j;

               // Use the discounted probability, if possible, otherwise use
               // the backoff probability. This will be adjusted afterwards to
               // get a probability

               double count2 = Tagger.ngram.getCount2At(k);
               if(count2 > 0.0)
               {
                  p2[k] = d * (count2 / Tagger.ngram.getCount1At(i));
                  a += p2[k];
               } // fi

               else
               {
                  p2[k] = p1[j];
                  b += p2[k];
               } // else
            } // for

            a = (1.0 - a) / b;
            for(int j = 0; j < num_tags; j++)
            {
               k = (i * num_tags) + j;
               if(Tagger.ngram.getCount2At(k) == 0.0)
                 p2[k] *= a;
            } // for
        } // for 2-grams

        m = 0.0;
        for(int i = 0; i < num_states; i++)
        {
            p = p1[i];
            a = Math.abs(p - prState[i]);
            if(a > m)
              m = a;

            prState[i] = p;

            for(int j = 0; j < num_tags; j++)
            {
                p = p2[(i * num_tags) + j];
                a = Math.abs(p - prTrans[i][j]);
                if(a > m)
                  m = a;

                prTrans[i][j] = p;
            } // for
        } // for
   } // smooth
} // class Predict
