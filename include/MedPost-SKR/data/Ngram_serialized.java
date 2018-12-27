package gov.nih.nlm.nls.tagger;

import gov.nih.nlm.nls.tagger.Tags;

import java.io.File;
import java.io.BufferedReader;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.IOException;

public class Ngram
{
    private static final String ngram_file = 
      System.getProperty("ngram_file", 
     "/nfsvol/nls/MEDLINE_Baseline_Repository/Larrys_Tagger/models/ngrams.cur");

    private static final String count1_file = 
      System.getProperty("count1_file", 
     "/nfsvol/nls/MEDLINE_Baseline_Repository/Larrys_Tagger/models/ngramCount1.serial");

    private static final String count2_file = 
      System.getProperty("count2_file", 
     "/nfsvol/nls/MEDLINE_Baseline_Repository/Larrys_Tagger/models/ngramCount2.serial");

    private static final String prTag_file = 
      System.getProperty("prTag_file", 
     "/nfsvol/nls/MEDLINE_Baseline_Repository/Larrys_Tagger/models/ngramPrTag.serial");

    private static int num_tags = Tags.num_tags;
    private static final double count0 = 101216;

    private double count1[] = null;
    private double count2[] = null;
    private double pr_tag[] = null;

    // PUBLIC METHODS  ------------------------------------------------

    // Our default constructor does nothing
    public Ngram()
    {
        scangram();
    } // Ngram


    /** 
     * Get pr_tag array.
     *
     * @return double[] containing pr_tag.
     */
    public double[] getPrTag()
    {
      return(pr_tag);
    } // getPrTag


    /** 
     * Get Ngram count0.
     *
     * @return double containing count0.
     */
    public double getCount0()
    {
      return(count0);
    } // getCount0


    /** 
     * Get Ngram count1 at position pos.
     *
     * @param int pos Position within count1 required.
     * @return double containing count1[pos].
     */
    public double getCount1At(int pos)
    {
      return(count1[pos]);
    } // getCount1At


    /** 
     * Get Ngram count2 at position pos.
     *
     * @param int pos Position within count2 required.
     * @return double containing count2[pos].
     */
    public double getCount2At(int pos)
    {
      return(count2[pos]);
    } // getCount2At


    // PRIVATE METHODS  ------------------------------------------------

    private void scangram()
    {
        try
        {
           ObjectInputStream ois = 
                new ObjectInputStream(new FileInputStream(count1_file));
           count1 = (double [])ois.readObject();
           ois.close();

           ObjectInputStream ois2 = 
                new ObjectInputStream(new FileInputStream(count2_file));
           count2 = (double [])ois2.readObject();
           ois2.close();

           ObjectInputStream ois3 = 
                new ObjectInputStream(new FileInputStream(prTag_file));
           pr_tag = (double [])ois3.readObject();
           ois3.close();

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
