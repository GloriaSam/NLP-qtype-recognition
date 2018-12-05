/*
===========================================================================
*
*                            PUBLIC DOMAIN NOTICE                          
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

import java.net.*;
import java.io.*;

import gov.nih.nlm.nls.mps.*;

/**
 * Multithreaded tagger server.
 *
 * @author	Jim Mork
 * @version	3.0, May 3, 2006
 * @since	1.0
 *
 * History:
 * version	3.0, May 3, 2006
 * changes:
 *       - Added support for MedPost and Penn Treebank tagged output
 *
 * version	2.0, June 9, 2005
 * changes:
 *       - Added support for prologfull and upcaret output formats
 *
 * version	1.0, August 18, 2004
**/

public class taggerServer
{      
   /** server port, defaults to 1795, use system property: taggerserver.port to change. */
   int serverPort =
     Integer.parseInt(System.getProperty("taggerserver.port", "1795"));
   ServerSocket aServerSocket;
   Tagger tagger = new Tagger();

   public taggerServer() 
   {
      // Create the server socket.

      try{
         aServerSocket = new ServerSocket(serverPort);
      } catch(IOException ioe) {
         System.out.println("Could not create server socket at " + 
                            serverPort + ".");
         System.exit(-1);
      } // try ... catch
      
      // Successfully created Server Socket. Now wait for connections.

      while(true)
      {                  
         try {
            // Accept incoming connections - will block until a client
            // connects to the server.

            Socket aClientSocket = aServerSocket.accept();
         
            // For each client, we will start a service thread to
            // service the client requests. Starting a thread lets the
            // taggerServer accept multiple connections simultaneously.
            
            ClientTaggerThread aClientThread =
                      new ClientTaggerThread(aClientSocket);

            aClientThread.start();
         } catch(IOException ioe) {
            System.out.println("Exception on accept, ignoring. Trace:");
            ioe.printStackTrace();
         } // try ... catch
      } // while true
   } // taggerServer

   
   public static void main (String[] args)
   {
      new taggerServer();   
   }

   /**
    * Client thread to run tagger.
    *
    * @author	Jim Mork
    * @version	1.0, August 18, 2004
    * @since	1.0
   **/

   class ClientTaggerThread extends Thread
   {
      Socket localClientSocket;
      
      ClientTaggerThread(Socket s)
      {
         localClientSocket = s;
      } // ClientTaggerThread
      
      public void run()
      {         
         // Obtain the input stream and the output stream for the socket
 
         BufferedReader in = null;
         PrintWriter out = null;

         try {                        
            in = new BufferedReader(new InputStreamReader(localClientSocket.getInputStream()));
            out = new PrintWriter(new OutputStreamWriter(localClientSocket.getOutputStream()));

            // Recognized Input:
            //    Line0    : May be proceeded by "syn|prolog" or
            //               "syn|prologfull" or "syn|upcaret" line which is 
            //               used to figure out which output format to use.
            //               NEW: You may also now specify "|penn", "|medpost",
            //                    or "|specialist" to get either Penn Treebank,
            //                    MedPost, or Specialist tagging respectively.
            //               The default assumed to be "syn|prolog|specialist".
            //    Line1 - N: Text to be tagged
            //    LineN+1  : <cr>
            //    LineN+2  : <cr>
            //    LineN+3  : ^THE_END^<cr>
            //    ....
            // Above repeats until we find "EOF<cr>" on a line by itself
            //    LineEOF  : EOF<cr>
            // Marks end of text to be processed.  Close down connection

            // Output Format (prolog and prologfull):
            //    Line1    : [<cr>
            //    Line2 - N: Normal tagged text
            //    LineN+1  : ].<cr>
            //    LineN+2  : ^THE_END^<cr>
            //    LineN+3  : %%<cr>

            // Output Format (upcaret):
            //    Line1 - N: "word^tag" tagged text format
            //    LineN+2  : ^THE_END^<cr>
            //    LineN+3  : %%<cr>

            boolean EOF_found = false;
            boolean doPrologFull = false;
            boolean doPrologOutput = true;
            boolean tagPennTreebank = false;
            boolean tagMedPost = false;
            boolean tagSpecialist = true;
            String line = "";

            // Loop until we see the "EOF<cr>".  We want to pick up
            // each item separated by the "^THE_END^" markers to be tagged.
            // Items can be any text.

            while(!EOF_found)
            {
               boolean done = false;

               line = "";
               if((line = in.readLine()) == null)
                 EOF_found = true;

               if(!EOF_found && line.startsWith("EOF"))
               {
                  done = true;
                  EOF_found = true;
               } // fi

               // Find "^THE_END^" delimited text to be tagged.

               if(!done && !EOF_found)
               {
                  StringBuffer query = new StringBuffer();
                  while(!done)
                  {
                     if(line.startsWith("^THE_END^"))
                       done = true;

                     else if(line.startsWith("syn|"))
                     {
                        if(line.startsWith("syn|prologfull"))
                        {
                           doPrologOutput = true;
                           doPrologFull = true;
                        } // fi

                        else if(line.startsWith("syn|prolog"))
                        {
                           doPrologOutput = true;
                           doPrologFull = false;
                        } // else fi

                        else if(line.startsWith("syn|upcaret"))
                        {
                           doPrologOutput = false;
                           doPrologFull = false;
                        } // else fi

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
                     } // else fi

                     else
                       query.append(line).append(" ");

                     line = "";
                     if(!done)
                       if((line = in.readLine()) == null)
                         EOF_found = true;
                  } // while

                  // Call the Synchronized tagText method so we ensure the
                  // various threads don't step on each other's toes.

                  String result = ""; 
                  result = tagger.tagTextSynchronized(query.toString(),
                        doPrologOutput, doPrologFull, tagPennTreebank,
                        tagMedPost, tagSpecialist);

                  // Send back this result

                  out.println(result);
                  out.println("%%");
                  out.flush();
               } // fi !done ...
            } // while !EOF_found

            // Cleanup

            in.close();
            out.close();
            localClientSocket.close();
         } catch(Exception e) { 
              e.printStackTrace();
         } // try ... catch
      } // run
   } // class ClientTaggerThread
} // taggerServer

