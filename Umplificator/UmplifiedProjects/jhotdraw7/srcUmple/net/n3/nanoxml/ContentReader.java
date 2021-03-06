/*PLEASE DO NOT EDIT THIS CODE*/
/*This code was generated using the UMPLE 1.15.0.963 modeling language!*/

package net.n3.nanoxml;
import java.io.Reader;
import java.io.IOException;

public class ContentReader
{

  //------------------------
  // MEMBER VARIABLES
  //------------------------

  //------------------------
  // CONSTRUCTOR
  //------------------------

  public ContentReader()
  {}

  //------------------------
  // INTERFACE
  //------------------------

  public void delete()
  {}
  
  //------------------------
  // DEVELOPER CODE - PROVIDED AS-IS
  //------------------------
  
  private IXMLReader reader;
/**
    * Buffer.
    */
   private String buffer;
/**
    * Pointer into the buffer.
    */
   private int bufferIndex;
/**
    * The entity resolver.
    */
   private IXMLEntityResolver resolver;
/**
    * Cleans up the object when it's destroyed.
    */
   protected void finalize()
      throws Throwable
   {
      this.reader = null;
      this.resolver = null;
      this.buffer = null;
      super.finalize();
   }
/**
    * Reads a block of data.
    *
    * @param outputBuffer where to put the read data
    * @param offset first position in buffer to put the data
    * @param size maximum number of chars to read
    *
    * @return the number of chars read, or -1 if at EOF
    *
    * @throws java.io.IOException
    *		if an error occurred reading the data
    */
   public int read(char[] outputBuffer,
                   int    offset,
                   int    size)
      throws IOException
   {
      try {
         int charsRead = 0;
         int bufferLength = this.buffer.length();

         if ((offset + size) > outputBuffer.length) {
            size = outputBuffer.length - offset;
         }

         while (charsRead < size) {
            String str = "";
            char ch;

            if (this.bufferIndex >= bufferLength) {
               str = XMLUtil.read(this.reader, '&');
               ch = str.charAt(0);
            } else {
               ch = this.buffer.charAt(this.bufferIndex);
               this.bufferIndex++;
               outputBuffer[charsRead] = ch;
               charsRead++;
               continue; // don't interprete chars in the buffer
            }

            if (ch == '<') {
               this.reader.unread(ch);
               break;
            }

            if ((ch == '&') && (str.length() > 1)) {
               if (str.charAt(1) == '#') {
                  ch = XMLUtil.processCharLiteral(str);
               } else {
                  XMLUtil.processEntity(str, this.reader, this.resolver);
                  continue;
               }
            }

            outputBuffer[charsRead] = ch;
            charsRead++;
         }

         if (charsRead == 0) {
            charsRead = -1;
         }

         return charsRead;
      } catch (XMLParseException e) {
         throw new IOException(e.getMessage());
      }
   }
/**
    * Skips remaining data and closes the stream.
    *
    * @throws java.io.IOException
    *		if an error occurred reading the data
    */
   public void close()
      throws IOException
   {
      try {
         int bufferLength = this.buffer.length();

         for (;;) {
            String str = "";
            char ch;

            if (this.bufferIndex >= bufferLength) {
               str = XMLUtil.read(this.reader, '&');
               ch = str.charAt(0);
            } else {
               ch = this.buffer.charAt(this.bufferIndex);
               this.bufferIndex++;
               continue; // don't interprete chars in the buffer
            }

            if (ch == '<') {
               this.reader.unread(ch);
               break;
            }

            if ((ch == '&') && (str.length() > 1)) {
               if (str.charAt(1) != '#') {
                  XMLUtil.processEntity(str, this.reader, this.resolver);
               }
            }
         }
      } catch (XMLParseException e) {
         throw new IOException(e.getMessage());
      }
   }
}