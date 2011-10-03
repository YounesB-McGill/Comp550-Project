/*

Copyright 2010 Andrew Forward, Timothy C. Lethbridge

This file is made available subject to the open source license found at:
http://cruise.site.uottawa.ca/UmpleMITLicense.html

*/

package cruise.umple.implementation.java;

import java.io.File;

import org.junit.*;

import cruise.umple.compiler.UmpleModel;
import cruise.umple.implementation.ExternalClassTemplateTest;

import cruise.umple.util.SampleFileWriter;

public class JavaExternalClassTemplateTest extends ExternalClassTemplateTest
{
  
  @Test
  public void DoNotCreateExternalClasses()
  {
    assertUmpleTemplateFor("ExternalClassTest.ump",languagePath + "/ExternalClassTest_MyClass."+ languagePath +".txt","MyException");
    Assert.assertEquals(false, (new File(pathToInput + "/example/Exception.java")).exists());
  }
  @Test
  public void GeneratePathTest()
  {
	  UmpleModel model = createUmpleSystem(pathToInput , languagePath + "/ClassTemplateTest_BuildOutputPath.ump");
	  model.generate();

	  String actual = SampleFileWriter.readContent(new File(pathToInput, languagePath + "/java_code/example/Student.java"));
	  System.out.print(actual);
	  
	  String expected = SampleFileWriter.readContent(new File(pathToInput, languagePath + "/ClassTemplateTest_BuildOutputPath.ump.txt"));
	  Assert.assertEquals(expected, actual);
  }
}