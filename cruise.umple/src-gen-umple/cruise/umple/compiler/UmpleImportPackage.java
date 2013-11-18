/*PLEASE DO NOT EDIT THIS CODE*/
/*This code was generated using the UMPLE 1.18.0.3036 modeling language!*/

package cruise.umple.compiler;

/**
 * @umplesource UmpleImport.ump 49
 * @umplesource UmpleImport_CodeModels.ump 38
 */
// line 49 "../../../../src/UmpleImport.ump"
// line 38 "../../../../src/UmpleImport_CodeModels.ump"
public class UmpleImportPackage extends UmpleImportElement
{
  @java.lang.annotation.Retention(java.lang.annotation.RetentionPolicy.RUNTIME)
  public @interface umplesourcefile{int[] line();String[] file();int[] javaline();int[] length();}

  //------------------------
  // MEMBER VARIABLES
  //------------------------

  //------------------------
  // CONSTRUCTOR
  //------------------------

  public UmpleImportPackage(String aId, String aName)
  {
    super(aId, aName);
  }

  //------------------------
  // INTERFACE
  //------------------------

  public void delete()
  {
    super.delete();
  }

  @umplesourcefile(line={42},file={"UmpleImport_CodeModels.ump"},javaline={41},length={5})
   public String generateUmple(){
    StringBuilder umpleBuilder = this.getUmpleBuilder();
		umpleBuilder.append("namespace " + this.getName() + ";");
		return umpleBuilder.toString();
  }

}