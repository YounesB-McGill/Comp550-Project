/*PLEASE DO NOT EDIT THIS CODE*/
/*This code was generated using the UMPLE 1.15.0.963 modeling language!*/

package cruise.umple.compiler;

public class MethodParameter extends UmpleVariable
{

  //------------------------
  // MEMBER VARIABLES
  //------------------------

  //MethodParameter Attributes
  private boolean isAutounique;

  /**
   * TODO: should default to false, but constructors would need updating
   */
  private boolean isList;
  private boolean isDerived;
  private boolean isLazy;

  //------------------------
  // CONSTRUCTOR
  //------------------------

  public MethodParameter(String aName, String aType, String aModifier, String aValue, boolean aIsAutounique)
  {
    super(aName, aType, aModifier, aValue);
    isAutounique = aIsAutounique;
    isList = false;
    isDerived = false;
    isLazy = false;
  }

  //------------------------
  // INTERFACE
  //------------------------

  public boolean setIsAutounique(boolean aIsAutounique)
  {
    boolean wasSet = false;
    isAutounique = aIsAutounique;
    wasSet = true;
    return wasSet;
  }

  public boolean setIsList(boolean aIsList)
  {
    boolean wasSet = false;
    isList = aIsList;
    wasSet = true;
    return wasSet;
  }

  public boolean setIsDerived(boolean aIsDerived)
  {
    boolean wasSet = false;
    isDerived = aIsDerived;
    wasSet = true;
    return wasSet;
  }

  public boolean setIsLazy(boolean aIsLazy)
  {
    boolean wasSet = false;
    isLazy = aIsLazy;
    wasSet = true;
    return wasSet;
  }

  public boolean getIsAutounique()
  {
    return isAutounique;
  }

  public boolean getIsList()
  {
    return isList;
  }

  public boolean getIsDerived()
  {
    return isDerived;
  }

  public boolean getIsLazy()
  {
    return isLazy;
  }

  public boolean isIsAutounique()
  {
    return isAutounique;
  }

  public boolean isIsList()
  {
    return isList;
  }

  public boolean isIsDerived()
  {
    return isDerived;
  }

  public boolean isIsLazy()
  {
    return isLazy;
  }

  public void delete()
  {
    super.delete();
  }

}