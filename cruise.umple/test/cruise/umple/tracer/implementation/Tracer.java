/*PLEASE DO NOT EDIT THIS CODE*/
/*This code was generated using the UMPLE @UMPLE_VERSION@ modeling language!*/



public class Tracer
{

  //------------------------
  // MEMBER VARIABLES
  //------------------------

  //Tracer Attributes
  private int id;
  private String name;

  //------------------------
  // CONSTRUCTOR
  //------------------------

  public Tracer(int aId, String aName)
  {
    id = aId;
    name = aName;
  }

  //------------------------
  // INTERFACE
  //------------------------

  public boolean setId(int aId)
  {
    boolean wasSet = false;
    id = aId;
    wasSet = true;
    if( id <= 6 )
    {
      System.out.println("id=" + aId);
    }
    return wasSet;
  }

  public boolean setName(String aName)
  {
    boolean wasSet = false;
    name = aName;
    wasSet = true;
    if( name != "tim" )
    {
      System.out.println("name=" + aName);
    }
    return wasSet;
  }

  public int getId()
  {
    return id;
  }

  public String getName()
  {
    return name;
  }

  public void delete()
  {}

}