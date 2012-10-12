# PLEASE DO NOT EDIT THIS CODE
# This code was generated using the UMPLE 1.15.0.963 modeling language!
# NOTE: Ruby generator is experimental and is missing some features available in
# in other Umple generated languages like Java or PHP

module CruiseAssociations

class StudentAO


  #------------------------
  # MEMBER VARIABLES
  #------------------------

  #StudentAO Attributes - for documentation purposes
  #attr_reader :number

  #StudentAO Associations - for documentation purposes
  #attr_reader :mentor

  #------------------------
  # CONSTRUCTOR
  #------------------------

  def initialize(a_number, a_mentor)
    @initialized = false
    @deleted = false
    @number = a_number
    @mentor = nil
    did_add_mentor = set_mentor(a_mentor)
    raise "Unable to create student due to @mentor" unless did_add_mentor
    @initialized = true
  end

  #------------------------
  # INTERFACE
  #------------------------

  def set_number(a_number)
    was_set = false
    raise "Mandatory relationship with mentor not satisfied" if (@initialized and !@deleted and @mentor.nil?)
    @number = a_number
    was_set = true
    was_set
  end

  def get_number
    raise "Mandatory relationship with mentor not satisfied" if (@initialized and !@deleted and @mentor.nil?)
    @number
  end

  def get_mentor
    raise "Mandatory relationship with mentor not satisfied" if (@initialized and !@deleted and @mentor.nil?)
    @mentor
  end

  def set_mentor(a_mentor)
    was_set = false
    raise "Mandatory relationship with mentor not satisfied" if (@initialized and !@deleted and @mentor.nil?)
    # Must provide @mentor to student
    if a_mentor.nil?
      return was_set
    end

    # @mentor already at maximum (2)
    if a_mentor.number_of_students >= MentorAO.maximum_number_of_students
      return was_set
    end
    
    existing_mentor = @mentor
    @mentor = a_mentor
    if !existing_mentor.nil? and !existing_mentor.eql?(a_mentor)
      didRemove = existing_mentor.remove_student(self)
      unless didRemove
        @mentor = existing_mentor
        return was_set
      end
    end
    @mentor.add_student(self)
    was_set = true
    was_set
  end

  def delete
    @deleted = true
    raise "Mandatory relationship with mentor not satisfied" if (@initialized and !@deleted and @mentor.nil?)
    @placeholder_mentor = @mentor
    @mentor = nil
    @placeholder_mentor.remove_student(self)
  end

end
end