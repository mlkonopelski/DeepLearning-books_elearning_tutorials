"""
We create a base class to dynmically create other classes e.g. based on argument passed. 
One example could be Train factory to create Train,Val,Test classes based on argument.
Each will have some comon methods to ovewrite and their own.
"""

from abc import ABCMeta, abstractmethod

class IPerson(metaclass=ABCMeta):
    '''
    meta=ABCMEtda means that this is a Abstract class and we cannot create objects from it
    '''
    @staticmethod 
    @abstractmethod
    def person_method():
        '''Interface method'''

# # this doesn't work: TypeError: Can't instantiate abstract class IPerson with abstract method person_method
# p1 = IPerson()
# p1.person_method()

# new class
class Student(IPerson):
    def __init__(self) -> None:
        self.name = 'Basic Student Name'
    
    def person_method(self):
        """
        This method needs to be overridden we use it or not because it's an abstract method. 
        """
        print('im student')

class Teacher(IPerson):
    def __init__(self) -> None:
        self.name = 'Basic Teacher name'

    def person_method(self):
        print('im teacher')

# s1 = Student()
# s1.person_method()
# t1 = Teacher()
# t1.person_method()

class PersonFactory:
    @staticmethod
    def build_person(person_type: str):
        if person_type == 'Student':
            return Student()
        elif person_type == 'Teacher':
            return Teacher()
        else:
            raise Exception('Invalid person type')
# s1 = PersonFactory.build_person('student')
# t1 = PersonFactory.build_person('teacher')

if __name__ == '__main__':

    choice = input('What type of person do you want?\n')
    p1 = PersonFactory.build_person(choice)
    p1.person_method()
