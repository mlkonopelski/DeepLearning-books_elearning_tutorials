'''
Getter and Setter method to read and write to private attributes
'''

class Person:

    def __init__(self, name, age, gender) -> None:
        self.__name = name
        self.__age = age
        self.__gender = gender
    
    @property
    def Name(self):
        return self.__name
    
    @Name.setter
    def Name(self, value):
        self.__name = value

p1 = Person('Peter', 16, 'M')

print(p1.Name)
p1.__name = 'Wojtek'
print(p1.Name)
p1.Name = 'Karol'
print(p1.Name)
