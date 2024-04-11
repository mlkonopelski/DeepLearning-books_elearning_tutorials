'''
Singletons are used if we only want to instantiate the class once and 
errors if somewhere else the code tries to instantiate it again.

Use cases: 1) e.g. having multiple scripts but have one logger. 
           2) Connection Pool to databases. You want to have as little active connections as possible and resuse old active

'''
from abc import ABCMeta, abstractmethod

class IPerson(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def print_data():
        '''impelment in child class'''


class PersonSingleton(IPerson):
    __instance = None

    @staticmethod
    def get_instance():
        if PersonSingleton.__instance == None:
            PersonSingleton('Default Name', 0)
        return PersonSingleton.__instance
        
    def __init__(self, name, age) -> None:
        if PersonSingleton.__instance != None:
            raise Exception('Singleton cannot be instantiated more than 1')
        self.name = name 
        self.age = age
        PersonSingleton.__instance = self

    @staticmethod
    def print_data():
        print(f'Name: {PersonSingleton.__instance.name} Age: {PersonSingleton.__instance.age}')

p1 = PersonSingleton('Ted', 20)
print(p1)
p1.print_data()

p2 = PersonSingleton.get_instance()
print(p2)
p2.print_data()


# This should error because PersonSIngelton is already instantiated
p2 = PersonSingleton('Al', 15)
