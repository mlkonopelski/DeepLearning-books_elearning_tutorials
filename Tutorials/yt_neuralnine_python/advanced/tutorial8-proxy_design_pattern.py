'''
Proxy method instantiate objects inside other class therefore we can 
have more control of how to run their methods etc.
Example: torch.layer = nn.Module() etc. nn.Conv2D
'''

from abc import ABCMeta, abstractmethod

class IPerson(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def person_method():
        '''Interface method'''

class Person(IPerson):
    def person_method(self):
        print('im person')

class ProxyPerson(IPerson):
    def __init__(self) -> None:
        self.person = Person()

    def person_method(self):
        print('im the proxy func')
        self.person.person_method()

pp1 = ProxyPerson()
pp1.person_method()