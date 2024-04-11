'''
One 1 class is composed of multiple other instances of classes.
E.g. method 'add_member' adds a new Person to class School 

'''

from abc import ABCMeta, abstractmethod

class IDepartment(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, employees) -> None:
        '''implement is child class'''

    @staticmethod
    @abstractmethod
    def print_department():
        '''implement in child class'''


class Accounting(IDepartment):

    def __init__(self, employees) -> None:
        self.employees = employees

    def print_department(self):
        print(f'Acconting department: {self.employees}')


class Development(IDepartment):

    def __init__(self, employees) -> None:
        self.employees = employees

    def print_department(self):
        print(f'Development department: {self.employees}')


class ParentDepartment(IDepartment):
    def __init__(self, employees) -> None:
        self.employees = employees
        self.base_employees = employees
        self.sub_depts = []

    def add(self, dept):
        self.sub_depts.append(dept)
        self.employees += dept.employees

    def print_department(self):
        print(f'Parent Department Base Employees: {self.base_employees}')
        for dept in self.sub_depts:
            dept.print_department()

dept1 = Accounting(200)
dept2 = Development(300)
dept3 = ParentDepartment(10)
dept3.add(dept1)
dept3.add(dept2)
dept3.print_department()
    