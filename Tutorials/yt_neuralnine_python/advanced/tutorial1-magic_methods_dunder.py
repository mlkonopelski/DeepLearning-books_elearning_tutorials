"""
Python Magic methods are the methods starting and ending with double underscores ‘__’. 
They are defined by built-in classes in Python and commonly used for operator overloading. 
"""


# Example of magic method __del__
class Person:

    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age

    def __del__(self):
        print('Object Person is deconstructed')


p = Person('Mike', 25)
print('Object Person is created')

# Example of overloading magic method
class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """
        METHOD OVERLOADING!
        """
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other.x, self.y * other.y)

    def __str__(self) -> str:
        return f'x: {self.x}, y: {self.y}'

    def __repr__(self) -> str:
        ... # same as above

    def __len__(self):
        return 10 # just example value
    
    def __call__(self, value):
        print(f'Nmber: {value}')


v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v1)
print(v1 - v1)
print(v1 * v1)
print(len(v1))
v1(5)