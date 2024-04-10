"""
Decrators
"""

def mydecorator(func):

    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        print('Im decorating your function')
        return value
    return wrapper

def hello_world():
    print('hello world')

# Straightforward but not pythonic
mydecorator(hello_world)()

#pythonic way
@mydecorator
def hello(person):
    print(f'hello {person}')
    return person

hello('Mike')

# practical case1
PATH = 'Tutorials/yt_neuralnine_python/advanced/'
def logged(func):
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        with open(PATH + 'log_file.txt', 'w') as f:
            fname = func.__name__
            f.write(f'{fname} returned value {value}')
        return value
    return wrapper

@logged
def add(x, y):
    return x + y

v = add(10, 20)

# practical case 2
import time

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        print(f'Execution time: {time.time() - start}')
    return wrapper

@timed
def add_func(x):
    r = 0
    for i in range(x):
        r += i
    return r

r1 = add_func(1)
r1 = add_func(1000)
r1 = add_func(1000000)
r1 = add_func(1000000000)