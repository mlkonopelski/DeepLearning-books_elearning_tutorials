'''
MYPY is verification tool to check all declared types vs actual. 
It might be sueful for long scripts. 
'''

# python3 Tutorials/yt_neuralnine_python/advanced/tutorial6-type_hinting.py # OK
# mypy Tutorials/yt_neuralnine_python/advanced/tutorial6-type_hinting.py # NOK

def function(value: int) -> str:
    return str(value)

print(function(10))
# print(function('Hello World'))

def other_function(otherparameter: str) :
    print(otherparameter)

other_function(function(10))
