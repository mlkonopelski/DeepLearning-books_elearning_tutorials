import sys

# Seq 1, 9,000,0000
def my_generator(n):
    for x in range(n):
        yield x ** 3

values = my_generator(100)
for x in values:
    print(x)

print(sys.getsizeof(values))
values = my_generator(1000)
print(sys.getsizeof(values))

def infinite_sequence():
    result = 1
    while True:
        yield result
        result *= 5

values = infinite_sequence()
print(next(values))
print(next(values))
print(next(values))
print(next(values))
print(next(values))
