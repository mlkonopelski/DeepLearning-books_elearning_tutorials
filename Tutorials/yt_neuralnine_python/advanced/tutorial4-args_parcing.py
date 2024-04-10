import sys 
import getopt

# Easy way
# fname = sys.argv[1]
# msg = sys.argv[2]

# Proper way
opts, args = getopt.getopt(sys.argv[1:], 'f:m:', ['fname', 'msg'])


# python3 tutorial4-args_parcing.py log.file hellow\ world
print(args)
print(opts)
# ['log.file', 'hellow world']
# []

# python3 tutorial4-args_parcing.py -m hellow\ world -f log.file # changed order
print(args)
print(opts)
# []
# [('-f', 'log.file'), ('-m', 'hellow world')]
# access
for key, value in opts:
    if key == '-f':
        fname = value
    elif key == '-m':
        msg = value
    else:
        print(f'Unknown: {key}={value}')
