import sys 
import getopt

# fname = sys.argv[1]
# msg = sys.argv[2]


opts, args = getopt.getopt(sys.argv[1:], 'f:m:', ['fname', 'msg'])


# python3 tutorial4-args_parcing.py log.file hellow\ world
print(args)
print(opts)
# ['log.file', 'hellow world']
# []

# python3 tutorial4-args_parcing.py -f log.file -m hellow\ world
print(args)
print(opts)
# []
# [('-f', 'log.file'), ('-m', 'hellow world')]
