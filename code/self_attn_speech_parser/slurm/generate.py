import sys

with open(sys.argv[1]) as a, open(sys.argv[2], 'w') as b:

    b.write(a.readline().strip())

    for line in a:
        b.write(' ')
        b.write(line.strip())

    b.write('\n')
