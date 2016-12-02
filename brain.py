import numpy as np
import scipy, collections

from itertools import permutations
from itertools import combinations

year = 2017
input = sorted([int(c) for c in str(year)])
largest = int(str(max(input))*len(input))
smallest = int(str(min(input))*len(input))

input_int = sorted([c for c in str(year)])

X = []
for i in range(len(input)):
    if i != 0:
        X += [''.join(p) for p in list(permutations(input_int, i+1))]
X = [int(e) for e in X if int(e) > 0]
for i in input:
    for j in range(len(input)+1):
        if i > 0 and j > 0:
            X.append(int(str(i) * j))

S = dict()
for i in X:
    for j in X:
        v = i*j
        k = str(i) + '^2 + ' + str(j) + '^2 = ' + str(v)
        raw = collections.Counter(k)
        if raw['7'] >= 8:
            S[k] = v



S = sorted(S)





X_num = sorted([int(e) for e in X])

