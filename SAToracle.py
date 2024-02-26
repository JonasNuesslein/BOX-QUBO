import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


instance = None


def create_instance(V, K, k):
    global instance
    F = []
    while len(F) < K:
        C = np.random.choice(range(1,V+1), size=k, replace=False)
        sign = np.random.choice([-1,+1], size=k)
        C *= sign
        C.sort()
        C = tuple(C)
        if C not in F:
            F.append(C)
    instance = F


def oracle(x):
    sat = 0
    for C in instance:
        for l in C:
            if (l < 0 and x[abs(l)-1] == 0) or (l > 0 and x[abs(l)-1] == 1):
                sat += 1
                break
    return -sat


def get_optimum():
    rc2 = RC2(WCNF())
    for C in instance:
        rc2.add_clause(list(C), weight=1)
    model = rc2.compute()
    solution = [1 if i > 0 else 0 for i in model]
    return oracle(solution)




