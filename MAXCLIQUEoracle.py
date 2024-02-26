import numpy as np
from dwave_qbsolv import QBSolv


graph = None
V = -1


def create_instance(V_, E):
    global graph
    global V
    V = V_
    new_graph = np.zeros((V,V))
    edges = [(i,j) for i in range(V) for j in range(V) if i < j]
    selected_edges = np.random.choice(len(edges), size=E, replace=False)
    for e in selected_edges:
        i,j = edges[e]
        new_graph[i][j] = 1
        new_graph[j][i] = 1
    graph = new_graph



def oracle(x):
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and x[i] == 1 and x[j] == 1 and graph[i][j] == 0:
                return 1   # x is an invalide state
    return -np.sum(x)



def get_optimum():
    Q = {}
    Q_matrix = np.zeros((V,V))
    for i in range(V):
        for j in range(V):
            if i == j:
                Q[(i,i)] = -1
                Q_matrix[i][i] = -1
            elif i < j and graph[i][j] == 0:
                Q[(i,j)] = 3
                Q_matrix[i][j] = 3
    response = QBSolv().sample_qubo(Q, num_repeats=100)
    r = response.samples()[0]
    x = np.zeros(V)
    for i in range(V):
        x[i] = r[i]
    return np.dot(np.dot(Q_matrix, x), x)


