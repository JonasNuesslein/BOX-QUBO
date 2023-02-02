import numpy as np
from dwave_qbsolv import QBSolv
import pickle



class MAXCLIQUE:

    def __init__(self, V, E):
        self.graph = np.zeros((V, V))
        self.V = V
        self.E = E
        edges = [(i,j) for i in range(V) for j in range(V) if i < j]
        selected_edges = np.random.choice(len(edges), size=E, replace=False)
        for e in selected_edges:
            i,j = edges[e]
            self.graph[i][j] = 1
            self.graph[j][i] = 1

    def oracle(self, x):
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j and x[i] == 1 and x[j] == 1 and self.graph[i][j] == 0:
                    return 1   # x is an invalide state
        return -np.sum(x)

    def get_optimum(self):
        Q = {}
        Q_matrix = np.zeros((self.V, self.V))
        for i in range(self.V):
            for j in range(self.V):
                if i == j:
                    Q[(i,i)] = -1
                    Q_matrix[i][i] = -1
                elif i < j and self.graph[i][j] == 0:
                    Q[(i,j)] = 3
                    Q_matrix[i][j] = 3
        response = QBSolv().sample_qubo(Q, num_repeats=100)
        r = response.samples()[0]
        x = np.zeros(self.V)
        for i in range(self.V):
            x[i] = r[i]
        return np.dot(np.dot(Q_matrix, x), x)

    def save(self, name):
        pickle.dump((self.graph, self.V, self.E), open(name, "wb"))

    def load(self, name):
        self.graph, self.V, self.E = pickle.load(open(name, "rb"))


