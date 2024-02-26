import numpy as np
import dimod
from dwave_qbsolv import QBSolv



class Choi:

    def __init__(self, formula, V):
        self.V = V
        self.seq = []
        for C in formula:
            self.seq.extend(list(C))
        self.n = len(self.seq)
        self.Q = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.Q[i][j] = -1
                elif i < j and abs(self.seq[i]) == abs(self.seq[j]) and np.sign(self.seq[i] * self.seq[j]) == -1:
                    self.Q[i][j] = 10
                elif i < j and i//len(formula[0]) == j//len(formula[0]):
                    self.Q[i][j] = 10

    def Q_to_dict(self):
        Q_dict = {}
        for i in range(self.n):
            for j in range(self.n):
                if self.Q[i][j] != 0 and i <= j:
                    Q_dict[(i,j)] = self.Q[i][j]
        return Q_dict

    def sample(self, oracle, n=3):
        Q_dict = self.Q_to_dict()
        # sample n solution vectors and predict y value
        sa_sampler = dimod.samplers.SimulatedAnnealingSampler()
        samples = sa_sampler.sample_qubo(Q_dict, num_reads=n)
        X = samples.record['sample'].tolist()
        X_star = [self.get_solution(x) for x in X]
        Y_star = [oracle(x) for x in X_star]
        return X_star, Y_star

    def sample_qbsolv(self, oracle):
        Q_dict = self.Q_to_dict()
        response = QBSolv().sample_qubo(Q_dict, num_repeats=1000)
        x = self.get_solution(response.samples()[0])
        y = oracle(x)
        return x, y

    def get_solution(self, xqubo):
        x = np.zeros(self.V)
        for i in range(self.n):
            if self.seq[i] > 0 and xqubo[i] == 1:
                x[abs(self.seq[i])-1] = 1
        return x


