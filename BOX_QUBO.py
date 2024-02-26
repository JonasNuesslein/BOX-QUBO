import numpy as np
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import initializers
import dimod
from dwave_qbsolv import QBSolv



class BOX_QUBO:

    def __init__(self, init_X, init_Y, ratio):
        self.X = init_X
        self.Y = init_Y
        self.ratio = ratio
        self.n = len(init_X[0])
        self.n_params = (self.n+1)*self.n//2

        self.D = [[init_X[i], init_Y[i]] for i in range(len(init_X))]
        self.D.sort(key=lambda x:x[1])
        self.threshold = self.D[int(len(self.D)*self.ratio)][1]

        p_input = Input(1, name="p_input")
        mask_input = Input(self.n_params, name="mask_input")
        predicted_Q = Dense(self.n_params, activation='linear', use_bias=False, name="Q")(p_input)
        masked_Q = Multiply()([predicted_Q, mask_input])
        predicted_E = Dense(1, activation="linear", trainable=False,
                            kernel_initializer=initializers.Ones(),
                            bias_initializer=initializers.Zeros())(masked_Q)
        predicted_E_offset = Dense(1, activation="linear", name="offset")(predicted_E)
        self.model = Model(inputs=[p_input, mask_input], outputs=predicted_E_offset)
        opt = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(loss="mse", optimizer=opt)

    def get_flat_upper_triangular_matrix(self, X):
        flat_upper_X = np.zeros(self.n_params)
        j = 0
        for i in range(self.n):
            flat_upper_X[j:j+self.n-i] = X[i][i:]
            j += self.n-i
        return flat_upper_X

    def get_Q(self):
        Q_flat = self.model.get_layer('Q').get_weights()[0][0]
        offset = self.model.get_layer('offset').get_weights()
        scaling = offset[0][0][0]
        bias = offset[1][0]

        Q = np.zeros((self.n, self.n))
        j = 0
        for i in range(self.n):
            Q[i][i:] = Q_flat[j:j+self.n-i]
            j += self.n-i
        Q *= scaling

        return Q, bias

    def Q_to_dict(self, Q):
        Q_dict = {}
        for i in range(self.n):
            for j in range(self.n):
                if Q[i][j] != 0 and i <= j:
                    Q_dict[(i,j)] = Q[i][j]
        return Q_dict

    def train(self, n_cycles, n_epochs):

        p_inputs, mask_inputs = [], []
        for i in range(len(self.D)):
            p_inputs.append([1])
            mask_inputs.append(self.get_flat_upper_triangular_matrix(np.outer(self.D[i][0], self.D[i][0])))
        p_inputs, mask_inputs = np.array(p_inputs), np.array(mask_inputs)

        for c in range(n_cycles):
            print("Cycle: ", c, " / ", n_cycles)
            predicted_Y = self.model.predict({"p_input": p_inputs, "mask_input": mask_inputs})
            target_p_inputs = []
            target_mask_inputs = []
            target_Y = []
            for i in range(len(self.D)):
                if (i < int(len(self.D)*self.ratio) and self.D[i][1] < 1) or False:
                    target_p_inputs.append([1])
                    target_mask_inputs.append(mask_inputs[i])
                    target_Y.append([self.D[i][1]])
                elif predicted_Y[i][0] < self.threshold:
                    target_p_inputs.append([1])
                    target_mask_inputs.append(mask_inputs[i])
                    target_Y.append([self.threshold])
            target_p_inputs = np.array(target_p_inputs)
            target_mask_inputs = np.array(target_mask_inputs)
            target_Y = np.array(target_Y)

            self.model.fit({"p_input": target_p_inputs, "mask_input": target_mask_inputs}, target_Y, epochs=n_epochs,
                                    batch_size=1024, verbose=1, shuffle=True)

    def test(self):
        Q, bias = self.get_Q()
        for i in range(300): #int(len(self.D)*self.ratio)
            if i < int(len(self.D)*self.ratio) and self.D[i][1] < 1:
                print("*** ", end='')
            predicted_y = np.dot(np.dot(Q, self.D[i][0]), self.D[i][0]) + bias
            print(self.D[i][1], "   <=>   ", predicted_y)

    def sample(self, oracle, n=50):
        Q, bias = self.get_Q()
        Q_dict = self.Q_to_dict(Q)
        # sample n solution vectors and predict y value
        sa_sampler = dimod.samplers.SimulatedAnnealingSampler()
        samples = sa_sampler.sample_qubo(Q_dict, num_reads=n)
        X_star = samples.record['sample'].tolist()

        response = QBSolv().sample_qubo(Q_dict, num_repeats=1000)
        xqubo = [response.samples()[0][i] for i in range(self.n)]
        X_star.append(xqubo)

        Y_star = [oracle(x) for x in X_star]

        energies = [np.dot(np.dot(Q, x), x) + bias for x in X_star]
        print("pre: ", energies)
        print("Q: ", np.max(np.abs(Q)))
        print("bias: ", bias)

        return X_star, Y_star

    def add(self, X_star, Y_star):
        for i in range(len(X_star)):
            if X_star[i] not in self.X:
                self.X.append(X_star[i])
                self.Y.append(Y_star[i])
                self.D.append([X_star[i],Y_star[i]])
        self.D.sort(key=lambda x: x[1])
        self.threshold = self.D[int(len(self.D) * self.ratio)][1]



def optimize(oracle, init_X, init_Y, ratio, training_length, h):
    box_qubo = BOX_QUBO(init_X, init_Y, ratio)
    box_qubo.train(n_cycles=h[0], n_epochs=h[1])
    chart = [np.min(box_qubo.Y)]

    for e in range(training_length):

        print("#### Epoche", e, "####")
        box_qubo.test()
        print("BOX-QUBO chart: ", chart)

        X_star, Y_star = box_qubo.sample(oracle)
        print("Y_star: ", Y_star)
        box_qubo.add(X_star, Y_star)
        box_qubo.train(n_cycles=h[2], n_epochs=h[3])
        chart.append(np.min(box_qubo.Y))

        Q, bias = box_qubo.get_Q()
        energies = [np.dot(np.dot(Q, x), x) + bias for x in X_star]
        print("post: ", energies)
        print("threshold: ", box_qubo.threshold)

        print("######################")

    Q, bias = box_qubo.get_Q()
    return Q, bias, chart

