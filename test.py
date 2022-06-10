import numpy as np
import BOX_QUBO
import MAXCLIQUEoracle


V, E, init_trainings_size, training_length = 30, 350, 10000, 3
oracle = MAXCLIQUEoracle.oracle

MAXCLIQUEoracle.create_instance(V, E)
init_X = np.random.choice(2, size=(init_trainings_size, V), p=[0.8, 0.2]).tolist()
init_Y = np.zeros(init_trainings_size).tolist()
for i in range(init_trainings_size):
    init_Y[i] = oracle(init_X[i])

print("Optimum: ", MAXCLIQUEoracle.get_optimum())
print("Min_init: ", np.min(init_Y))

Q, bias, chart = BOX_QUBO.optimize(oracle, init_X, init_Y, ratio=0.03, training_length=training_length)

print()
print()
print("Best found solution: ", chart[-1])

