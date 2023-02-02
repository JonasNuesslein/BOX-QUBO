import numpy as np
import BOX_QUBO
import MAXCLIQUE


V, E, init_trainings_size, training_length = 25, 250, 4000, 4

mc = MAXCLIQUE.MAXCLIQUE(V, E)

init_X = np.random.choice(2, size=(init_trainings_size, V), p=[0.8, 0.2]).tolist()
init_Y = np.zeros(init_trainings_size).tolist()
for s in range(init_trainings_size):
    init_Y[s] = mc.oracle(init_X[s])

print("Optimum: ", mc.get_optimum())
print("Min_init: ", np.min(init_Y))

Q, bias, chart = BOX_QUBO.optimize(mc.oracle, init_X, init_Y, ratio=0.03, training_length=training_length)
print("BOX_QUBO(3%): ", chart)

