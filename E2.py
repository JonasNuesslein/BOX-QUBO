import numpy as np
import BOX_QUBO
import MAXCLIQUEoracle
#import FMQA


V, E, init_trainings_size, training_length = 30, 350, 10000, 10
oracle = MAXCLIQUEoracle.oracle


for i in range(15):

    MAXCLIQUEoracle.create_instance(V, E)
    init_X = np.random.choice(2, size=(init_trainings_size, V), p=[0.8, 0.2]).tolist()
    init_Y = np.zeros(init_trainings_size).tolist()
    for ii in range(init_trainings_size):
        init_Y[ii] = oracle(init_X[ii])
    MAXCLIQUEoracle.V = V
    print("Optimum: ", MAXCLIQUEoracle.get_optimum())
    print("Min_init: ", np.min(init_Y))

    Q, bias, chart = BOX_QUBO.optimize(oracle, init_X, init_Y, ratio=0.1, training_length=training_length, h=[20,200,10,100])
    print("BOX_QUBO(10%): ", chart)

    Q, chart = FMQA.optimize(oracle, init_X, init_Y, training_length=training_length)
    print("FMQA: ", chart)

    print()

