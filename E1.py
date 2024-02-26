import numpy as np
import BOX_QUBO
import SAToracle
import FMQA
import Choi


V, K, k, init_trainings_size, training_length = 20, 300, 4, 10000, 30
oracle = SAToracle.oracle


for i in range(15):

    SAToracle.create_instance(V, K, k)
    init_X = np.random.choice(2, size=(init_trainings_size, V), p=[0.5, 0.5]).tolist()
    init_Y = np.zeros(init_trainings_size).tolist()
    for t in range(init_trainings_size):
        init_Y[t] = oracle(init_X[t])
    print("Optimum: ", SAToracle.get_optimum())
    print("Min_init: ", np.min(init_Y))

    Q, bias, chart = BOX_QUBO.optimize(oracle, init_X, init_Y, ratio=0.999, training_length=training_length, h=[5,1000,5,300])
    print("BOX_QUBO(100%): ", chart)

    Q, bias, chart = BOX_QUBO.optimize(oracle, init_X, init_Y, ratio=0.03, training_length=training_length, h=[5,1000,5,300])
    print("BOX_QUBO(3%): ", chart)

    X_star, Y_star = Choi.Choi(SAToracle.instance, V).sample_qbsolv(oracle)
    print("Choi: ", np.min(Y_star))

    Q, chart = FMQA.optimize(oracle, init_X, init_Y, training_length=training_length)
    print("FMQA: ", chart)

    print()

