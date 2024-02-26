import numpy as np
import fmqa
import dimod



def optimize(oracle, init_X, init_Y, training_length):

    xs = np.array(init_X)
    ys = np.array(init_Y)

    model = fmqa.FMBQM.from_data(xs, ys)
    sa_sampler = dimod.samplers.SimulatedAnnealingSampler()
    chart = [np.min(ys)]

    for e in range(training_length):
        res = sa_sampler.sample(model, num_reads=50)
        xs = np.r_[xs, res.record['sample']]
        ys = np.r_[ys, [oracle(x) for x in res.record['sample']]]
        model.train(xs, ys)
        chart.append(np.min(ys))
        print("#### Epoche", e, "####")
        print("FMQA chart: ", chart)
        print("######################")

    return model.to_qubo(), chart



