from functools import partial
import numpy as np
from uncertainty_testbed.generators.data_generator_explicit import AnalyticBinaryClassGenerator
from uncertainty_testbed.utilities.functions import map_to_constant


if __name__ == "__main__":
    # generate some data
    s = partial(map_to_constant, c=0.)
    eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=10, name="chisq", threshold=0.5,
                                      noise_distribution="chisq", noise_scale=s)
    x_all, y_all = eg.generate_labeled(130000)
    x = x_all[np.min(x_all, axis=1) > 0, :]
    y = y_all[np.min(x_all, axis=1) > 0]

    t0, y0 = 0, 0
    for ti, yi in zip(np.sum(np.square(x), axis=1), y):
        if abs(t0 - eg.threshold) > abs(ti - eg.threshold):
            t0, y0 = ti, yi

    print(eg.threshold)
    print(t0, y0)
    print(np.sum(np.square(x), axis=1))
    print(y)
