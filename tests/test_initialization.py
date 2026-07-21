import numpy as np

from yasir_agdo_mt.optimizers.initialization import (
    initialize_population,
)


def test_population_shape():

    lb = np.array([1, 2, 3])

    ub = np.array([10, 20, 30])

    pop = initialize_population(
        npop=50,
        lb=lb,
        ub=ub,
    )

    assert pop.shape == (50, 3)


def test_population_bounds():

    lb = np.array([1, 2])

    ub = np.array([5, 10])

    pop = initialize_population(
        npop=100,
        lb=lb,
        ub=ub,
    )

    assert np.all(pop >= lb)

    assert np.all(pop <= ub)