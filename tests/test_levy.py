import numpy as np

from yasir_agdo_mt.optimizers.levy import levy_step


def test_levy_shape():
    step = levy_step(5)

    assert step.shape == (5,)


def test_levy_is_finite():
    step = levy_step(20)

    assert np.all(np.isfinite(step))