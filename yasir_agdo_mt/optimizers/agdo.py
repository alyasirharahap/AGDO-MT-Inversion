from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.objective import evaluate_mt_model
from .initialization import initialize_population
from .levy import levy_step


@dataclass(slots=True)
class AGDOConfig:
    """
    Configuration for the Adam Gradient Descent Optimizer.
    """

    npop: int = 120
    niter: int = 150

    learning_rate: float | Callable[[int, int], float] = 0.05

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    levy_beta: float = 1.5

    seed: int | None = None

    tolerance: float | None = None
    patience: int | None = None

    verbose: bool = False


@dataclass(slots=True)
class AGDOResult:
    """
    Result returned by the AGDO optimizer.
    """

    best_model: np.ndarray
    best_fitness: float

    convergence: np.ndarray
    population_history: np.ndarray

    iterations: int
    converged: bool


def agdo_mt(
    frequency: np.ndarray,
    rho_obs: np.ndarray,
    phase_obs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    config: AGDOConfig | None = None,
) -> AGDOResult:
    """
    One-dimensional MT inversion using the
    Adam Gradient Descent Optimizer (AGDO).
    """

    if config is None:
        config = AGDOConfig()

    frequency = np.asarray(frequency, dtype=float)
    rho_obs = np.asarray(rho_obs, dtype=float)
    phase_obs = np.asarray(phase_obs, dtype=float)

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if frequency.ndim != 1:
        raise ValueError("frequency must be one-dimensional.")

    if rho_obs.ndim != 1:
        raise ValueError("rho_obs must be one-dimensional.")

    if phase_obs.ndim != 1:
        raise ValueError("phase_obs must be one-dimensional.")

    if (
        frequency.size != rho_obs.size
        or frequency.size != phase_obs.size
    ):
        raise ValueError(
            "frequency, rho_obs and phase_obs must have identical lengths."
        )

    if lb.shape != ub.shape:
        raise ValueError(
            "Lower and upper bounds must have identical shapes."
        )

    if np.any(lb >= ub):
        raise ValueError(
            "Each lower bound must be smaller than its upper bound."
        )

    ndim = lb.size

    if ndim % 2 == 0:
        raise ValueError(
            "Model dimension must be odd (2 × nlayer − 1)."
        )

    rng = np.random.default_rng(config.seed)

    population = initialize_population(
        npop=config.npop,
        lb=lb,
        ub=ub,
        rng=rng,
    )

    m = np.zeros((config.npop, ndim))
    v = np.zeros((config.npop, ndim))

    fitness = np.zeros(config.npop)

    for i in range(config.npop):

        fitness[i] = evaluate_mt_model(
            model=population[i],
            frequency=frequency,
            rho_obs=rho_obs,
            phase_obs=phase_obs,
        )

    best_index = np.argmin(fitness)

    best_model = population[best_index].copy()
    best_fitness = fitness[best_index]

    convergence = np.zeros(config.niter)
    population_history = np.zeros(config.niter)

    converged = False
    no_improvement = 0

    for iteration in range(config.niter):

        t = iteration + 1

        if callable(config.learning_rate):
            lr = config.learning_rate(t, config.niter)
        else:
            lr = config.learning_rate

        for i in range(config.npop):

            omega = (
                rng.random()
                * (
                    (t / config.niter) ** 2
                    - 2 * (t / config.niter)
                    + 0.5
                )
            )

            alpha = np.cos(
                1 - rng.random() * 2 * np.pi
            )

            x_new = (
                omega * population[i]
                + alpha * population[i]
            )

            population_mean = np.mean(
                population,
                axis=0,
            )

            C = 1.0 - rng.random()

            P = population_mean - x_new

            grad = best_model - C * P

            m[i] = (
                config.beta1 * m[i]
                + (1.0 - config.beta1) * grad
            )

            v[i] = (
                config.beta2 * v[i]
                + (1.0 - config.beta2) * (grad ** 2)
            )

            m_hat = (
                m[i]
                / (1.0 - config.beta1 ** t)
            )

            v_hat = (
                v[i]
                / (1.0 - config.beta2 ** t)
            )

            x_new = (
                x_new
                - lr
                * m_hat
                / (
                    np.sqrt(v_hat)
                    + config.eps
                )
            )

            if rng.random() < 0.30:

                delta = (
                    rng.random()
                    * (
                        (t/config.niter) ** 2
                        -2 * (t/config.niter)
                        +1
                    )
                )

                theta = 2 * t / config.niter

                x_new = (
                    best_model
                    + levy_step(
                        ndim,
                        beta=config.levy_beta,
                        rng=rng,
                    )
                    * delta
                    * (
                        best_model
                        - population[i] * theta
                    )
                )

            x_new = np.clip(
                x_new,
                lb,
                ub,
            )

            new_fitness = evaluate_mt_model(
                model=x_new,
                frequency=frequency,
                rho_obs=rho_obs,
                phase_obs=phase_obs,
            )

            if new_fitness < fitness[i]:

                population[i] = x_new
                fitness[i] = new_fitness

                if new_fitness < best_fitness:

                    best_fitness = new_fitness
                    best_model = x_new.copy()

        # --------------------------------------------------
        # Convergence history
        # --------------------------------------------------

        convergence[iteration] = best_fitness
        population_history[iteration] = np.mean(fitness)

        # --------------------------------------------------
        # Progress display
        # --------------------------------------------------

        if config.verbose:

            if (
                (iteration + 1) % 10 == 0
                or iteration == 0
                or iteration == config.niter - 1
            ):

                print(
                    f"Iteration {iteration + 1:4d} | "
                    f"Best Fitness = {best_fitness:.6f}"
                )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if (
            config.tolerance is not None
            and config.patience is not None
        ):

            if iteration > 0:

                improvement = (
                    convergence[iteration - 1]
                    - convergence[iteration]
                )

                if improvement < config.tolerance:
                    no_improvement += 1
                else:
                    no_improvement = 0

                if no_improvement >= config.patience:

                    converged = True

                    convergence = convergence[: iteration + 1]
                    population_history = population_history[: iteration + 1]

                    break

    # ------------------------------------------------------
    # Return optimization result
    # ------------------------------------------------------

    iterations = iteration + 1

    return AGDOResult(
        best_model=best_model,
        best_fitness=best_fitness,
        convergence=convergence,
        population_history=population_history,
        iterations=iterations,
        converged=converged,
    )    