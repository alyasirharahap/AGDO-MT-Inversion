# AGDO for 1D Magnetotelluric Inversion

This repository contains an implementation of the Adam Gradient Descent Optimizer (AGDO) applied to the 1D Magnetotelluric (MT) inverse problem.

## Algorithm Description

AGDO is a hybrid optimization method that combines adaptive gradient-based updating with a simple population search mechanism. The update process is guided by the best solution in the population while still allowing limited exploration through random variation.

The algorithm is designed to remain simple while still being able to handle nonlinear inverse problems such as MT data inversion.

## Working Principle

The process begins by generating an initial set of candidate resistivity models. Each model is evaluated using a forward MT simulation to produce synthetic responses.

The difference between observed and modeled data is used to calculate the misfit. Based on this value, each candidate model is updated iteratively using information from the best solution, an adaptive momentum term, and a small random perturbation.

This cycle is repeated until the solution reaches convergence or the maximum number of iterations is achieved.

## Characteristics

The algorithm is relatively simple in structure and does not require extensive parameter tuning. It is designed to be computationally efficient and suitable for repeated inversion experiments.

The update strategy allows stable convergence behavior while maintaining enough flexibility to avoid premature stagnation.

## Summary

AGDO is a straightforward optimization approach for MT inversion that focuses on simplicity, stability, and computational efficiency.
