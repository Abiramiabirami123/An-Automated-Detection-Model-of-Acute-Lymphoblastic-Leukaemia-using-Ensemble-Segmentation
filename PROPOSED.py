import time
import numpy as np
import math
from numpy import mean


def PROPOSED(pop, costFunc, lb, ub, iterations):
    # costFunc: function to minimize
    # dim: number of dimensions
    # popSize: population size
    # iterations: maximum number of iterations
    # lb: lower bounds of the search space
    # ub: upper bounds of the search space

    # Initializing the population
    popSize, dim = pop.shape[0], pop.shape[1]
    pop = np.random.uniform(low=lb, high=ub, size=(popSize, dim))
    fit = np.zeros(popSize)
    bestFit = np.inf
    bestSol = np.zeros(dim)

    Convergence_curve = np.zeros(iterations)
    ct = time.time()
    # Main loop
    for it in range(iterations):
        for i in range(popSize):
            # Computing the fitness of the current solution
            fit[i] = costFunc(pop[i])

            # Updating the global best solution
            if fit[i] < bestFit:
                bestFit = fit[i]
                bestSol = pop[i].copy()

            # Generating the next candidate solution using ROA
            newSol = pop[i].copy()
            for j in range(dim):

                p = min(fit)/((max(fit) * mean(fit) * fit[i]))
                if p >= 0.5:

                    newSol[j] = pop[i][j] + math.exp(-it) * (bestSol[j] - pop[i][j]) + math.exp(
                        -it) * np.random.normal()
                else:
                    newSol[j] = pop[i][j] - math.exp(-it) * (bestSol[j] - pop[i][j]) + math.exp(
                        -it) * np.random.normal()

                # Applying boundary constraints
                if newSol[j] < lb:
                    newSol[j] = lb
                elif newSol[j] > ub:
                    newSol[j] = ub

            # Updating the population
            pop[i] = newSol

        Convergence_curve[it] = bestFit

    ct = time.time() - ct

    return bestSol, bestFit, Convergence_curve, ct