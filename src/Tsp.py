# Genetic algorithm development

import matplotlib.pyplot as plt
import sys
import array
import random
import numpy as np
# import all the DEAP parts
from deap import algorithms
from deap import base

from deap import creator
from deap import tools
numCities = 10
random.seed(169)
x = np.random.rand(numCities)
print(x)
y = np.random.rand(numCities)
plt.plot(x, y)
plt.show()

# We want to minimize the distance so the weights have to be negative
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", array.array, typecode='i',
               fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(numCities), numCities)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.indices)

#toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.indices, 10)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def evalTSP(individual):
    diffx = np.diff(x[individual])
    diffy = np.diff(y[individual])
    distance = np.sum(diffx**2 + diffy**2)
    return distance,


toolbox.register("evaluate", evalTSP)


def main():  # start with a population of 300 individuals
    pop = toolbox.population(n=300)
    # only save the very best one
    hof = tools.HallOfFame(1)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 140, halloffame=hof)
    # plot the best one
    ind = hof[0]
    plt.figure(2)
    plt.plot(x[ind], y[ind])
    hof = tools.HallOfFame(1)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 140, halloffame=hof)
    # plot the best one
    ind = hof[0]
    plt.figure(2)
    plt.plot(x[ind], y[ind])
    plt.ion()
    plt.show()
    plt.pause(0.001)
    return pop, hof


if __name__ == "__main__":
    main()
