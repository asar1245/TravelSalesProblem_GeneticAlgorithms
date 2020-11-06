# Genetic algorithm development

# imports
import matplotlib.pyplot as plt
import sys
import array
import random
import numpy as np
from Transformation import geoVals
import pandas as pd
# import all the DEAP parts
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# get cities dataFrame, (x,y,z), with geo centric coordinates
pdCities = geoVals()
citiesMatrix = pdCities.to_numpy()
x = citiesMatrix[0]
y = citiesMatrix[1]
z = citiesMatrix[2]
numCities = 9
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

# toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.indices, 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def evalTSP(individual):
    # for diff distance till we return to same point
    # therefore, same beggining and end element
    print('individual is ', individual)
    xAxis = np.concatenate((x[individual], (x[individual])[0:1]))
    yAxis = np.concatenate((y[individual], (y[individual])[0:1]))
    zAxis = np.concatenate((z[individual], (z[individual])[0:1]))
    diffx = np.diff(xAxis)
    diffy = np.diff(yAxis)
    diffz = np.diff(zAxis)
    sumOfSquares = diffx**2 + diffy**2 + diffz**2
    sqrtOfSquares = np.sqrt(sumOfSquares)
    sumDistance = np.sum(sqrtOfSquares)
    # return tuple, distance and empty
    return sumDistance,


# register evaluate personalized function
toolbox.register("evaluate", evalTSP)


def main():  # start with a population of 300 individuals
    pop = toolbox.population(n=300)
    # only save the very best one
    hof = tools.HallOfFame(1)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 2, halloffame=hof)
    # plot the best one
    ind = hof[0]
    plt.figure(2)
    plt.plot(x[ind], y[ind])
    hof = tools.HallOfFame(2)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 50, halloffame=hof)
    # plot the best one
    ind1 = hof[0]
    ind2 = hof[1]
    print('best fitness individual1', ind1)
    print('best fitness individual2', ind2)
    print('best fitness distance 1: ', evalTSP(ind1))
    print('best fitness distance 2: ', evalTSP(ind2))
    plt.figure(2)
    plt.plot(x[ind], y[ind])
    plt.ion()
    plt.show()
    plt.pause(0.001)
    return pop, hof


if __name__ == "__main__":
    main()
