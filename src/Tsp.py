# Genetic algorithm development

# Imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d axis
import random
import array
import numpy as np
from Transformation import geoVals
import pandas as pd

# Import all the DEAP parts
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Get cities dataFrame, (x,y,z), with geo centric coordinates
pdCities = geoVals()
citiesMatrix = pdCities.to_numpy()
citiesMatrix = citiesMatrix / 1000  # to set scale to km
x = citiesMatrix[0]
y = citiesMatrix[1]
z = citiesMatrix[2]

# number of cities
_, numCities = citiesMatrix.shape

# We want to minimize the distance between cities
#  so the weights have to be negative
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# We define the individual for the genetic algorithm
creator.create("Individual", array.array, typecode='i',
               fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Genetic algorithm attributes registration
toolbox.register("indices", random.sample, range(numCities), numCities)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# Function to make the first and last element the same
# in order to travel back to same city and calculate
# well distances and visualization
def closePath(individual, indX, indY, indZ):
    xAxis = np.concatenate((indX[individual], (indX[individual])[0:1]))
    yAxis = np.concatenate((indY[individual], (indY[individual])[0:1]))
    zAxis = np.concatenate((indZ[individual], (indZ[individual])[0:1]))
    return xAxis, yAxis, zAxis


# Function to calculate the distances between cities and
# the global path distance
def evalTSP(individual):
    # close path, to return to same city condition
    xAx, yAx, zAx = closePath(individual, x, y, z)
    diffx = np.diff(xAx)
    diffy = np.diff(yAx)
    diffz = np.diff(zAx)
    sumOfSquares = diffx**2 + diffy**2 + diffz**2
    sqrtOfSquares = np.sqrt(sumOfSquares)
    sumDistance = np.sum(sqrtOfSquares)
    # return tuple, distance and empty
    return sumDistance,


# Register the global path distance as the cos funtion of the
# genetic algortithm
toolbox.register("evaluate", evalTSP)


# main function
def main():  # start with a population of 300 individuals
    pop = toolbox.population(n=300)
    # only save the very best one
    hof = tools.HallOfFame(1)
    # use one of the built in GA's with a probablilty of mating of 0.75
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.75, 0.2, 140, halloffame=hof, )
    # plot the best one
    ind = hof[0]
    # make first and last element the same, to close the circle graphically
    xAx, yAx, zAx = closePath(ind, x, y, z)

    print('\nBest fitness individual:', ind)
    print('Best fitness distance: ',
          f'{round(evalTSP(ind)[0], 3)} kilometers\n')

    # GRAPHS
    # Figure of two plots
    # Cities names numpyArray in correct order
    citiesNames = pdCities.columns.values
    # Plot of cities position
    graph1 = plt.figure(1)
    # add a plot
    plot1Axes = graph1.add_subplot(121, projection='3d')
    # Name the plot
    plot1Axes.set_title('Cities Map',
                        fontdict={'fontsize': 30,
                                  'verticalalignment': 'baseline'})
    # make plot scatter
    plot1Axes.scatter(x, y, z)
    # label points
    i1 = 0
    for i1 in range(numCities):  # plot each point + it's index as text above
        plot1Axes.text(citiesMatrix[0, i1], citiesMatrix[1, i1], citiesMatrix[2, i1],
                       '%s' % (citiesNames[i1]), size=12, zorder=1,
                       color='k')
    # add a plot
    plot2Axes = graph1.add_subplot(122, projection='3d')
    # Name the plot
    plot2Axes.set_title('Best Route', fontdict={'fontsize': 30})
    # make plot of lines
    plt.plot(xAx, yAx, zAx)
    # label second plot
    i2 = 0
    for i2 in range(numCities):  # plot each point + it's index as text above
        plot2Axes.text(citiesMatrix[0, i2], citiesMatrix[1, i2], citiesMatrix[2, i2],
                       '%s' % (citiesNames[i2]), size=12, zorder=1, color='k')
    # show figure
    plt.show()


# run main method
if __name__ == "__main__":
    main()
