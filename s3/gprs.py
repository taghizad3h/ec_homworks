import logging
import math
import operator
import random
from operator import attrgetter

import numpy as np
from deap import algorithms, base, creator, gp, tools
from deap.tools.selection import selRandom
from numpy import genfromtxt


def protectedDiv(left, right):

    if math.isinf(left):
        left = 987

    if math.isinf(right):
        right = 564

    if right == 0:
        right = .1

    try:
        ans = left / right
        if math.isnan(ans) or math.isinf(ans):
            return 1
        else:
            return ans
    except ZeroDivisionError:
        return 1


class GPRegressionSolver:

    def __init__(self, X, y, constants=[1,2],
         use_parsimony_pressure=False, parsimony_pressure_w1=0.1, parsimony_pressure_w2=0.9, tourn_size=3
        , var_names=[], min_height=1, max_height=10, mut_min_height=0, mut_max_height=2):

        self.X = X
        self.y = y
        self.constants = constants
        self.parsimony_pressure_w1 = parsimony_pressure_w1
        self.parsimony_pressure_w2 = parsimony_pressure_w2
        n_args=len(X[0])+len(constants)
        pset = gp.PrimitiveSet("MAIN", n_args)
        self.pset = pset
        self.creator = creator
        toolbox = base.Toolbox()
        self.toolbox = toolbox


        ########## declaring types ############
        
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        # pset.addPrimitive(math.cos, 1)
        # pset.addPrimitive(math.sin, 1)
        # pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1), float)
        # pset.addTerminal(1)

        for i in range(len(var_names)):
            key = 'ARG'+str(i)
            val = var_names[i]
            pset.renameArguments(**{key:val})
        

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        

        ######## initializig vriables ##########
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.nsgaiiFitness, constants=constants)
        toolbox.register("select", tools.selDoubleTournament, fitness_size =5 , parsimony_size=1.4, fitness_first=False)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=mut_min_height, max_=mut_max_height)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))


    def simpleFitness(self, individual, constants):
        logging.debug('in')
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        sqerrors = []
        for x, yy in zip(self.X, self.y):
            args = np.concatenate((x, constants))
            ans = func(*args)
            if math.isnan(ans):
                print(individual)
                print(args)
                exit()
            sqerrors.append(math.sqrt((ans - yy)**2))
        fitness = math.fsum(sqerrors) / len(self.X)
        logging.debug('out')
        return fitness

    def nsgaiiFitness(self, individual, constants):
        fitness = self.simpleFitness(individual, constants)
        return fitness, len(individual)

    def solve(self, pop_size=300, n_generations=100, mating_prob=0.5, mut_prob=0.1, verbose=True):
        logging.debug('solving')
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        mstats = tools.MultiStatistics(fitness=stats_fit)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)
        logging.debug('running')
        pop, log = algorithms.eaSimple(pop, self.toolbox, mating_prob, mut_prob, n_generations, stats=mstats, halloffame=hof, verbose=verbose)

        return pop, log
