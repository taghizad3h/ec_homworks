import logging
import operator


from numpy import genfromtxt
import numpy as np
from data_helper import get_first_dataset
from gprs import GPRegressionSolver
from tools import draw
from es import ES
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

################  config  ###############

# random.seed(318)
constants = [0.1, 1.1]
data_path = 'data/Concrete_Data.csv'
population_size = 300
verbose = True
mating_prob = 0.5
mutating_prob = 0.1
n_generations = 10
parsimonyPressure = False
parsimonyW1 = 0.01
parsimonyW2 = 0.99
tournsize = 3


############ preparing data #############
X, y, var_names = get_first_dataset(data_path)
constants = np.array(constants, dtype=float)
if len(var_names)>0:
    for i in range(len(constants)):
        var_names.append('constant_'+str(i))



########### running algorithm ###########
gpsolver = GPRegressionSolver(X, y, constants=constants, var_names=var_names)
last_pop, log = gpsolver.solve(n_generations=n_generations)


######## saving best solution found #########
best = max(last_pop, key=operator.attrgetter("fitness"))
logging.info('best individual\'s fittness: '+ str(best.fitness))
draw(best, 'out/best_ind.pdf')



######### using evolution strategy ##########

def my_cost(constants):
    return gpsolver.simpleFitness(best, constants)[0]

logging.info('current constatns '+str(constants))
mes = ES(len(constants), my_cost)
mes.set_ans(constants)
mes.evolve()
logging.info('best found constants '+str(mes.ans))