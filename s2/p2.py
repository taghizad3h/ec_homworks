import logging
import operator


from numpy import genfromtxt
import numpy as np
from data_helper import get_first_dataset
from gprs import GPRegressionSolver
from tools import draw
from es import ES
from sklearn.model_selection import train_test_split


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


################  config  ###############

# random.seed(318)
constants = [-1.1, 0.1, 1.1]
data_path = 'data/Concrete_Data.csv'
population_size = 300
verbose = True
mating_prob = 0.5
mutating_prob = 0.1
n_generations = 1000
parsimonyPressure = True
parsimonyW1 = 0.01
parsimonyW2 = 0.99
tournsize = 3
min_height=1
max_height=5
mut_min_height=0
mut_max_height=4


############ preparing data #############
X, y, var_names = get_first_dataset(data_path)

constants = np.array(constants, dtype=float)
if len(var_names)>0:
    for i in range(len(constants)):
        var_names.append('constant_'+str(i))


########### running algorithm ###########
gpsolver = GPRegressionSolver(X, y, constants=constants, var_names=var_names, 
    use_parsimony_pressure=parsimonyPressure, parsimony_pressure_w1=parsimonyW1, parsimony_pressure_w2=parsimonyW2
    , tourn_size=tournsize, min_height=min_height, max_height=max_height, mut_min_height=mut_min_height
    , mut_max_height=mut_max_height)

last_pop, log = gpsolver.solve(n_generations=n_generations)
best_fitness = log.chapters['fitness'].select('min')
avg_size = log.chapters['size'].select('avg')
np.savetxt('best_fits_parsimonyp2.txt', best_fitness)
np.savetxt('avg_size_parsimonyp2.txt', avg_size)


######## saving best solution found #########
best = max(last_pop, key=operator.attrgetter("fitness"))
logging.info('best individual\'s fittness: '+ str(best.fitness))
draw(best, 'out/best_ind.pdf')