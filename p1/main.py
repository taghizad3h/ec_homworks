from evo_strategy import ES
from cost_functions import rosen, ackely
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


n_dims = 3
cost_func = rosen
iterations = 100000

es = ES(n_dims, cost_func)

es.evolve(iterations=iterations)
