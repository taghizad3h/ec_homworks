from evo_strategy import ES
from cost_functions import rosen, ackely
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


## settings
n_dims = 1000
cost_func = rosen
iterations = 1000
consider_cov = False

es = ES(n_dims, cost_func)
es.evolve(iterations=iterations, consider_cov=consider_cov)
# es.plot_history()
es.save_history('second_model_cov_rosen_1000dim')



n_dims = 1000
cost_func = ackely
iterations = 1000
consider_cov = False

es = ES(n_dims, cost_func)
es.evolve(iterations=iterations, consider_cov=consider_cov)
# es.plot_history()
es.save_history('second_model_cov_ackely_1000dim')