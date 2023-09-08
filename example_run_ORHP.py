import time
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sympy
import networkx as nx
import os
from src.ORHP import run_SA_based_solution_algorithm
from multiprocessing import Pool

# ------- hyperparameters for the SiouxFalls network -------
directory = "Data/Networks/SiouxFalls/"
net_file = '{}SiouxFalls_net.tntp'.format(directory)
trip_file = '{}SiouxFalls_trips (GA).tntp'.format(directory)
node_file = '{}SiouxFalls_node.tntp'.format(directory)
net_name = 'SiouxFalls'
downtown_factor=1

N1=10
N2=500
N3=50
epsilon=0.001
path_set_size=10
mu_u=0.5
mu_t = 1.5
mu_p = 2.0
if_linear_cost=True
if_large_net=False
lr_info = [0.5, 1, 1]
gamma_list = np.round(np.arange(0.1, 1.1, 0.1), 1)
# ------- hyperparameters for the SiouxFalls network -------


# ------- hyperparameters for the Pittsburgh network -------
# directory = "Data/Networks/Pittsburgh/"
# net_file = '{}pitts_net_new_0716.tntp.txt'.format(directory)
# trip_file = '{}pitts_trips_new_0716.tntp.txt'.format(directory)
# node_file = None
# net_name = 'PGH'
# downtown_factor=1

# N1=10
# N2=500
# N3=70
# epsilon=0.001
# path_set_size=15
# mu_u=0.5
# mu_t = 1.5
# mu_p = 2.0
# if_linear_cost=False
# if_large_net=True
# lr_info = [0.01, 1, 1]
# gamma_list = np.round(np.arange(0.1, 1.1, 0.2), 1)
# ------- hyperparameters for the Pittsburgh network -------

def run_ORHP(gamma):
    print(gamma)
    scheme="discriminatory subsidies"
    record = \
        run_SA_based_solution_algorithm(downtown_factor=downtown_factor, net_name=net_name,
            lr_info=lr_info, gamma=gamma, directory=directory, net_file=net_file,
            trip_file=trip_file, node_file=node_file, demand_ratio=0.5, iter_N=N3, iter_V=N1, iter_I=N2,
            path_set_size=path_set_size, mu_t=mu_t, mu_u=mu_u, mu_p=mu_p, scheme=scheme, SC=True, 
                                        if_linear_cost=if_linear_cost, if_large_net=if_large_net)

    path = directory + f'output/ORHP/mu_t_{mu_t}_mu_p_{mu_p}_downtown_{downtown_factor}/'
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + "gamma={} lr={} scheme={}.pickle".format(gamma, lr_info[0], scheme)
    with open(file_name, 'wb') as f:
            """
            record = [TTT_list, cost_omega_list, obj_list, FTT_list, _, vc_list, toll_list] 
            """
            pickle.dump(record, f) 
            print('Results of ORHP are dumped into file: {}.'.format(file_name))

if __name__ == '__main__':
    # gamma_list = np.round(np.arange(0.1, 1.1, 0.1), 1)
    # gamma_list = [0.0]
    st_time = time.time()
    num_cpus = 5
    with Pool(num_cpus) as p:
        print(p.map(run_ORHP, gamma_list))
    end_time = time.time()
    print(f'ORHP of {net_name} with gamma of {gamma_list} is done! Running time took {end_time-st_time} seconds.')