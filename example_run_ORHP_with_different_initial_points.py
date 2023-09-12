import sys
sys.path.insert(0, 'src')
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sympy
import networkx as nx
import os
from src.ORHP import run_SA_based_solution_algorithm
from multiprocessing import Pool
from functools import partial

def run_ORHP_with_random_initializaiton(network, random_seed):
    if network == 'SiouxFalls':
        # ------- hyperparameters for the SiouxFalls network -------
        directory = "Data/Networks/SiouxFalls/"
        net_file = '{}SiouxFalls_net.tntp'.format(directory)
        trip_file = '{}SiouxFalls_trips (GA).tntp'.format(directory)
        node_file = '{}SiouxFalls_node.tntp'.format(directory)
        net_name = 'SiouxFalls'
        downtown_factor = 1
        N1 = 10
        N2 = 500
        N3 = 50
        path_set_size = 10
        mu_u = 0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost = True
        if_large_net = False
        lr_info = [0.5, 1, 1]
        omega_scale = 4
        gamma = 0
    elif network == 'PGH':
        # ------- hyperparameters for the Pittsburgh network -------
        directory = "Data/Networks/Pittsburgh/"
        net_file = '{}pitts_net_new_0716.tntp.txt'.format(directory)
        trip_file = '{}pitts_trips_new_0716.tntp.txt'.format(directory)
        node_file = None
        net_name = 'PGH'
        downtown_factor = 1
        N1 = 10
        N2 = 500
        N3 = 70
        path_set_size = 15
        mu_u = 0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost = False
        if_large_net = True
        lr_info = [0.01, 1, 1]
        omega_scale = 0.1
        gamma = 0

    scheme="discriminatory subsidies"
    record = \
        run_SA_based_solution_algorithm(downtown_factor=downtown_factor, net_name=net_name,
            lr_info=lr_info, gamma=gamma, directory=directory, net_file=net_file,
            trip_file=trip_file, node_file=node_file, demand_ratio=0.5, iter_N=N3, iter_V=N1, iter_I=N2,
            path_set_size=path_set_size, mu_t=mu_t, mu_u=mu_u, mu_p=mu_p, scheme=scheme, SC=True, 
                                        if_linear_cost=if_linear_cost, if_large_net=if_large_net, 
                                        if_random_intial_omega=True, random_seed=random_seed, omega_scale=omega_scale)

    path = directory + f'output/ORHP/mu_t_{mu_t}_mu_p_{mu_p}_downtown_{downtown_factor}/'
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + "gamma={} lr={} scheme={} random_seed={}.pickle".format(gamma, lr_info[0], scheme, random_seed)
    with open(file_name, 'wb') as f:
            """
            record = [TTT_list, cost_omega_list, obj_list, FTT_list, _, vc_list, toll_list] 
            """
            pickle.dump(record, f) 
            print('Results of ORHP are dumped into file: {}.'.format(file_name))

if __name__ == '__main__':
    # parameters for the run_ORHP_with_random_initializaiton function
    network = 'SiouxFalls'
    seed_list=[0, 1, 2, 3, 4]

    # use parallel processes to distribute tasks with different random seeds
    num_cpus = 2  # scale up if you have more cpus
    st_time = time.time()
    partial_run_ORHP = partial(run_ORHP_with_random_initializaiton, network=network)
    with Pool(num_cpus) as p:
        print(p.map(partial_run_ORHP, seed_list))
    end_time = time.time()
    print(f'ORHP of {network} with random seeds of {seed_list} is done! Running time took {end_time-st_time} seconds.')