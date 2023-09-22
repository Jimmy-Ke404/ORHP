import sys
sys.path.insert(0, 'src')
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from src.ORHP import run_SA_based_solution_algorithm
from multiprocessing import Pool
from functools import partial
import argparse

def run_ORHP(gamma, network=None):
    """
    This function solves optimal ride-haling pricing, which mitigates network congestion by inducing, using subsidies,
    ride-hailing vehicles to alternative routes so as to relieve over-congested roads.

    Args:
        gamma (float, should be non-negative): 
            A value measures tradeoff between total travel time reduction and subsidy cost. gamma=0 means the objective 
            function tries to minimize the total travel time no matter how much the subsidy cost is. Larger gamma means 
            less total travel time reduction and lower subsidy cost.
        network (str):
            The network name. Currently supported networks are 'SiouxFalls' (i.e., the Sioux Falls network) 
            and 'PGH' (i.e., the Pittsburgh network).

    The descriptions of hyperparameters are as follows:
        downtown_factor: the downtown factor measures the ratio of the ride-hailng penatration rate on links in the 
            downtown area and the ride-hailing penatration rate on the other links.
        N1: maximum steps of the algorithm 3 in the paper
        N2: maximum steps of the algorithms 1 and 2 in the paper
        N3: maximum steps of the algorithm 4 in the paper
        path_set_size: the number of paths in the path set of each OD pair
        mu_u: the value of time of the ride-hailing passengers
        mu_t: the operating cost of travel time for the ride-hailing company
        mu_p: the average price of ride-hailing per unit travel time
        if_linear_cost: whether the link travel time function is linear
        if_large_net: whether the network is large. If so, using an approximation method to find paths, and using 
            enumeration otherwise
        lr_info: the learning rate is initialized as lr_info[0], and every lr_info[1] steps, the learning rate = 
            learning rate / lr_info[2].
    """
    
    if network == 'SiouxFalls':
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
        path_set_size=10
        mu_u=0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost=True
        if_large_net=False
        lr_info = [0.5, 1, 1]
    elif network == 'PGH':
        # ------- hyperparameters for the Pittsburgh network -------
        directory = "Data/Networks/Pittsburgh/"
        net_file = '{}pitts_net_new_0716.tntp.txt'.format(directory)
        trip_file = '{}pitts_trips_new_0716.tntp.txt'.format(directory)
        node_file = None
        net_name = 'PGH'
        downtown_factor=1
        N1=10
        N2=500
        N3=70
        path_set_size=15
        mu_u=0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost=False
        if_large_net=True
        lr_info = [0.01, 1, 1]

    scheme="discriminatory subsidies"
    # solve optimal pricing using a sensitivity analysis-based method
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
    network_list = ['SiouxFalls', 'PGH']  # supported networks

    # parse arguments
    parser = argparse.ArgumentParser(
                    description='This script solves ORHP on top of ME-FOSC.')
    parser.add_argument('network', choices=network_list,
                        help=f'Choose a network from {network_list}')
    parser.add_argument('--num_cpus', default=2,
                        help='The number of CPUs used for task parallel.')
    args = parser.parse_args()

    # use parallel processes to distribute tasks with different gammas
    gamma_list = np.round(np.arange(0.1, 1.1, 0.1), 1)
    num_cpus = args.num_cpus  # scale up if you have more cpus
    st_time = time.time()
    partial_run_ORHP = partial(run_ORHP, network=args.network)
    with Pool(num_cpus) as p:
        p.map(partial_run_ORHP, gamma_list)
    end_time = time.time()
    print(f'ORHP of {args.network} with gamma of {gamma_list} is done! Running time took {end_time-st_time} seconds.')