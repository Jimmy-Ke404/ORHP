import sys
sys.path.insert(0, 'src')
import pickle
import numpy as np
from src.ME import run_ME
import os
import argparse

def run_example_ME(ME='ME-FOSC', network='SiouxFalls',downtown_factor_list=[1], demand_ratio_list = np.arange(0, 1.1, 0.1)):
    """
    This function runs various network equilibria on a given network with different ride-hailing penatration rates.

    Args:
        ME (str): 
            Mixed equilibrium type. Currently supported input values are as follows.
            - 'ME-FO': Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following FO.
            - 'ME-SO': Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following SO.
            - 'ME-FOSC': Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following FOSC.
            - 'Baseline-SO': A baseline where all vehicles follow SO.
            - 'Baseline-UE': A baseline where all vehicles follow UE.
        network (str):
            The network name. Currently supported networks are 'SiouxFalls' (i.e., the Sioux Falls network) 
            and 'PGH' (i.e., the Pittsburgh network).
        downtown_factor_list (a list of ints):
            A list of downtown factor values to experiment on. The downtown factor measures the ratio of the 
            ride-hailng penatration rate on links in the downtown area and the ride-hailing penatration rate on
            the other links.
        demand_ratio_list (a list of floats):
            A list of ride-hailing demand ratios to experiment on. For example, a demand ratio of 0.2 means
            the ride-hailing demands count 20% of total travel demands.

            
    The descriptions of hyperparameters are as follows:
        N1: maximum steps of the algorithm 3 in the paper
        N2: maximum steps of the algorithms 1 and 2 in the paper
        epsilon: threshold of the gap function to stop the algorithm
        path_set_size: the number of paths in the path set of each OD pair
        mu_u: the value of time of the ride-hailing passengers
        mu_t: the operating cost of travel time for the ride-hailing company
        mu_p: the average price of ride-hailing per unit travel time
        if_linear_cost: whether the link travel time function is linear
        if_large_net: whether the network is large. If so, using an approximation method to find paths, and using enumeration otherwise
    """
    if network == 'SiouxFalls':
        # ------- hyperparameters for the SiouxFalls network -------
        directory = "Data/Networks/SiouxFalls/"
        net_file = '{}SiouxFalls_net.tntp'.format(directory)
        trip_file = '{}SiouxFalls_trips (GA).tntp'.format(directory)
        node_file = '{}SiouxFalls_node.tntp'.format(directory)
        net_name = 'SiouxFalls'
        N1=10 
        N2=500
        epsilon=0.001
        path_set_size=10
        mu_u=0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost=True
        if_large_net=False
    elif network == 'PGH':
        # ------- hyperparameters for the Pittsburgh network -------
        directory = "Data/Networks/Pittsburgh/"
        net_file = '{}pitts_net_new_0716.tntp.txt'.format(directory)
        trip_file = '{}pitts_trips_new_0716.tntp.txt'.format(directory)
        node_file = None
        net_name = 'PGH'
        N1=10
        N2=500
        epsilon=0.001
        path_set_size=15
        mu_u = 0.5
        mu_t = 1.5
        mu_p = 2.0
        if_linear_cost=False
        if_large_net=True

    if 'FO' in ME:
        FB = 'FO'
    else:
        FB = 'SO'
    if 'SC' in ME:
        SC = True
    else:
        SC = False

    if_baseline = None
    if 'Baseline' in ME:
        FB = 'SO'
        if 'UE' in ME:
            if_baseline = 'UE'
        else:
            if_baseline = 'SO'
    
    for downtown_factor in downtown_factor_list:
        output_dir = "{}output/{}/mu_t_{}_mu_p_{}_downtown_{}/".format(directory, ME, mu_t, mu_p, downtown_factor)
        TTT_list, FTT_list, FC_list, path_flows_RH_list, path_TT_list, path_set_list = [], [], [], [], [], []

        for demand_ratio in demand_ratio_list:
            # run a ME with a demand ratio
            TTT, FTT, FC, path_flows_RH, path_TT, path_set = run_ME(downtown_factor=downtown_factor,
                            FB=FB, SC=SC, iteration_num1=N1, iteration_num2=N2,
                            epsilon=epsilon, output_dir=output_dir, net_file=net_file,
                            trip_file=trip_file, node_file=node_file, net_name=net_name, 
                            path_set_size=path_set_size, mu_t=mu_t, mu_u=mu_u, mu_p=mu_p,
                            demand_ratio=demand_ratio, if_linear_cost=if_linear_cost, 
                            if_large_net=if_large_net, if_baseline=if_baseline)
            TTT_list.append(TTT)
            FTT_list.append(FTT)
            FC_list.append(FC)
            path_flows_RH_list.append(path_flows_RH)
            path_TT_list.append((path_TT))
            path_set_list.append(path_set)
            print('{} with ride-hailing demand ratio of {} is done.'.format(ME, round(demand_ratio, 1)))
        if SC:
            ME = "ME-FOSC"
        else:
            ME = "ME-{}".format(FB)
        record = {}
        record['TTT_list'] = TTT_list
        record['FTT_list'] = FTT_list
        record['FC_list'] = FC_list
        record['RH_path_flows_list'] = path_flows_RH_list
        record['Path_TT_list'] = path_TT_list

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open('{}record of TTT_FTT_FC.pickle'.format(output_dir), 'wb') as f:
            pickle.dump(record, f)
            print(record)
            print('{}record of TTT_FTT_FC.pickle is saved.'.format(output_dir))

if __name__ == '__main__':
    # supported parameters
    ME_list = ['ME-FO', 'ME-SO', 'ME-FOSC', 'Baseline-SO', 'Baseline-UE']  # supported equilibria
    network_list = ['SiouxFalls', 'PGH']  # supported networks

    # parse arguments
    parser = argparse.ArgumentParser(
                    description='This script solves the UE, the SO, and various ME')
    parser.add_argument('network', choices=network_list,
                        help=f'Choose a network from {network_list}')
    parser.add_argument('equilibrium_type', choices=ME_list,
                        help=f'Choose a equilibrium from {ME_list}')
    args = parser.parse_args()

    # solve the requested equilibrium
    run_example_ME(ME=args.equilibrium_type, network=args.network)
    