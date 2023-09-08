import pickle
import os.path
import numpy as np
from src.ME import run_ME
import os

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
# ------- hyperparameters for the SiouxFalls network -------

# ------- hyperparameters for the Pittsburgh network -------
# directory = "Data/Networks/Pittsburgh/"
# net_file = '{}pitts_net_new_0716.tntp.txt'.format(directory)
# trip_file = '{}pitts_trips_new_0716.tntp.txt'.format(directory)
# node_file = None
# net_name = 'PGH'

# N1=10
# N2=500
# epsilon=0.001
# path_set_size=15
# mu_u = 0.5
# mu_t = 1.5
# mu_p = 2.0

# if_linear_cost=False
# if_large_net=True
# ------- hyperparameters for the Pittsburgh network -------

demand_ratio_list = np.arange(0, 1.1, 0.1)
# demand_ratio_list = [0.5]
# ME_list = ['ME-FO', 'ME-SO', 'ME-FOSC', 'Baseline-SO', 'Baseline-UE']
ME_list = ['ME-FO', 'ME-SO', 'Baseline-SO']
for ME in ME_list: 
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
    
    # for downtown_factor in [1, 2]:
    for downtown_factor in [1]:
        output_dir = "{}output/{}/mu_t_{}_mu_p_{}_downtown_{}/".format(directory, ME, mu_t, mu_p, downtown_factor)
        TTT_list, FTT_list, FC_list, path_flows_RH_list, path_TT_list, path_set_list = [], [], [], [], [], []

        for demand_ratio in demand_ratio_list:
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