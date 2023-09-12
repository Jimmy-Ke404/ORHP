# This file run ME with SC and ME without SC, and then save the results as pickle files.

import networkx as nx
import math
import time
import matplotlib.pyplot as plt
import pickle
import os.path
import numpy as np
from Network import Network



def run_ME(downtown_factor, FB="FO", slope=None, demand_factor=None, SC=True,
            iteration_num1=20, iteration_num2=40, epsilon=0.001, 
           output_dir=None, net_file=None, trip_file=None, node_file=None, net_name=None, path_set_size=None, 
           mu_t=1, mu_u=0.5, mu_p=1.5, demand_ratio=0, if_linear_cost=False, 
           if_large_net=False, if_baseline=None):
    GAPs = []
    time_start = time.time()
    # step 0 of A1
    nw = Network(downtown_factor=downtown_factor, net_name=net_name, net_file=net_file, 
                 trip_file=trip_file, node_file=node_file, path_set_size=path_set_size, mu_t=mu_t,
                 mu_u=mu_u, mu_p=mu_p, demand_ratio=demand_ratio, slope=slope, demand_factor=demand_factor, 
                 FB=FB, if_linear_cost=if_linear_cost, if_large_net=if_large_net, if_baseline=if_baseline)

    # return 0

    incentives = nw.get_zero_incentives()  # done
    path_flows_driving, path_flows_ride_hailing = nw.get_initial_path_flows()

    for i in range(iteration_num1):
        # step 1 of A1
        gap_driving = []
        gap_ride_hailing = []
        for v in range(iteration_num2):
            lbd = 1.0 / (v + 1)
            # step 1 of A2
            nw.load_path_flow(path_flows_driving=path_flows_driving, path_flows_ride_hailing=path_flows_ride_hailing)
            # step 2 and 3 of A2
            auxiliary_path_flows_driving, auxiliary_path_flows_ride_hailing = nw.get_auxiliary_path_flows(
                incentives=incentives)
            # step 4 of A2

            g_driving, g_ride_hailing = nw.cal_gap(path_flows_driving, path_flows_ride_hailing)
            # print(v, g_driving, g_ride_hailing)
            gap_driving.append(g_driving)
            gap_ride_hailing.append(g_ride_hailing)

            if v < iteration_num2 - 1:
                path_flows_driving, path_flows_ride_hailing = nw.update_path_flows(path_flows_driving,
                                                                                   path_flows_ride_hailing,
                                                                                   auxiliary_path_flows_driving,
                                                                                   auxiliary_path_flows_ride_hailing,
                                                                                   lbd)
        time_end = time.time()
        # print('time cost of iteration', i, ':', time_end - time_start, 's')
        plt.plot(gap_driving, label='driving')
        plt.plot(gap_ride_hailing, label='ride hailing')
        plt.xlabel("iteration")
        plt.ylabel("gap of the inner loop")
        plt.legend()
        name = '{}' + 'demand_ratio=' + str(round(demand_ratio, 1)) + '_' + str(i) + '.png'

        if not os.path.exists(output_dir+ '/inner_loop_gaps/'):
            os.makedirs(output_dir+ '/inner_loop_gaps/')
        plt.savefig(name.format(output_dir+ '/inner_loop_gaps/'))
        plt.cla()

        # step 2 of A1
        last_incentives = incentives
        if SC:
            incentives = nw.update_incentives(i, last_incentives)  # done

            # # heuristic update
            # for key in incentives.keys():
            #     np_ar = i/(i+1) * np.array(last_incentives[key]) + 1/(i+1) * np.array(incentives[key])
            #     incentives[key] = np_ar.tolist()

        else:
            incentives = nw.get_zero_incentives()

        # step 3 of A2
        GAP = nw.cal_gap_d(path_flows_ride_hailing, incentives, last_incentives)  # done
        GAPs.append(GAP)
        # if GAP <= epsilon:
        #     break
    # nw.print_VC()

    FC = nw.cal_fleet_cost(path_flows_ride_hailing, last_incentives)
    # print('generalized fleet cost without tolls is {}'.format(FC))

    # save path flows and incentives (compensations)
    output = [path_flows_driving, path_flows_ride_hailing, last_incentives]
    output_file = '{}' + 'demand_ratio=' + str(round(demand_ratio, 1)) + '_output.pickle'

    plt.plot(GAPs)
    plt.xlabel("iteration")
    plt.ylabel("gap of the outer loop")
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    name = '{}' + 'demand_ratio=' + str(round(demand_ratio, 1)) + 'outer_loop.png'

    # if SC:
    #     output_path = directory + "output/ME_with_SC/"
        
    # else:
    #     output_path = directory + "output/ME_without_SC/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle_file = output_file.format(output_dir)
    fig_file = name.format(output_dir)
    with open(pickle_file, 'wb') as f:
        pickle.dump(output, f)
    # print(fig_file, "is saved")
    plt.savefig(fig_file)
    plt.cla()

    TTT = 0
    for (u, v, d) in nw.graph.edges(data=True):
        edge_TT = nw.graph[u][v]["time"] * nw.graph[u][v]["object"].flow
        TTT += edge_TT
    FTT = nw.cal_fleet_travel_time()
    # if FB == 'SO':
    #     ME = 'ME-SO'
    # elif SC:
    #     ME = 'ME-FOSC'
    # else:
    #     ME = 'ME-FO'
    vc_list = nx.get_edge_attributes(nw.graph, 'vc')

    # path = "{}output/{}/".format(directory, ME)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # save VC ratios
    file_name = '{}VC ratios with demand ratio of {}.pickle'.format(output_dir, round(demand_ratio, 1))
    with open(file_name, 'wb') as f:
        pickle.dump(vc_list, f)
        print("V/C ratios are dumped to {}.".format(file_name))


    return TTT, FTT, FC, path_flows_ride_hailing, nw.path_costs_driving, nw.od_path_set


