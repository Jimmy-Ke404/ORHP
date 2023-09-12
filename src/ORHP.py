from Network import Network
import matplotlib.pyplot as plt
# from scipy.optimize import linprog
import numpy as np
import time
import pickle
import sympy
# import matlab
# import matlab.engine
import networkx as nx
import os


def cal_incentives_dif(new_incentives, incentives):
    dif = []
    for key in incentives.keys():
        for i in range(len(incentives[key])):
            dif.append(abs(incentives[key][i] - new_incentives[key][i]))
    return sum(dif)/len(dif)

def cal_ELI_idx(bar_Delta, bar_Lambda, idx_eq_path):
    time_s = time.time()
    idx_LD = []
    mat = np.concatenate((bar_Delta, bar_Lambda), axis=0)
    cols = mat.shape[1]
    for i in range(cols):
        mat_before = np.delete(mat, idx_LD, axis=1)
        mat_after = np.delete(mat, idx_LD + [i], axis=1)
        if np.linalg.matrix_rank(mat_before) == np.linalg.matrix_rank(mat_after):
            idx_LD.append(i)
    # -------test
    # mat_after = np.delete(mat, idx_LD, axis=1)
    # print("the rank of mat_after is {}".format(np.linalg.matrix_rank(mat_after)))
    # try:
    #     B_inv = np.linalg.inv(mat_after)
    #     print("mat_after is invertible!")
    # except:
    #     print("mat_after is non-invertible, so calculate pseudo-inverse.")
    #     B_inv = np.linalg.pinv(mat_after)
    # -------test
    idx_LI = np.arange(cols)
    idx_LI = np.delete(idx_LI, idx_LD)
    idx = idx_eq_path[idx_LI]
    time_e = time.time()
    print("rank of bar_A_eq is {}, the number of equilibrated paths is {}, "
          "and the number of calculated equilibrated linear independent paths is {}"
          .format(np.linalg.matrix_rank(mat), cols ,len(idx)))
    print("time for cal_ELI_idx (second):", round(time_e-time_s, 3))
    return idx

def cal_gradient(nw, incentives, gamma, n, path_flows_driving, path_flows_ride_hailing, scheme):
    x_D, x_R, q_D, q_R = [], [], [], []
    for (u, v, d) in nw.graph.edges(data=True):
        x_D.append(nw.graph[u][v]["object"].flow_driving)
        x_R.append(nw.graph[u][v]["object"].flow_ride_hailing)
    for key in nw.od_path_set.keys():
        q_D.append(nw.od_vols_driving[key])
        q_R.append(nw.od_vols_ride_hailing[key])
    x = x_D + x_R
    q = q_D + q_R

    # option 4: find ELI paths
    Delta, Lambda, idx_eq_path = nw.get_incidence_mat2(path_flows_driving, path_flows_ride_hailing)
    bar_Delta = Delta[:, idx_eq_path]
    bar_Lambda = Lambda[:, idx_eq_path]
    idx = cal_ELI_idx(bar_Delta, bar_Lambda, idx_eq_path)
    print("length of idx_eq_path, idx", len(idx_eq_path), len(idx))
    # end of option 4 ----------------

    grad_x_wrt_omega = nw.cal_grad_x_wrt_omega(Delta, Lambda, idx)
    grad_z_wrt_x = nw.cal_grad_z_wrt_x(gamma)
    grad1 = grad_x_wrt_omega.T @ grad_z_wrt_x

    if scheme == "discriminatory subsidies":
        grad_z_wrt_omega = nw.cal_grad_z_wrt_omega(gamma)
        gradient = grad1 + grad_z_wrt_omega  # subsidy scheme
    elif scheme == "discriminatory tolls":
        gradient = grad1  # toll scheme
    elif scheme == "discriminatory unconstrained subsidies":
        gradient = grad1  # costs of subsidies are not considered in the objective function
    else:
        print('ERROR when cal_gradient! The scheme does not exist!')



    gradient = np.ravel(gradient)
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient = gradient/norm
    return gradient

def update_omega(omega, gradient, lr):
    gradient = np.ravel(gradient)
    i = 0
    new_omega = {}
    for key in omega.keys():
        new_omega[key] = omega[key] - lr * gradient[i]
        if new_omega[key] < 0:
            new_omega[key] = 0
        i += 1
    return new_omega

def update_omega_adagrad(omega, gradient_list, lr):
    d = len(omega.keys())
    G = [0] * d
    for gradient in gradient_list:
        gradient = np.ravel(gradient)
        for i in range(d):
            G[i] += gradient[i]**2
    i = 0
    new_omega = {}
    gradient = np.ravel(gradient_list[-1])
    # print("gradient in list", gradient)
    for key in omega.keys():
        if G[i] == 0:
            new_omega[key] = omega[key]
        else:
            new_omega[key] = omega[key] - lr * gradient[i] * (G[i] ** (-0.5))
            # print("i, gradient[i] * (G[i] ** (-0.5))", i, gradient[i] * (G[i] ** (-0.5)))
        if new_omega[key] < 0:
            new_omega[key] = 0
        i += 1
    return new_omega


def update_omega_adam(omega, gradient, lr, m, v, t):
    beta1, beta2, e = 0.9, 0.999, 10e-8
    m = beta1 * m + (1-beta1) * gradient
    v = beta2 * v + (1-beta2) * np.square(gradient)
    m_hat = m / (1-beta1**(t+1))
    v_hat = v / (1-beta2**(t+1))
    v_hat_sqrt = np.sqrt(v_hat)
    i = 0
    new_omega = {}
    for key in omega.keys():
        new_omega[key] = omega[key] - lr * m_hat[i] / (v_hat_sqrt[i] + e)
        if new_omega[key] < 0:
            new_omega[key] = 0
        i += 1
    return new_omega, m, v

def update_omegaRMSProp(omega, gradient, lr, v):
    gamma = 0.9
    v = gamma * v + (1-gamma) * np.square(gradient)
    e = 10e-6
    i = 0
    new_omega = {}
    for key in omega.keys():
        new_omega[key] = omega[key] - lr * gradient[i]/((v[i] + e) ** 0.5)
        if new_omega[key] < 0:
            new_omega[key] = 0
        i += 1
    return new_omega, v

def run_ME_with_subsidies(t, nw=None, omega=None, iteration_num1=10, iteration_num2=500, directory=None, epsilon_V=None, epsilon_I=None, SC=True):
    GAPs = []
    FC_list = []
    FC_list2 = []
    nw.apply_omega(omega)
    # step 0 of A1
    incentives = nw.get_zero_incentives()
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
            if epsilon_I:
                if g_driving <= epsilon_I and g_ride_hailing <= epsilon_I:
                    break
            if v < iteration_num2 - 1:
                path_flows_driving, path_flows_ride_hailing = nw.update_path_flows(path_flows_driving,
                                                                                   path_flows_ride_hailing,
                                                                                   auxiliary_path_flows_driving,
                                                                                   auxiliary_path_flows_ride_hailing,
                                                                                   lbd)
        # print("gap_driving", gap_driving[::50])
        # print("gap_ride_hailing", gap_ride_hailing[::50])
        # step 2 of A1
        last_incentives = incentives
        if SC:
            incentives = nw.update_incentives(i, last_incentives)  # done
        # step 3 of A1
        GAP = nw.cal_gap_d(path_flows_ride_hailing, incentives, last_incentives)  # done
        GAPs.append(GAP)
        FC = nw.cal_fleet_cost(path_flows_ride_hailing, incentives)
        FC2 = nw.cal_fleet_cost(path_flows_ride_hailing, last_incentives)
        FC_list.append(FC)
        FC_list2.append(FC2)
        if epsilon_V:
            if GAP <= epsilon_V:
                break
        # nw.check_assump_pos_x()

    # print("FC_list", FC_list)
    # print("FC_list2", FC_list2)
    # print("GAP_d before calculation of f*", GAPs[-1])
    # print("GAPs", GAPs)

    gap_driving = gap_driving[::iteration_num2//10]
    gap_ride_hailing = gap_ride_hailing[::iteration_num2//10]

    # plt.scatter(range(len(gap_driving)), gap_driving,  label="gap_driving[::I//10]")
    # plt.scatter(range(len(gap_ride_hailing)), gap_ride_hailing,  label="gap_ride_hailing[::I//10]")
    # plt.legend()
    # for i in range(len(gap_driving)):  # <--
    #     plt.text(i, gap_ride_hailing[i], round(gap_ride_hailing[i], 4), ha='center', va='bottom')
    # plt.xlabel("iteration")
    # plt.ylabel("gap")
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # name = 't=' + str(t) + '_inner_loop.png'
    # path = directory + "/output/ME_with_subsidies/"
    # if not os.path.exits(path):
    #     os.makedirs(path)
    # fig_file = path + name
    # # print(fig_file, "is saved")
    # plt.savefig(fig_file)
    # plt.cla()
    TTT = 0
    cost_omega = 0
    for (u, v, d) in nw.graph.edges(data=True):
        edge_TT = nw.graph[u][v]["time"] * nw.graph[u][v]["object"].flow
        TTT += edge_TT
        edge_cost = nw.graph[u][v]["object"].flow_ride_hailing * nw.graph[u][v]["object"].omega
        cost_omega += edge_cost
    # print(demand_ratio, 'TTT:', TTT)
    FTT = nw.cal_fleet_travel_time()
    output = [path_flows_driving, path_flows_ride_hailing, last_incentives, omega]

    return nw, TTT, FTT, cost_omega, path_flows_driving, path_flows_ride_hailing, last_incentives, output

def run_SA_based_solution_algorithm(downtown_factor, net_name, lr_info, gamma, directory, net_file, trip_file, node_file, demand_ratio, iter_N,
                                    iter_V=10, iter_I=500, path_set_size=10, mu_t=1, mu_u=0.5, mu_p=1.5,
                                    scheme="discriminatory subsidies", epsilon_V=None, epsilon_I=None, SC=True, 
                                    if_linear_cost=False, if_large_net=False, if_random_intial_omega=False, random_seed=0, omega_scale=1):
    TTT_list = []
    cost_omega_list = []
    obj_list = []
    omega_list = []
    gradient_list = []
    FTT_list = []
    nw = Network(downtown_factor=downtown_factor, net_name=net_name, net_file=net_file, trip_file=trip_file, node_file=node_file, path_set_size=path_set_size,
                 demand_ratio=demand_ratio, mu_t=mu_t, mu_u=mu_u, mu_p=mu_p, scheme=scheme, 
                 if_linear_cost=if_linear_cost, if_large_net=if_large_net)
    # get initial omega
    omega = nw.get_initial_omega(if_random_intial_omega, random_seed, omega_scale)
    # print("path_set.keys:", nw.od_path_set.keys())
    m = np.zeros(len(omega.keys()))
    v = np.zeros(len(omega.keys()))
    lr = lr_info[0]
    for n in range(iter_N):
        # omega_list.append(omega)
        if n % lr_info[1] == 0 and n != 0:
            lr = lr/lr_info[2]
        nw = Network(downtown_factor=downtown_factor, net_name=net_name, net_file=net_file, trip_file=trip_file, node_file=node_file, path_set_size=path_set_size,
                     demand_ratio=demand_ratio, mu_t=mu_t, mu_u=mu_u, mu_p=mu_p, scheme=scheme, 
                     if_linear_cost=if_linear_cost, if_large_net=if_large_net)

        nw, TTT, FTT, cost_omega, path_flows_driving, path_flows_ride_hailing, incentives, output = \
            run_ME_with_subsidies(t=n, nw=nw, omega=omega, iteration_num1=iter_V, iteration_num2=iter_I,
                                  directory=directory, epsilon_V=epsilon_V, epsilon_I=epsilon_I, SC=SC)
        TTT_list.append(TTT)
        cost_omega_list.append(cost_omega)
        obj_list.append(TTT+gamma*cost_omega)
        # print(n, "----record-----")
        # print('TTT_list', TTT_list)
        # print('obj_list', obj_list)
        gradient = cal_gradient(nw, incentives, gamma, n, path_flows_driving, path_flows_ride_hailing, scheme)
        gradient_list.append(gradient)
        FTT_list.append(FTT)
        # print("gradient", gradient)
        if n < iter_N-1:
            print("n, lr:", n, lr)
            # omega = update_omega(omega, gradient, lr)
            omega = update_omega_adagrad(omega, gradient_list, lr)
            # omega, m, v = update_omega_adam(omega, gradient, lr, m, v, n)
            # omega, v = update_omegaRMSProp(omega, gradient, lr, v)
        vc_list = nx.get_edge_attributes(nw.graph, 'vc')
        # print("vc_list:", vc_list)
        toll_list = nx.get_edge_attributes(nw.graph, 'toll')
        # print('toll_list:', toll_list)
    
    # calculate total subsidies for every path
    path_subsidies = nw.get_path_subsidies()
    path_TT = nw.get_path_TT()

    record = {
            'TTT_VS_iter': TTT_list,
            'subsidy_cost_VS_iter': cost_omega_list,
            'generalized_cost_VS_iter': obj_list,
            'FTT_VS_iter': FTT_list,
            'link VC ratios': vc_list,
            'link subsidies (tolls)': toll_list,
            'path_flows_driving': path_flows_driving, 
            'path_flows_ride_hailing': path_flows_ride_hailing,
            'path_subsidies': path_subsidies,
            'path_TT': path_TT
            }

    return record

