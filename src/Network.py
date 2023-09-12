# This script builds network-related classes

from ast import Pass
import networkx as nx
import math
import time
import matplotlib.pyplot as plt
import pickle
import os.path
import numpy as np
import sympy
from scipy.linalg import lu
import scipy
import queue
import sys
from Node_Link import Node, Link
from generate_driving_ridehailing_demands import generate_pickup_idle_flows, split_demands

def get_ind_LI_row(mat):
    mat = np.mat(mat, dtype=int)
    P,L,U = lu(mat.T)
    idx = []
    for r in range(U.shape[0]):
        mylist = U[r, :]
        d = next((i for i, x in enumerate(mylist) if abs(x) > 10e-5), None)
        if d in idx:
            continue;
        else:
            idx.append(d)

    return idx

def reduce_singular_mat(mat):
    # delete linearly dependent columns
    idx_LD_cols = []
    cols = mat.shape[1]
    for i in range(cols):
        mat_before = np.delete(mat, idx_LD_cols, axis=1)
        mat_after = np.delete(mat, idx_LD_cols + [i], axis=1)
        if np.linalg.matrix_rank(mat_before) == np.linalg.matrix_rank(mat_after):
            idx_LD_cols.append(i)

    mat = np.delete(mat, idx_LD_cols, axis=1)

    # delete linearly dependent rows
    idx_LD_rows = []
    rows = mat.shape[0]
    for i in range(rows):
        mat_before = np.delete(mat, idx_LD_rows, axis=0)
        mat_after = np.delete(mat, idx_LD_rows + [i], axis=0)
        if np.linalg.matrix_rank(mat_before) == np.linalg.matrix_rank(mat_after):
            idx_LD_rows.append(i)
    mat = np.delete(mat, idx_LD_rows, axis=0)

    return mat, idx_LD_cols, idx_LD_rows


class Network:
    """
    Class for handling Transportation Networks. This class contains methods to read various TNTP format files from the source and methods of network-wide operations

    Parameters
    ----------
    link_file :     string
                    file path of network file, which containing various link information

    trip_file :     string
                    file path of trip table. An Origin label and then Origin node number, followed by Destination node numders and OD flow

    node_file :     string
                    file path of node file, which containing coordinates information of nodes

    SO:             boolean
                    True if objective is to find system optimal solution,
                    False if objective is to find user equilibrium
    Attributes
    ----------
    graph :         networkx.DiGrapy
                    graph of links with Link object and travel time under the current condition

    origins :       list
                    list of origin nodes

    od_vols :       dictionary
                    key: tuple(origin node, destination node), value: traffic flow
    """
    link_fields = {"from": 1, "to": 2, "capacity": 3, "length": 4, "t0": 5, "B": 6, "beta": 7, "V": 8}

    def __init__(self, downtown_factor=1, net_name=None, net_file=None, trip_file=None, node_file=None, path_set_size=1000, 
                 demand_ratio=0.5, mu_t=1, mu_u=0.5, mu_p=1.5, pickup_rate=0.05, idle_rate=0.05,
                 scheme="discriminatory subsidies", slope=None, demand_factor=None, 
                 FB="FO", if_linear_cost=False, if_large_net=False, if_baseline=None):
        
        self.downtown_factor = downtown_factor
        self.net_name = net_name
        self.net_file = net_file
        self.trip_file = trip_file
        self.node_file = node_file
        self.graph = None
        self.od_path_set = {}  # self.od_path_set[o, d] is the path set of od.
        self.path_costs_driving = {}
        self.path_costs_ride_hailing = {}
        self.path_set_size = path_set_size
        self.demand_ratio = demand_ratio
        self.mu_t = mu_t
        self.mu_u = mu_u
        self.mu_p = mu_p
        self.pickup_rate = pickup_rate
        self.idle_rate = idle_rate
        self.od_vols = {}
        self.od_vols_driving = {}
        self.od_vols_ride_hailing = {}
        self.scheme = scheme  # "discriminatory subsidies", "discriminatory tolls", "anonymous tolls"
        self.slope = slope
        self.demand_factor = demand_factor
        self.FB = FB
        self.len_cap = 2.0
        self.if_linear_cost = if_linear_cost
        self.if_large_net = if_large_net
        self.if_baseline = if_baseline

        self.build_datastructure()


    def build_datastructure(self):
        """
        Method for opening .tntp format network information files and preparing variables for the analysis
        """
        links, nodes = self.open_net_file()
        self.open_trip_file()

        graph = nx.DiGraph()

        for l in links:
            graph.add_edge(l.from_node, l.to_node, object=l, time=l.get_time(), MFC=l.get_MFC(), vc=l.get_VC_(),
                           toll=l.get_toll())

        if self.node_file != None:
            self.open_node_file(graph)
            # Visualization.reLocateLinks(graph)
        self.graph = graph

        self.initialize_path_set()

        self.optimal_paths_driving = {}
        self.optimal_paths_ride_hailing = {}
        for key in self.od_vols.keys():
            self.optimal_paths_driving[key] = []
            self.optimal_paths_ride_hailing[key] = []
            self.path_costs_driving[key] = []
            self.path_costs_ride_hailing[key] = []

    def open_net_file(self):
        """
        Method for opening network file, containing various link information

        Returns
        -------
        list
            list of Link objects having current link condition
        list
            list of Node objects

        """
        # the input files of the Pittsburgh network have different formats
        if 'Pittsburgh' in self.net_file: 
            with open(self.net_file, 'r') as f:
                lines = f.readlines()
            nodes = {}
            links = []
            for line in lines:
                data = line.split(" ")
                if data[0] == "init":
                    continue;
                try:
                    origin_node = str(int(float(data[0])))
                except IndexError:
                    continue
                to_node = str(int(float(data[1])))
                capacity = float(data[2])
                length = float(data[3])
                t0 = float(data[4])
                alpha = float(data[5])
                beta = float(data[6])

                if origin_node not in nodes:
                    n = Node(node_id=origin_node)
                    nodes[origin_node] = n

                if to_node not in nodes:
                    n = Node(node_id=to_node)
                    nodes[to_node] = n

                l = Link(link_id=len(links), length=length, capacity=capacity, t0=t0, alpha=alpha, beta=beta,
                        from_node=origin_node, to_node=to_node, flow=float(0.0), demand_ratio=self.demand_ratio,
                        mu_t=self.mu_t, scheme=self.scheme, slope=self.slope, FB=self.FB, net_file=self.net_file, 
                        if_linear_cost=self.if_linear_cost)
                links.append(l)
            
            
        else:
            with open(self.net_file, 'rb') as f:
                lines = f.readlines()

            links_info = []

            header_found = False
            for line in lines:
                if not header_found and line.startswith(b"~"):
                    header_found = True
                elif header_found:
                    links_info.append(line)

            nodes = {}
            links = []

            for line in links_info:
                data = line.split(b"\t")

                try:
                    origin_node = str(int(data[self.link_fields["from"]]))
                except IndexError:
                    continue
                to_node = str(int(data[self.link_fields["to"]]))
                capacity = float(data[self.link_fields["capacity"]])
                length = float(data[self.link_fields["length"]])
                t0 = float(data[self.link_fields["t0"]])
                alpha = float(data[self.link_fields["B"]])
                beta = float(data[self.link_fields["beta"]])

                if origin_node not in nodes:
                    n = Node(node_id=origin_node)
                    nodes[origin_node] = n

                if to_node not in nodes:
                    n = Node(node_id=to_node)
                    nodes[to_node] = n

                l = Link(link_id=len(links), length=length, capacity=capacity, t0=t0, alpha=alpha, beta=beta,
                        from_node=origin_node, to_node=to_node, flow=float(0.0), demand_ratio=self.demand_ratio,
                        mu_t=self.mu_t, scheme=self.scheme, slope=self.slope, FB=self.FB, net_file=self.net_file, 
                        if_linear_cost=self.if_linear_cost)
                links.append(l)
        return links, nodes.values()

    def open_node_file(self, graph):
        """
        Method for opening node file, containing position information of nodes \n
        This method adds 'pos' key-value pair in graph variable
        """
        with open(self.node_file, 'rb') as f:
            n = 0
            for i in f:
                row = i.split(b"	")
                if n == 0:
                    n += 1

                else:
                    try:
                        if self.node_file == "berlin-center_node.tntp":
                            ind, x, y = str(int(row[0])), float(row[1]), float(row[3])
                        else:
                            ind, x, y = str(int(row[0])), float(row[1]), float(row[2])
                        graph.nodes[ind]["pos"] = (x, y)
                    except:
                        # print(row)
                        continue;


    def open_trip_file(self):
        """
        Method for opening trip tables containing OD flows of each OD pair

        """
        
        # generate ride-hailing trips and driving trips
        demand_split = split_demands(self.trip_file, self.net_name, self.demand_ratio, self.downtown_factor)
        demands_D_file, demands_RH_file =\
              demand_split.generate_demand_files(folder=f'splitted_demands_downtown_factor_{self.downtown_factor}')
        
        pickup_idle_generator = generate_pickup_idle_flows(demands_D_file, demands_RH_file, self.pickup_rate, self.idle_rate)
        demands_D_file2 = demands_D_file.replace('.tntp', '_with_pickup_idle_flows.tntp')
        pickup_idle_generator.add_flows_to_driving_demands(demands_D_file2)


        # read driving trips
        with open(demands_D_file2, 'r') as f:
            lines = f.readlines()

        current_origin = None
        for line in lines:
            line = line.rstrip()
            if line.startswith("Origin"):
                origin = str(int(line.split()[1]))
                current_origin = origin

            elif current_origin != None and len(line) < 3:
                # print "blank",line,
                current_origin = None

            elif current_origin != None:
                # to_process = line[0:-2]
                for el in line.split(";"):
                    try:
                        dest = str(int(el.split(":")[0]))
                        demand = float(el.split(":")[1])
                        self.od_vols_driving[current_origin, dest] = demand
                        self.od_vols[current_origin, dest] = demand

                    except:
                        continue
        
        # read ride hailing trips
        with open(demands_RH_file, 'r') as f:
            lines = f.readlines()

        current_origin = None
        for line in lines:
            line = line.rstrip()
            if line.startswith("Origin"):
                origin = str(int(line.split("Origin")[1]))
                current_origin = origin

            elif current_origin != None and len(line) < 3:
                # print "blank",line,
                current_origin = None

            elif current_origin != None:
                # to_process = line[0:-2]
                for el in line.split(";"):
                    try:
                        dest = str(int(el.split(":")[0]))
                        demand = float(el.split(":")[1])
                        self.od_vols_ride_hailing[current_origin, dest] = demand
                        self.od_vols[current_origin, dest] += demand

                    except:
                        continue

        origins = [str(i) for i, j in self.od_vols]
        self.origins = list(dict.fromkeys(origins).keys())

        if self.if_baseline is not None:
            if self.if_baseline == 'SO':
                self.od_vols_ride_hailing = self.od_vols
                for key in self.od_vols_driving:
                    self.od_vols_driving[key] = 0
            elif self.if_baseline == 'UE':
                self.od_vols_driving = self.od_vols
                for key in self.od_vols_ride_hailing:
                    self.od_vols_ride_hailing[key] = 0
            else:
                raise ValueError("Network argument if_baseline is not properly defined. Please define if_baseline as UE or SO.")

        print(f"trips are loaded successfully.")
        # print(f"trips are loaded successfully. \n ride-hailing demands: \
        #       {self.od_vols_ride_hailing} \n driving trips: {self.od_vols_driving}")
    

    def initialize_path_set(self):
        file_name = str(self.net_file)
        file_name = file_name.replace("net", "path_sets")
        if os.path.isfile(file_name):
            # print('path set initialization file exits and is loaded from{}'.format(file_name))
            with open(file_name, 'rb') as f:
                od_path_set_full = pickle.load(f)
            for key in self.od_vols.keys():
                k_path_set = []
                if self.path_set_size > len(od_path_set_full[key]):
                    k = len(od_path_set_full[key])
                else:
                    k = self.path_set_size
                for i in range(k):
                    k_path_set.append(od_path_set_full[key][i])
                self.od_path_set[key] = k_path_set
                # print('k_path_set:', key, k_path_set)
        elif self.if_large_net == True:
            lengths = self.cal_shortest_paths_length(weight='time')
            for key in self.od_vols.keys():
                k_paths = self.k_shortest_paths_for_large_networks(O=key[0], D=key[1], weight="time", max_paths=self.path_set_size, lengths=lengths[key[0]])
                self.od_path_set[key] = k_paths
                print("---size of the path set for OD " + str(key) + ":" + str(len(self.od_path_set[key])))
                print("the shortest path is: " + str(self.od_path_set[key][0]))
                print(key, self.od_path_set[key][0])
                print("path travel time of the shortest path is: " + str(
                    nx.path_weight(G=self.graph, path=self.od_path_set[key][0], weight="time")))

            with open(file_name, 'wb') as f:
                pickle.dump(self.od_path_set, f)
        else:
            # initialize path sets by enumerating of k_nearest_paths
            full_path_set = {}
            for key in self.od_vols.keys():
                # self.od_path_set[key] = []
                # paths = nx.all_simple_paths(G=self.graph, source=key[0], target=key[1], cutoff=None)

                # paths = list(nx.shortest_simple_paths(G=self.graph, source=key[0], target=key[1], weight="time"))
                # for path in map(nx.utils.pairwise, paths):
                #     self.od_path_set[key].append(list(path))
                k_paths, paths = self.k_shortest_paths(source=key[0], target=key[1], weight="time", k=self.path_set_size)
                self.od_path_set[key] = k_paths
                full_path_set[key] = paths
                print("---size of the path set for OD " + str(key) + ":" + str(len(self.od_path_set[key])))
                print("the shortest path is: " + str(self.od_path_set[key][0]))
                print("path travel time of the shortest path is: " + str(
                    nx.path_weight(G=self.graph, path=self.od_path_set[key][0], weight="time")))
            with open(file_name, 'wb') as f:
                pickle.dump(full_path_set, f)
        # print("OD path sets are initialized.")

    def k_shortest_paths(self, source, target, weight, k):
        k_paths = []
        paths = list(nx.shortest_simple_paths(G=self.graph, source=source, target=target, weight=weight))
        if k > len(paths):
            k = len(paths)
        for i in range(k):
            k_paths.append(paths[i])
        return k_paths, paths

    def cal_shortest_paths_length(self, weight):
        shortest_paths_length = dict()
        for key in self.od_vols.keys():
            shortest_paths_length[key[0]] = \
                    nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(self.graph, key[0], weight=weight)
        return shortest_paths_length

    def k_shortest_paths_for_large_networks(self, O, D, weight, max_paths, lengths):
        cap_length = lengths[D] * self.len_cap
        paths = []
        Q = queue.Queue()
        Q.put((D,0,[D],{D}))
        while not Q.empty():
            if len(paths) >= max_paths:
                break
            (n,l,path,visited) = Q.get()
            #print(n,l,path)
            if n == O and path[0] == D:
                if len(paths) < max_paths:
                    c_path = path.copy()
                    c_path.reverse()
                    paths.append(c_path)
            for v in [x[0] for x in list(self.graph.in_edges(n))]:
                if v in visited:
                    continue;
                e = self.graph.edges[(v,n)]
                leng = l + e[weight]
                if v in lengths: # added by Jiachao
                    if leng + lengths[v] < cap_length:  # lengths[v] is the shortest path length from O to v
                        new_path = path.copy()
                        new_path.append(v)
                        new_visited = visited.copy()
                        new_visited.add(v)
                        Q.put((v, leng, new_path, new_visited))
        return paths

    def all_or_nothing_assignment(self):
        """
        Method for implementing all-or-nothing assignment based on the current graph. \n
        It updates link traffic flow
        """
        # for edge in self.graph.edges(data=True):  # add vol to link objects
        #     edge[2]['object'].vol = 0

        shortestpath_graph = {}
        for i in self.origins:
            shortestpath_graph[i] = nx.single_source_dijkstra(self.graph, i,
                                                              weight="time")  # "weight" should be l.time?
        for (i, j) in self.od_vols:
            odvol = self.od_vols[(i, j)]
            path = shortestpath_graph[str(i)][1][str(j)]
            for p in range(len(path) - 1):
                fnode, tnode = path[p], path[p + 1]
                # self.graph[fnode][tnode]["object"].vol += odvol
                self.graph[fnode][tnode]["object"].flow += odvol

    def update_linkcost(self):
        """
        Method for updating link travel time.
        """
        for (u, v, d) in self.graph.edges(data=True):
            self.graph[u][v]["time"] = d["object"].get_time()
            self.graph[u][v]["MFC"] = d["object"].get_MFC()
            self.graph[u][v]["vc"] = d["object"].get_VC_()
            self.graph[u][v]["toll"] = d["object"].get_toll()

    # step 1 of path-based MSA
    def load_path_flow(self, path_flows_driving, path_flows_ride_hailing):
        # load path flows to network and update link costs
        # path_flows[from_node, to_node][path_idx] is the path flow.

        for (u, v, d) in self.graph.edges(data=True):  # set link flows to be 0
            self.graph[u][v]["object"].flow = 0
            self.graph[u][v]["object"].flow_driving = 0
            self.graph[u][v]["object"].flow_ride_hailing = 0

        for key in self.od_path_set.keys():
            for k in range(len(self.od_path_set[key])):
                path_flow_driving = path_flows_driving[key][k]
                path_flow_ride_hailing = path_flows_ride_hailing[key][k]
                path = self.od_path_set[key][k]
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    self.graph[from_node][to_node]["object"].flow_driving += path_flow_driving
                    self.graph[from_node][to_node]["object"].flow_ride_hailing += path_flow_ride_hailing
                    self.graph[from_node][to_node]["object"].flow += path_flow_driving
                    self.graph[from_node][to_node]["object"].flow += path_flow_ride_hailing

        self.update_linkcost()

    def get_initial_omega(self, if_random=False, random_seed=0, scale=1):
        if not if_random:
            omega = {}
            for (u, v, d) in self.graph.edges(data=True):
                omega[u, v] = 0
        else:
            np.random.seed(random_seed)
            omega = {}
            for (u, v, d) in self.graph.edges(data=True):
                omega[u, v] = np.random.uniform() * scale
        return omega

    def apply_omega(self, omega):
        for (u, v, d) in self.graph.edges(data=True):
            self.graph[u][v]["object"].omega = omega[u, v]

    # step 2 & 3 of path-based MSA, update path costs and optimal paths
    def get_auxiliary_path_flows(self, incentives=None):
        auxiliary_path_flows_driving = {}
        auxiliary_path_flows_ride_hailing = {}
        for key in self.od_path_set.keys():
            path_costs_driving = []
            path_costs_ride_hailing = []

            # calculate path costs
            for i in range(len(self.od_path_set[key])):
                path = self.od_path_set[key][i]
                path_cost_driving = nx.path_weight(G=self.graph, path=path, weight="time")
                path_cost_ride_hailing = nx.path_weight(G=self.graph, path=path, weight="MFC")
                if incentives:
                    path_cost_ride_hailing = path_cost_ride_hailing + incentives[key][i]
                path_costs_driving.append(path_cost_driving)
                path_costs_ride_hailing.append(path_cost_ride_hailing)

            # all-or-nothing assignment
            minimum_cost_driving, shortest_path_idx_driving = min((val, idx) for (idx, val)
                                                                  in enumerate(path_costs_driving))
            minimum_cost_ride_hailing, shortest_path_idx_ride_hailing = min((val, idx) for (idx, val)
                                                                            in enumerate(path_costs_ride_hailing))
            auxiliary_path_flows_driving[key] = [0] * len(self.od_path_set[key])
            auxiliary_path_flows_ride_hailing[key] = [0] * len(self.od_path_set[key])
            auxiliary_path_flows_driving[key][shortest_path_idx_driving] = self.od_vols_driving[key]
            auxiliary_path_flows_ride_hailing[key][shortest_path_idx_ride_hailing] = self.od_vols_ride_hailing[key]

            # update path costs and optimal paths
            self.path_costs_driving[key] = path_costs_driving
            self.path_costs_ride_hailing[key] = path_costs_ride_hailing
            self.optimal_paths_driving[key] = [shortest_path_idx_driving, minimum_cost_driving]
            self.optimal_paths_ride_hailing[key] = [shortest_path_idx_ride_hailing, minimum_cost_ride_hailing]
        return auxiliary_path_flows_driving, auxiliary_path_flows_ride_hailing

    # step 4 of path-based MSA
    def update_path_flows(self, path_flows_driving, path_flows_ride_hailing,
                          auxiliary_path_flows_driving, auxiliary_path_flows_ride_hailing, lbd):
        new_path_flows_driving = {}
        new_path_flows_ride_hailing = {}
        for key in path_flows_driving.keys():
            new_path_flows_driving[key] = []
            new_path_flows_ride_hailing[key] = []
            for i in range(len(path_flows_driving[key])):
                new_flow_driving = (1 - lbd) * path_flows_driving[key][i] + lbd * auxiliary_path_flows_driving[key][i]
                new_flow_ride_hailing = (1 - lbd) * path_flows_ride_hailing[key][i] + \
                                        lbd * auxiliary_path_flows_ride_hailing[key][i]
                new_path_flows_driving[key].append(new_flow_driving)
                new_path_flows_ride_hailing[key].append(new_flow_ride_hailing)
        return new_path_flows_driving, new_path_flows_ride_hailing

    # initialization of path-based MSA
    def get_initial_path_flows(self):
        intial_path_flows_driving = {}
        intial_path_flows_ride_hailing = {}
        for key in self.od_path_set.keys():
            path_costs = []
            for path in self.od_path_set[key]:
                path_cost = nx.path_weight(G=self.graph, path=path, weight="time")
                path_costs.append(path_cost)
            minimum_cost, shortest_path_idx = min((val, idx) for (idx, val) in enumerate(path_costs))
            self.optimal_paths_driving[key] = [shortest_path_idx, minimum_cost]
            intial_path_flows_driving[key] = [0] * len(self.od_path_set[key])
            intial_path_flows_ride_hailing[key] = [0] * len(self.od_path_set[key])
            intial_path_flows_driving[key][shortest_path_idx] = self.od_vols_driving[key]
            intial_path_flows_ride_hailing[key][shortest_path_idx] = self.od_vols_ride_hailing[key]
        return intial_path_flows_driving, intial_path_flows_ride_hailing

    def get_zero_incentives(self):
        incentives = {}
        for key in self.od_path_set.keys():
            incentives[key] = [0] * len(self.od_path_set[key])
        return incentives

    def update_incentives(self, n, last_incentives):
        incentives = {}
        for key in self.od_path_set.keys():
            incentives[key] = []
            for i in range(len(self.od_path_set[key])):
                incentive = (self.mu_u + self.mu_p) * \
                            (self.path_costs_driving[key][i] - self.optimal_paths_driving[key][1])
                incentive = n/(n+1) * last_incentives[key][i] + 1/(n+1) * incentive
                incentives[key].append(incentive)
        return incentives

    # calculate gap based on eq. 36 of SODTA
    def cal_gap(self, path_flows_driving, path_flows_ride_hailing):
        nu_driving = 0
        nu_ride_hailing = 0
        de_driving = 0
        de_ride_hailing = 0
        for key in self.path_costs_driving.keys():
            for i in range(len(path_flows_driving[key])):
                nu_driving += path_flows_driving[key][i] * (self.path_costs_driving[key][i] -
                                                            self.optimal_paths_driving[key][1])
                # print('nu_driving:', key, i, path_flows_driving[key][i], self.path_costs_driving[key][i], self.optimal_paths_driving[key][1])
                # print(key, i, path_flows_ride_hailing[key][i], self.path_costs_ride_hailing[key][i])
                nu_ride_hailing += path_flows_ride_hailing[key][i] * (self.path_costs_ride_hailing[key][i] -
                                                                      self.optimal_paths_ride_hailing[key][1])
                de_driving += path_flows_driving[key][i] * self.optimal_paths_driving[key][1]
                de_ride_hailing += path_flows_ride_hailing[key][i] * self.optimal_paths_ride_hailing[key][1]
        # print('nu_driving and de_driving:', nu_driving, de_driving)
        if de_driving == 0:
            gap_driving = 0
        else:
            gap_driving = nu_driving / de_driving
        if de_ride_hailing == 0:
            gap_ride_hailing = 0
        else:
            gap_ride_hailing = nu_ride_hailing / de_ride_hailing
        return gap_driving, gap_ride_hailing

    def cal_gap_d(self, path_flows_ride_hailing, incentives, last_incentives):
        gaps = []
        for key in path_flows_ride_hailing.keys():
            for i in range(len(path_flows_ride_hailing[key])):
                gap = path_flows_ride_hailing[key][i] * (incentives[key][i] - last_incentives[key][i])
                gaps.append(gap ** 2)
        return sum(gaps)
        # return sum(gaps) / len(gaps)

    def cal_Max_Loaded_unfairness(self, path_flows_ride_hailing):
        Max_LU = {}
        LU = {}
        for key in self.od_path_set:
            path_costs_driving = []
            LU[key] = []
            for i in range(len(self.od_path_set[key])):
                path = self.od_path_set[key][i]
                path_cost_driving = nx.path_weight(G=self.graph, path=path, weight="time")
                path_costs_driving.append(path_cost_driving)
            path_cost_min = min(path_costs_driving)
            if path_cost_min == 0:
                continue;
            for i in range(len(path_costs_driving)):
                # loaded_unfairness = (path_costs_driving[i]/path_cost_min)*path_flows_ride_hailing[key][i]
                loaded_unfairness = (path_costs_driving[i] / path_cost_min)
                if path_flows_ride_hailing[key][i] > 0:
                    LU[key].append(loaded_unfairness)
            if LU[key] == []:
                continue;
            Max_LU[key] = max(LU[key])
        return Max_LU

    def get_incidence_mat(self, if_idx=False):
        file_name = str(self.net_file)
        file_name_Delta = file_name.replace("net", "Delta")
        file_name_Lambda = file_name.replace("net", "Lambda")
        file_name_idx = file_name.replace("net", "idx_of_LI_path")
        file_name_idx2 = file_name.replace("net", "idx_of_LI_row")
        if os.path.isfile(file_name_idx2):
            with open(file_name_Delta, 'rb') as f:
                Delta = pickle.load(f)
            with open(file_name_Lambda, 'rb') as f:
                Lambda = pickle.load(f)
            with open(file_name_idx, 'rb') as f:
                idx = pickle.load(f)
            idx = np.array(idx)
            with open(file_name_idx2, 'rb') as f:
                idx2 = pickle.load(f)
            idx2 = np.array(idx2)
        else:
            od_list = []
            link_list = []
            path_list = []
            for key in self.od_path_set.keys():
                od_list.append(key)
                for path in self.od_path_set[key]:
                    path_list.append(path)
            for (u, v, d) in self.graph.edges(data=True):
                link_list.append((u, v))
            Lambda_D = np.zeros((len(od_list), len(path_list)))
            Delta_D = np.zeros((len(link_list), len(path_list)))
            r = 0
            c = 0
            for key in self.od_path_set.keys():
                for i in range(len(self.od_path_set[key])):
                    Lambda_D[r][c] = 1
                    c += 1
                r += 1
            c = 0
            for path in path_list:
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    r = link_list.index((from_node, to_node))
                    Delta_D[r][c] = 1
                c += 1
            m_zero = np.zeros(Lambda_D.shape)
            m1 = np.concatenate((Lambda_D, m_zero), axis=1)
            m2 = np.concatenate((m_zero, Lambda_D), axis=1)
            Lambda = np.concatenate((m1, m2), axis=0)

            m_zero = np.zeros(Delta_D.shape)
            m1 = np.concatenate((Delta_D, m_zero), axis=1)
            m2 = np.concatenate((m_zero, Delta_D), axis=1)
            Delta = np.concatenate((m1, m2), axis=0)
            with open(file_name_Delta, 'wb') as f:
                pickle.dump(Delta, f)
            with open(file_name_Lambda, 'wb') as f:
                pickle.dump(Lambda, f)

            A_eq = np.concatenate((Delta, Lambda), axis=0)

            if os.path.isfile(file_name_idx):
                with open(file_name_idx, 'rb') as f:
                    idx = pickle.load(f)
                idx = np.array(idx)
            else:
                _, idx = sympy.Matrix(A_eq).rref()
                idx = np.array(idx)
                with open(file_name_idx, 'wb') as f:
                    pickle.dump(idx, f)

            A_eq = A_eq[:, idx]
            print("rank of A_eq before calculating idx and idx2", np.linalg.matrix_rank(A_eq), A_eq.shape)

            tilde_Delta = Delta[:, idx]
            tilde_Lambda = Lambda[:, idx]
            print("shape and rank of tilde Delta:", tilde_Delta.shape, np.linalg.matrix_rank(tilde_Delta))
            print("shape and rank of tilde Lambda:", tilde_Lambda.shape, np.linalg.matrix_rank(tilde_Lambda))
            _, idx2 = sympy.Matrix(A_eq.T).rref()  # option 1
            # idx2 = get_ind_LI_row(A_eq)  # option 2
            # idx2 = get_ind_LI_row(tilde_Delta)  # option 3

            idx2 = np.array(idx2)
            print("len(idx):", idx2.shape)
            with open(file_name_idx2, 'wb') as f:
                pickle.dump(idx2, f)
            print("rank of A_eq after calculating idx and idx2", np.linalg.matrix_rank(A_eq[idx2, :]))

        if if_idx:
            return Delta, Lambda, idx, idx2
        else:
            return Delta, Lambda

    def get_incidence_mat2(self, path_flows_driving, path_flows_ride_hailing):
        file_name = str(self.net_file)
        file_name_Delta = file_name.replace("net", "Delta")
        file_name_Lambda = file_name.replace("net", "Lambda")
        if os.path.isfile(file_name_Delta):
            with open(file_name_Delta, 'rb') as f:
                Delta = pickle.load(f)
            with open(file_name_Lambda, 'rb') as f:
                Lambda = pickle.load(f)
        else:
            od_list = []
            link_list = []
            path_list = []
            for key in self.od_path_set.keys():
                od_list.append(key)
                for path in self.od_path_set[key]:
                    path_list.append(path)
            for (u, v, d) in self.graph.edges(data=True):
                link_list.append((u, v))
            Lambda_D = np.zeros((len(od_list), len(path_list)))
            Delta_D = np.zeros((len(link_list), len(path_list)))
            r = 0
            c = 0
            for key in self.od_path_set.keys():
                for i in range(len(self.od_path_set[key])):
                    Lambda_D[r][c] = 1
                    c += 1
                r += 1
            c = 0
            for path in path_list:
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    r = link_list.index((from_node, to_node))
                    Delta_D[r][c] = 1
                c += 1
            m_zero = np.zeros(Lambda_D.shape)
            m1 = np.concatenate((Lambda_D, m_zero), axis=1)
            m2 = np.concatenate((m_zero, Lambda_D), axis=1)
            Lambda = np.concatenate((m1, m2), axis=0)

            m_zero = np.zeros(Delta_D.shape)
            m1 = np.concatenate((Delta_D, m_zero), axis=1)
            m2 = np.concatenate((m_zero, Delta_D), axis=1)
            Delta = np.concatenate((m1, m2), axis=0)
            with open(file_name_Delta, 'wb') as f:
                pickle.dump(Delta, f)
            with open(file_name_Lambda, 'wb') as f:
                pickle.dump(Lambda, f)

        idx_driving = []
        idx_ride_hailing = []
        idx = 0
        for key in path_flows_driving.keys():
            for i in range(len(path_flows_driving[key])):
                if "SiouxFalls" in self.net_file:
                    # identify equilibrated path
                    if self.path_costs_driving[key][i] - self.optimal_paths_driving[key][1] < 0.5:
                        idx_driving.append(idx)
                    if self.path_costs_ride_hailing[key][i] - self.optimal_paths_ride_hailing[key][1] < 0.5:
                        idx_ride_hailing.append(idx)
                    # suggestion from Jiachao
                    # if path_flows_driving[key][i] > 0:
                    #     idx_driving.append(idx)
                    # if path_flows_ride_hailing[key][i] > 0:
                    #     idx_ride_hailing.append(idx)
                else:
                    if self.path_costs_driving[key][i] - self.optimal_paths_driving[key][1] < 0.1:
                        idx_driving.append(idx)
                    if self.path_costs_ride_hailing[key][i] - self.optimal_paths_ride_hailing[key][1] < 0.1:
                        idx_ride_hailing.append(idx)
                idx += 1
        idx_driving = np.array(idx_driving)
        idx_ride_hailing = np.array(idx_ride_hailing)
        idx_ride_hailing = idx_ride_hailing + Delta.shape[1]/2
        idx_eq_path = np.concatenate((idx_driving, idx_ride_hailing), axis=0).astype(int)
        return Delta, Lambda, idx_eq_path.flatten()

    def cal_grad_x_wrt_omega(self, Delta, Lambda, idx):
        [m1, m2] = np.vsplit(Delta, 2)
        [Delta_D, _] = np.hsplit(m1, 2)
        [_, Delta_R] = np.hsplit(m2, 2)
        [m1, m2] = np.vsplit(Lambda, 2)
        [Lambda_D, _] = np.hsplit(m1, 2)
        [_, Lambda_R] = np.hsplit(m2, 2)
        tilde_Delta = Delta[:, idx]
        tilde_Lambda = Lambda[:, idx]
        # -------test
        # test_mat = np.concatenate((tilde_Delta, tilde_Lambda), axis=0)
        # print("the shape of test_mat is {}, and its rank is{}".format(test_mat.shape, np.linalg.matrix_rank(test_mat)))
        # try:
        #     B_inv = np.linalg.inv(test_mat)
        #     print("test_mat is invertible!")
        # except:
        #     print("test_mat is non-invertible, so calculate pseudo-inverse.")
        #     B_inv = np.linalg.pinv(test_mat)
        # -------test
        idx_D = [i for i in idx if i < Delta.shape[1]/2]
        idx_R = [int(i - Delta.shape[1]/2) for i in idx if i >= Delta.shape[1]/2]
        tilde_Delta_D = Delta_D[:, idx_D]
        tilde_Delta_R = Delta_R[:, idx_R]
        tilde_Lambda_D = Lambda_D[:, idx_D]
        tilde_Lambda_R = Lambda_R[:, idx_R]
        # print("tilde matrices (Lambda, tilde_Lambda_D, tilde_Lambda_R, Delta, tilde_Delta_D, tilde_Delta_R):",
        #       Lambda,
        #       tilde_Lambda_D, tilde_Lambda_R,
        #       Delta, tilde_Delta_D, tilde_Delta_R)


        # calculate gradient of c wrt omega
        Z = np.zeros(tilde_Delta_D.T.shape)
        if self.scheme == "discriminatory subsidies":
            # print("727WRONG!!!")
            grad_c_wrt_omega = np.concatenate((Z, -tilde_Delta_R.T), axis=0)  # subsidy scheme
        elif self.scheme == "discriminatory tolls":
            grad_c_wrt_omega = np.concatenate((Z, tilde_Delta_R.T), axis=0)  # toll scheme
        elif self.scheme == "discriminatory unconstrained subsidies":
            grad_c_wrt_omega = np.concatenate((Z, -tilde_Delta_R.T), axis=0)  # costs of subsidies are not considered in the objective function
        else:
            print('ERROR when cal_x_wrt_omega! The scheme does not exist!')

        # calculate gradient of c wrt f, and then B11
        list11, list21, list22 = [], [], []
        for (u, v, d) in self.graph.edges(data=True):
            t_1 = self.graph[u][v]['object'].bpr_derivative()
            t_2 = self.graph[u][v]['object'].bpr_2nd_derivative()
            x_R = self.graph[u][v]['object'].flow_ride_hailing
            list11.append(t_1)
            list21.append(self.mu_t * (t_1 + x_R * t_2))
            list22.append(self.mu_t * (2*t_1 + x_R * t_2))
        diag11 = np.diag(list11)
        diag12 = diag11
        diag21 = np.diag(list21)
        diag22 = np.diag(list22)
        m11 = tilde_Delta_D.T @ diag11 @ tilde_Delta_D
        m12 = tilde_Delta_D.T @ diag12 @ tilde_Delta_R
        m21 = tilde_Delta_R.T @ diag21 @ tilde_Delta_D
        m22 = tilde_Delta_R.T @ diag22 @ tilde_Delta_R
        m1 = np.concatenate((m11, m12), axis=1)
        m2 = np.concatenate((m21, m22), axis=1)
        grad_c_wrt_f = np.concatenate((m1, m2), axis=0)

        A = grad_c_wrt_f
        B = -tilde_Lambda.T
        C = tilde_Lambda
        D = np.zeros((np.shape(C)[0], np.shape(C)[0]))

        # direct inversion
        m1 = np.concatenate((A, B), axis=1)
        m2 = np.concatenate((C, D), axis=1)
        B = np.concatenate((m1, m2), axis=0)
        # print("the rank of B is {}, and the shape of B is {}".format(np.linalg.matrix_rank(B), B.shape))

        # print("the rank of tilde_Lambda.T is {}, and the shape of tilde_Lambda is {}".format(np.linalg.matrix_rank(tilde_Lambda.T), tilde_Lambda.shape))
        # print(tilde_Lambda.T)
        rank_B = np.linalg.matrix_rank(B)
        if rank_B == B.shape[0]:
            B_inv = np.linalg.inv(B)
            row, col = A.shape
            B11 = B_inv[:row, :col]
            grad_x_wrt_omega = - tilde_Delta @ B11 @ grad_c_wrt_omega
        else:
            print("B is non-invertible, so cal B_reduced.")
            B_reduced, idx_LD_cols, idx_LD_rows = reduce_singular_mat(B)
            b = np.concatenate((-grad_c_wrt_omega,  np.zeros((tilde_Lambda.shape[0], grad_c_wrt_omega.shape[1]))), axis=0)
            b_reduced = np.delete(b, idx_LD_rows, axis=0)
            x_reduced = np.linalg.solve(B_reduced, b_reduced)
            idx_LI_rows = np.delete(np.arange(B.shape[0]), idx_LD_rows)
            x = np.zeros((B.shape[0], grad_c_wrt_omega.shape[1]))
            x[idx_LI_rows, :] = x_reduced
            grad_f_wrt_omega = x[:tilde_Delta.shape[1], :]
            grad_x_wrt_omega = tilde_Delta @ grad_f_wrt_omega

            grad_pi_wrt_omega = x[tilde_Delta.shape[1]:, :]
            resi_1 = grad_c_wrt_f @ grad_f_wrt_omega - tilde_Lambda.T @ grad_pi_wrt_omega + grad_c_wrt_omega
            resi_2 = tilde_Lambda @ grad_f_wrt_omega
            print("the maximum and minimum of resi_1 is {} and {}, and the average is {}".format(np.max(resi_1), np.min(resi_1), np.average(resi_1)))
            print("the maximum and minimum of resi_2 is {} and {}, and the average is {}".format(np.max(resi_2), np.min(resi_2), np.average(resi_2)))


        # --------historical version------
        # try:
        #     B_inv = np.linalg.inv(B)
        # except:
        #     print("B is non-invertible, so calculate pseudo-inverse.")
        #     B_inv = np.linalg.pinv(B, rcond=1e-25)
        #     B_inv_ = scipy.linalg.pinv(B)

        # row, col = A.shape
        # B11 = B_inv[:row, :col]

        # grad_x_wrt_omega = - tilde_Delta @ B11 @ grad_c_wrt_omega

        # # --- test
        # B21 = B_inv[row:, :col]
        # grad_f_wrt_omega = - B11 @ grad_c_wrt_omega
        # grad_pi_wrt_omega = - B21 @ grad_c_wrt_omega
        # resi_1 = grad_c_wrt_f @ grad_f_wrt_omega - tilde_Lambda.T @ grad_pi_wrt_omega + grad_c_wrt_omega
        # resi_2 = tilde_Lambda @ grad_f_wrt_omega
        # print("the maximum and minimum of resi_1 is {} and {}, and the average is {}".format(np.max(resi_1), np.min(resi_1), np.average(resi_1)))
        # print("the maximum and minimum of resi_2 is {} and {}, and the average is {}".format(np.max(resi_2), np.min(resi_2), np.average(resi_2)))
        # print("the maximum and minimum of grad_c_wrt_f is {} and {}, and the average is {}".format(np.max(grad_c_wrt_f), np.min(grad_c_wrt_f), np.average(grad_c_wrt_f)))
        # print("the maximum and minimum of grad_c_wrt_omega is {} and {}, and the average is {}".format(np.max(grad_c_wrt_omega), np.min(grad_c_wrt_omega), np.average(grad_c_wrt_omega)))
        
        # B_inv_ = scipy.linalg.pinv(B)
        # B11_ = B_inv_[:row, :col]
        # B21_ = B_inv_[row:, :col]

        # grad_f_wrt_omega_ = - B11_ @ grad_c_wrt_omega
        # grad_pi_wrt_omega_ = - B21_ @ grad_c_wrt_omega
        # resi_1_ = grad_c_wrt_f @ grad_f_wrt_omega - tilde_Lambda.T @ grad_pi_wrt_omega + grad_c_wrt_omega
        # resi_2_ = tilde_Lambda @ grad_f_wrt_omega
        # print("the maximum and minimum of resi_1_ is {} and {}, and the average is {}".format(np.max(resi_1_), np.min(resi_1_), np.average(resi_1_)))
        # print("the maximum and minimum of resi_2_ is {} and {}, and the average is {}".format(np.max(resi_2_), np.min(resi_2_), np.average(resi_2_)))  
        # --- test

        # print("B11, grad_c_wrt_omega, B11 @ grad_c_wrt_omega", B11, grad_c_wrt_omega, B11 @ grad_c_wrt_omega)
        # --------historical version------

        return grad_x_wrt_omega

    def cal_grad_z_wrt_x(self, gamma):
        grad_z_wrt_x_D = []
        grad_z_wrt_x_R = []
        for (u, v, d) in self.graph.edges(data=True):
            t = self.graph[u][v]['object'].bpr()
            t_1 = self.graph[u][v]['object'].bpr_derivative()
            x = self.graph[u][v]['object'].flow
            omega = self.graph[u][v]['object'].omega
            base = t + x * t_1
            grad_z_wrt_x_D.append(base)
            if self.scheme == "discriminatory subsidies":
                # print("800WRONG!!!")
                grad_z_wrt_x_R.append(base + gamma * omega)  # subsidy scheme
            elif self.scheme == "discriminatory tolls":
                grad_z_wrt_x_R.append(base)  # toll scheme
            elif self.scheme == "discriminatory unconstrained subsidies":
                grad_z_wrt_x_R.append(base)  # costs of subsidies are not considered in the objective function
            else:
                print('ERROR when cal_grad_z_wrt_z! The scheme does not exist!')
        grad_z_wrt_x = np.mat(np.concatenate((grad_z_wrt_x_D, grad_z_wrt_x_R)))
        grad_z_wrt_x = grad_z_wrt_x.T
        return grad_z_wrt_x

    def cal_grad_z_wrt_omega(self, gamma):
        gamma_x_R = []
        for (u, v, d) in self.graph.edges(data=True):
            gamma_x_R.append(gamma * self.graph[u][v]['object'].flow_ride_hailing)
        grad_z_wrt_omega = np.mat(gamma_x_R).T

        return grad_z_wrt_omega

    # def cal_idx2_LI_row(self, A_eq):
    #     file_name = str(self.net_file)
    #     file_name_idx2_LI_row = file_name.replace("net", "idx2_of_LI_row")
    #     if os.path.isfile(file_name_idx2_LI_row):
    #         with open(file_name_idx2_LI_row, 'rb') as f:
    #             idx2_LI_row = pickle.load(f)
    #     else:
    #         _, idx2_LI_row = sympy.Matrix(A_eq.T).rref()
    #         with open(file_name_idx2_LI_row, 'wb') as f:
    #             pickle.dump(idx2_LI_row, f)
    #     idx2_LI_row = np.array(idx2_LI_row)
    #     return idx2_LI_row

    def cal_idx2_LI_row(self, A_eq):
        file_name = str(self.net_file)
        file_name_idx2_LI_row = file_name.replace("net", "idx2_of_LI_row")
        if os.path.isfile(file_name_idx2_LI_row):
            with open(file_name_idx2_LI_row, 'rb') as f:
                idx2_LI_row = pickle.load(f)
        else:
            idx2_LD_row = []
            rows, cols = A_eq.shape
            for i in range(rows):
                mat_before = np.delete(A_eq, idx2_LD_row, axis=0)
                mat_after = np.delete(A_eq, idx2_LD_row + [i], axis=0)
                if np.linalg.matrix_rank(mat_before) == np.linalg.matrix_rank(mat_after):
                    idx2_LD_row.append(i)
            idx2_LI_row = np.arange(rows)
            idx2_LI_row = np.delete(idx2_LI_row, idx2_LD_row)
            with open(file_name_idx2_LI_row, 'wb') as f:
                pickle.dump(idx2_LI_row, f)
        idx2_LI_row = np.array(idx2_LI_row)
        return idx2_LI_row

    def cal_fleet_cost(self, path_flows_ride_hailing, incentives):
        FC = 0
        for (u, v, d) in self.graph.edges(data=True):
            FC += self.mu_t * self.graph[u][v]["object"].flow_ride_hailing * self.graph[u][v]["time"]
        for key in path_flows_ride_hailing.keys():
            for i in range(len(path_flows_ride_hailing[key])):
                FC += path_flows_ride_hailing[key][i] * incentives[key][i]
        return FC

    def check_assump_pos_x(self):
        for (u, v, d) in self.graph.edges(data=True):
            if self.graph[u][v]["object"].flow_driving == 0:
                print("find link with zero driving flow:", u, v)
            if self.graph[u][v]["object"].flow_ride_hailing == 0:
                print("find link with zero ride_hailing flow:", u, v)

    def print_VC(self):
        for (u, v, d) in self.graph.edges(data=True):
            self.graph[u][v]["object"].get_VC()

    def cal_fleet_travel_time(self):
        fleet_travel_time = 0
        for (u, v, d) in self.graph.edges(data=True):
            fleet_travel_time += self.graph[u][v]["object"].flow_ride_hailing * self.graph[u][v]["time"]
        return fleet_travel_time

    def get_path_subsidies(self):
        path_subsidies = {}

        for key in self.od_path_set.keys():
            subsidies = []
            for path in self.od_path_set[key]:
                subsidies.append(nx.path_weight(G=self.graph, path=path, weight="toll"))
            path_subsidies[key] = subsidies
        
        return path_subsidies
    
    def get_path_TT(self):
        path_TT = {}

        for key in self.od_path_set.keys():
            TT = []
            for path in self.od_path_set[key]:
                TT.append(nx.path_weight(G=self.graph, path=path, weight="time"))
            path_TT[key] = TT
        
        return path_TT

    # def cal_path_travel_time(self, path_flows_ride_hailing):
    #     for key in self.od_path_set.keys():
    #         for i in range(len(self.od_path_set[key])):
    #             path = self.od_path_set[key][i]
    #             path_travel_time = nx.path_weight(G=self.graph, path=path, weight="time")
    #             print("path, path_travel_time:", path_travel_time)
