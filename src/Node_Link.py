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

class Node:
    def __init__(self, node_id=0):
        self.node_id = node_id


class Link(object):
    def __init__(self, **kwargs):
        self.link_id = None
        self.length = 0.0
        self.capacity = 0.0
        self.t0 = 0.0
        self.alpha = 0.5
        self.beta = 4.0
        self.from_node = 0
        self.to_node = 0
        self.flow = 0.0
        self.free_speed = 1.0
        self._time = None
        self.v = 0.

        self.flow_driving = 0.0
        self.flow_ride_hailing = 0.0
        self.demand_ratio = 0.5
        self.mu_t = 1
        self.omega = None
        self.scheme = "discriminatory subsidies"
        self.slope = None
        self.FB = "FO"
        self.net_file = None
        self.if_linear_cost = False

        for k, v in kwargs.items():
            self.__dict__[k] = v
        # if "SiouxFalls_net.tntp.txt" in self.net_file:
        #     self.alpha = self.alpha * 0.5

    def get_time(self):
        """
        Method for getting link travel time based on the BPR function \n
        This method is used when setting 'time' variable

        """
        return self.bpr()
    def get_VC(self):
        VC = float(self.flow) / float(self.capacity)
        # print("V/C ratio of", self.from_node, self.to_node, ":", VC, "link travel time:", self.bpr())

    def bpr(self, alpha=None, beta=None, flow=None):
        """
        Method for calculating the BPR function

        Parameters
        ----------
        alpha:      float
                    first BPR function parameter, usually 0.15

        beta:       float
                    second BPR function parameter, usually 4.0

        flow:       float
                    flow on link

        Return
        ------
        float
            link travel time
        """
        flow = self.flow
        VC = float(flow) / float(self.capacity)
        if self.if_linear_cost:  # linear cost for SiouxFalls
            return self.t0 * (1 + self.alpha * VC)
        else:  #bpr function
            return self.t0 * (1 + self.alpha * VC ** self.beta)
            # return self.t0 * (1 + self.alpha * VC)


    def bpr_derivative(self, alpha=None, beta=None, flow=None):
        beta = self.beta
        flow = self.flow
        if self.if_linear_cost:  # linear cost for SiouxFalls
            return self.t0 * self.alpha / self.capacity
        else:
            return self.t0 * float(self.alpha) * float(beta) / float(self.capacity) * (
                        float(flow) / float(self.capacity)) ** float(beta - 1)


    def bpr_2nd_derivative(self, alpha=None, beta=None, flow=None):
        if self.if_linear_cost:
            return 0
        else:
            flow = self.flow
            try:
                return (self.t0 * self.alpha * self.beta * (self.beta-1) / self.capacity**2) * (
                    flow / self.capacity) ** (self.beta - 2)
            except:
                raise ValueError("ERROR when calculating bpr_2nd_derivative", self.link_id, self.length, self.capacity, self.t0, self.alpha)

    def get_MFC(self):
        t = self.bpr()
        t_ = self.bpr_derivative()
        if self.FB == "SO":
            return self.mu_t * (t + self.flow * t_)
        else:  # FO
            if self.omega:
                if self.scheme == "discriminatory subsidies":
                    return self.mu_t * (t + self.flow_ride_hailing * t_) - self.omega  # subsidy scheme
                elif self.scheme == "discriminatory tolls":
                    return self.mu_t * (t + self.flow_ride_hailing * t_) + self.omega  # toll scheme
                elif self.scheme == "discriminatory unconstrained subsidies":
                    return self.mu_t * (t + self.flow_ride_hailing * t_) - self.omega
                else:
                    raise ValueError('ERROR when get_MFC! The scheme does not exist!')
            else:
                return self.mu_t * (t + self.flow_ride_hailing * t_)

    def get_VC_(self):
        return round(self.flow/self.capacity, 4)

    def get_toll(self):
        if not self.omega:
            return 0
        else:
            return round(self.omega, 4)
