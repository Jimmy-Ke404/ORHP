# Optimal Ride Hailing Pricing
## Introduction

## Requirements


## Basic WorkFlow
 
## Package Structure
``` 
Data  // input data and output data
   |-- Networks
   |   |-- Pittsburgh
   |   |   |-- output  // output files
   |   |   |-- shpfiles  // shapefiles for Pittsburgh network
   |   |   |-- splitted_demands_downtown_factor_1  // driving demands and ride-hailing demands with downtown factor of 1
   |   |   |-- splitted_demands_downtown_factor_2  // driving demands and ride-hailing demands with downtown factor of 2
   |   |   |-- pitts_net_new_0716.tntp.txt  //  link data of the Pittsburgh network
   |   |   |-- pitts_path_sets_new_0716.tntp.txt  //  path sets
   |   |   |-- pitts_trips_new_0716.tntp.txt  //  total travel demands
   |   |-- SiouxFalls
   |   |   |-- output  // output files
   |   |   |-- splitted_demands_downtown_factor_1  // driving demands and ride-hailing demands with downtown factor of 1
   |   |   |-- splitted_demands_downtown_factor_2  // driving demands and ride-hailing demands with downtown factor of 2
   |   |   |-- SiouxFalls_net.tntp  //  link data of the Sioux Falls network
   |   |   |-- SiouxFalls_node.tntp  //  node coordinates of the Sioux Falls network
   |   |   |-- SiouxFalls_path_sets.tntp  //  path sets
   |   |   |-- SiouxFalls_trips (GA).tntp  //  total travel demands
src  // source code
   |-- generate_driving_ridehailing_demands.py  // generate driving and ride-hailing demands based on total demands
   |-- ME.py  //  function for solving mixed equilibria
   |-- Network.py  //  a versatile class for network analysis, including traffic assignments, mixed equilibria, and ORHP
   |-- Node_Link.py  //  a node class and a link class
   |-- ORHP.py  //  functions solving ORHP using a sensitivity analysis-based algorithm
example_run_ME.py  //  an example for solving the user equilibrium, the system optimum, and various mixed equilibria
example_run_ORHP_with_different_initial_points.py  // an example for solving ORHP with different initial points
example_run_ORHP.py  // an example for solving ORHP with different gammas (trade-off between travel time reduction and subsidy cost)
Result_visualization_Pittsburgh_network.ipynb  //  plots for the Pittsburgh network
Result_visualization_SiouxFalls_network.ipynb  //  plots for the Sioux Falls network
```