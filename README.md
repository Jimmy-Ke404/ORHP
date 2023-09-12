# Optimal Ride Hailing Pricing
## Introduction

## Requirements
This repo was run on `Python==3.8.8` with following packages:
- `numpy==1.20.1`
- `matplotlib==3.3.4`
- `sympy==1.8`
- `networkx==2.5`
- `geopandas==0.10.2`
- `scipy==1.6.2`

## Basic WorkFlow
To implement this repo, you need to clone the repo to your machine, and then install packages as required in the `Requirements` section.

This repo supports solving network equilibria, solving optimal pricing with equilibrium constraints, and results visualization on the Sioux Falls network and the Pittsburgh network.

### Solve network equilibria
Currently, network equilibria supported by the repo include the user equilibrium (UE), the system optimum (SO), and various mixed equilibrium (ME). You can use the following command in your terminal.

```python example_run_ME.py {network} {equilibrium_type}```

where argument `network` should be one in `['SiouxFalls', 'PGH']`, and argument `equilibrium_type` should be one in the following

- `'ME-FO'`: Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following FO.
- `'ME-SO'`: Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following SO.
- `'ME-FOSC'`: Mixed equibrium of driving vehicles following UE and ride-hailing vehicles following FOSC.
- `'Baseline-SO'`: A baseline where all vehicles follow SO.
- `'Baseline-UE'`: A baseline where all vehicles follow UE.

### Solve optimal pricing problems with equilibrium constraints
The repo is able to solve ORHP on top of ME-FOSC by using command line.

```python example_run_ORHP.py {network}```

where argument `network` should be one in `['SiouxFalls', 'PGH']`. Note this will run ORHP with different gammas that measure a trade-off between total travel time reduction and subsidy cost, and smaller gamma means larger total travel time reduction.

### Results visualization
You can use this repo to plot results of network equilibria and optimal pricing after running corresponding experiments. Specifically, `Result_visualization_SiouxFalls_network.ipynb` includes code for plotting results of the Sioux Falls network, and `Result_visualization_Pittsburgh_network.ipynb` is for the Pittsburgh network.


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
