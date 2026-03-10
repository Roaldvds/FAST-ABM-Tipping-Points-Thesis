# -*- coding: utf-8 -*-
"""

@author: thoridwagenblast
"""

from src.model import AdaptationModel
import timeit
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np
import collections

start = timeit.default_timer()

steps = 3
replications = 1
base_seed = 12345
households = 1000

test = 'simple_run'
visuals = False
if test == 'simple_run':
    print(test)
    
    adaptation_model = AdaptationModel(nr_households = households, 
                                       set_hh_adapted_at_start=0.5,
                                       # population_density = [[0,1,6,5,0],
                                       #                       [0,6,5,5,5],
                                       #                       [0,7,7,8,6],
                                       #                       [0,6,2,7,4],
                                       #                        [0,5,4,9,2]],
                                       population_density = [[0,10,60,50,0],
                                                             [0,60,50,50,50],
                                                             [0,70,70,80,60],
                                                             [0,60,20,70,40],
                                                              [0,50,40,90,20]],
                                       # population_density = [[0,50,50,50,50],
                                       #                       [0,50,50,50,50],
                                       #                       [0,50,50,50,50],
                                       #                       [0,50,50,50,50],
                                       #                       [0,50,50,50,50]],
                                       #savings_rate = 0.05, # how much each household saves of their monthly income for the flood adaptation measures
                                       # what HH "talk" about and hence get influenced in by their connections
                                       exchange_what = 'threat+coping', #'threat' or 'coping'
                                       social_norm_calculation = 'all', #'connections', # 'neighbors', 'all'
                                       weight_SN_vs_PMT = 0.4, #'empirical', # 0.4
                                       per_HH_adapted_to_SN_prob_midpoint = 'uniform', #midpoint of the s-curve (logistic function) shwoing the relation between % HH adapted and the SN probability
                                       opinion_dynamics_model = 'assimilative', # 'similarity_biased'
                                       weight_opinion_others = 0.5,
                                       threshold_similarity_bias = 0.5,
                                       prop_inc_for_damage_adaptation = 0.2,
                                       
                                       # Grid related
                                       # grid_width = 5,
                                       # grid_height = 5,
                                       flood_depth_in_m = [99, 3, 1, 1, 1], # same length as grid_height, first 99 (water)
                                       flood_time = [10], #, 50],

                                       network_structure_file_path = 'data/processed/network_structures/social_network1.graphml',
                                       
                                       # fraction_neighbors = 0.15,
                                       # fraction_homophily = 0.25,
                                       # fraction_family_and_friends = 0.55,
                                       proportion_ties_added_per_step = 0.0,
                                       proportion_ties_removed_per_step = 0.0,
                                       fraction_ties_added_at_flood = 0.1,
                                       flood_remembrance_period = 5, # how long the experience of flood leads to more connection in the network
                                       
                                       consider_measure_every_x_steps = 1, # how often people consider taking a measure
                                        minimum_intention_to_consider_measure = 0, 
                                       
                                       # flood damage related: from Huizinga, de Moel --> damage factor
                                       # in dollar and adjusted for inflation to 2020 value
                                       # max_damage_per_sqm = 1216.65,
                                       # CCA_costs = {'dry-proofing': 6000, # taken from CRAB. this is in $
                                       #              'wet-proofing': 7000},
                                       # damage_reduction = {'dry-proofing': 0.5, # taken from CRAB
                                       #              'wet-proofing': 0.4},
                                       measures_aging = {'dry-proofing': 20, # 0 if no aging, int for number of years they age
                                                    'wet-proofing': 20},
                                        collect_agent_data = 0,
                                       seed = base_seed)
    if visuals:
        """ Plot the grid """
        # Get the grid size (based on max x and y values in your grid)
        max_x = max([key[1] for key in adaptation_model.cell_properties.keys()])
        max_y = max([key[0] for key in adaptation_model.cell_properties.keys()])
        
        # Create 2D arrays for the three properties
        cell_type_grid = np.zeros((max_x + 1, max_y + 1))
        population_grid = np.zeros((max_x + 1, max_y + 1))
        flood_depth_grid = np.zeros((max_x + 1, max_y + 1))
        
        # Fill the grids with corresponding values from the dictionary
        for (x, y), props in adaptation_model.cell_properties.items():
            # For cell type (mapping 'water' to 1 and 'residential' to 2)
            if props['type'] == 'water':
                cell_type_grid[x, y] = 1
            elif props['type'] == 'residential':
                cell_type_grid[x, y] = 2
            # For population (just store the population value directly)
            population_grid[x, y] = props['population']
            # For flood depth (store the flood depth value directly)
            flood_depth_grid[x, y] = props['flood_depth']
        
        # Create the plot with 3 subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # 1. Plot for 'cell_type' with 'water' as blue and 'residential' as grey
        ax[0].imshow(cell_type_grid, cmap='coolwarm', interpolation='nearest')
        ax[0].set_title('Cell Type')
        ax[0].axis('off')
        # 2. Plot for 'population' with shades of red (normalizing population)
        population_norm = plt.Normalize(vmin=population_grid.min(), vmax=population_grid.max())
        population_sm = ax[1].imshow(population_grid, cmap='Reds', interpolation='nearest', norm=population_norm)
        ax[1].set_title('Population')
        fig.colorbar(population_sm, ax=ax[1])
        # 3. Plot for 'flood_depth' with shades of blue (normalizing flood depth)
        # Define a colormap where 99 is black, and the rest is a blue gradient
        #  Manually scale the flood depth between 0 and 3 (ignoring 99)
        flood_depth_normalized = np.copy(flood_depth_grid)
        flood_depth_normalized[flood_depth_normalized == 99] = np.nan  # We'll treat 99 separately
        
        # Create a colormap where 99 is black, and the rest is a blue gradient
        cmap = ListedColormap(['black'] + plt.cm.Blues(np.linspace(0, 1, 256)).tolist())
        
        # Create a custom norm for flood depth between 0 and 3, excluding 99
        #TODO need to fix display of flood depths. 
        norm = BoundaryNorm([0, 0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], cmap.N)
        
        # Plot flood depth with 99 in black and other values in blue shades
        flood_depth_sm = ax[2].imshow(flood_depth_normalized, cmap=cmap, interpolation='nearest', norm=norm)
    
        ax[2].set_title('Flood Depth')
        fig.colorbar(flood_depth_sm, ax=ax[2])
        # Display the plot
        plt.tight_layout()
        plt.show()
    
        
        ''' Plot of network structure'''
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(adaptation_model.social_network)  # Use a spring layout for visualization
        nx.draw(adaptation_model.social_network, pos, node_size=50, with_labels=False)
        plt.title("Network Graph with Half-Normal Degree Distribution")
        plt.show()
        
        # ''' Plot of degree distribution'''
        # plt.figure(figsize=(8, 5))
        # degree_count = collections.Counter(adaptation_model.degrees)
        # degrees, counts = zip(*degree_count.items())
        # plt.bar(degrees, counts, color = 'b')
        # plt.title("Degree Distribution")
        # plt.xlabel("Degree")
        # plt.ylabel("Frequency")
        # plt.show()
        
    setup_done = timeit.default_timer()
    for j in range(steps):
        #print('step:', j)
        adaptation_model.step()
    # Data collection
    # data = adaptation_model.datacollector.get_agent_vars_dataframe()
    data_model = adaptation_model.datacollector.get_model_vars_dataframe()
    # data.to_csv('results/agent_data_test_.csv')
    data_model.to_csv('results/model_data_test.csv')
    #print(data.head(5))

stop = timeit.default_timer()
print('Setup time:', (setup_done - start)/60, 'min')
print('Run time: ', (stop - setup_done)/60, 'min')