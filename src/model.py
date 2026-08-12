# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Contains the model class for the flood adaptation model.
"""
import collections
import numpy as np
import pandas as pd
import random
from mesa import Model
from mesa.space import MultiGrid
import networkx as nx
from collections import Counter
from src.agent import Households

from src.functions import get_connections_distribution_halfnormal, construct_network, rewire_for_preferential_attachment, find_homophily_indexes, gini
from mesa.datacollection import DataCollector
from src.datacollection import model_datacollector, household_datacollector

steps = 100

sociodemographics = pd.read_csv('data/processed/Sociodemographics.csv')

class AdaptationModel(Model):
    """
    Class representing the household adaptation model. 
    Models the uptake of flood adaptation measures of households under the 
    influence of social networks and opinion diffusion.
    """
    def __init__(self,
                 seed = None,
                 nr_households = 100, # number of household agents
                 set_hh_adapted_at_start = False, # HH adapted at start. if false, uses survey data. Set [0,1] fraction of adapted at start
                 prop_inc_for_damage_adaptation = 0.05, # how much each household saves of their monthly income for the flood adaptation measures
                 
                 # what HH "talk" about and hence get influenced in by their connections
                 exchange_what = 'threat+coping', #'threat' or 'coping'
                 social_norm_calculation = 'connections', # 'district', 'all'
                 per_HH_adapted_to_SN_prob_midpoint = 0.5, #midpoint of the s-curve (logistic function) shwoing the relation between % HH adapted and the SN probability
                 weight_SN_vs_PMT = 'empirical', # weight assigned to social norm probability: value between 0 and 1, 'empirical': taking survey response as basis 
                 opinion_dynamics_model = 'assimilative', # 'similarity_biased'
                 weight_opinion_others = 0.5,
                 threshold_similarity_bias = 0.3,
                 
                 # Grid related
                 grid_width = 5,
                 grid_height = 5,
                 flood_depth_in_m = [99, 1, 0.6, 0.4, 0], # same length as grid_height, first 99 (water)
                 threshold_severe_flood_exp = 2,
                 flood_time = [10, 50],
                 population_density = [[0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5]], # this has to add up to the number of households
                                       
                 
                 # network structure related
                 network_structure_file_path = 'data/processed/network_structures/social_network1.graphml',
                 
                 fraction_neighbors = 0.15,
                 fraction_homophily = 0.25,
                 fraction_family_and_friends = 0.55,
                 proportion_ties_added_per_step = 0.01,
                 proportion_ties_removed_per_step = 0.01,
                 
                 fraction_ties_added_at_flood = 0.1,
                 worry_increase_with_flood = 0.4,
                 min_worry_after_flood = 0.1, #minimum worry after experiencing a flood
                 flood_remembrance_period = 5, # how long the experience of flood leads to more connection in the network
                 
                 consider_measure_every_x_steps = 5, # how often people consider taking a measure
                 minimum_intention_to_consider_measure = 0.7, # when people are convinced enough to consider measure
                 # flood damage related: from Huizinga, de Moel --> damage factor
                 # in dollar and adjusted for inflation to 2020 value
                 max_damage_per_sqm = 950,
                 CCA_costs = {'dry-proofing': 10300, # see brainstorm document
                              'wet-proofing': 8000},
                 damage_reduction = {'dry-proofing': 0.5, # see brainstorm document
                              'wet-proofing': 0.4},
                 measures_aging = {'dry-proofing': 10, # 0 if no aging, int for number of years they age
                              'wet-proofing': 10},
                transformative_threshold = 0.8, # adaptation uptake (any) after which the adaptation can be considered transformative
                collect_agent_data = 0, # if 1, collect agent data, if 0, don't collect agent data
                savings_rate_multiplier = 1.0,
                adaptation_cost_multiplier = 1.0
                 ):
        
        super().__init__(seed = seed)
        
        # defining the variables and setting the values
        self.nr_households = nr_households
        self.set_hh_adapted_at_start = set_hh_adapted_at_start
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.flood_depth_in_m = flood_depth_in_m
        self.threshold_severe_flood_exp = threshold_severe_flood_exp
        self.flood_time = flood_time
        self.population_density = population_density
        self.savings_rate_multiplier = savings_rate_multiplier
        self.adaptation_cost_multiplier = adaptation_cost_multiplier
        if sum(sum(inner_list) for inner_list in self.population_density) != self.nr_households:
            raise ValueError('Population distirbution is', self.population_density, 
                             ' and does not match the number of household agents. nr household agents:', 
                             self.nr_households)
        
        # self.max_nr_connections = max_nr_connections
        # self.perc_until_what_nr_conn = perc_until_what_nr_conn
        # self.most_conn_below = most_conn_below
        self.fraction_neighbors = fraction_neighbors
        self.fraction_homophily = fraction_homophily
        self.fraction_family_and_friends = fraction_family_and_friends
        self.proportion_ties_added_per_step = proportion_ties_added_per_step
        self.proportion_ties_removed_per_step = proportion_ties_removed_per_step
        self.fraction_ties_added_at_flood = fraction_ties_added_at_flood
        self.worry_increase_with_flood = worry_increase_with_flood
        self.min_worry_after_flood = min_worry_after_flood
        self.flood_remembrance_period = flood_remembrance_period
        self.consider_taking_measure_every_x_steps = consider_measure_every_x_steps
        self.minimum_intention_to_consider_measure = minimum_intention_to_consider_measure
        
        self.seed = seed
        random.seed(self.seed)
        self.max_damage_per_sqm = max_damage_per_sqm
        self.CCA_costs = CCA_costs
        self.damage_reduction = damage_reduction
        self.measures_aging = measures_aging
        self.prop_inc_for_damage_adaptation = prop_inc_for_damage_adaptation
        self.exchange_what = exchange_what
        self.opinion_dynamics_model = opinion_dynamics_model
        self.weight_opinion_others = weight_opinion_others
        self.threshold_similarity_bias = threshold_similarity_bias
        self.weight_SN_vs_PMT = weight_SN_vs_PMT
        self.social_norm_calculation = social_norm_calculation
        self.per_HH_adapted_to_SN_prob_midpoint = per_HH_adapted_to_SN_prob_midpoint
        self.transformative_threshold = transformative_threshold
        self.collect_agent_data = collect_agent_data
        
        self.cumulative_damage = 0
        self.cumulative_damage_no_adaptation = 0
        self.total_damage_experienced = 0
        self.fraction_wet_proofed = 0
        self.fraction_dry_proofed = 0
        self.fraction_adapted_any = 0
        self.fraction_adapted_both = 0
        self.transformative_adaptation = 0
        '''
        Making the space
        '''
        self.make_grid()
        self.add_HHagents_to_grid()
        
        '''
        making the social network structure
        '''
        # load network structure
        self.social_network = nx.read_graphml(network_structure_file_path)
        if self.social_network.number_of_nodes() != self.nr_households:
            raise ValueError('Error: ', self.nr_households - self.social_network.number_of_nodes(), 
                             'nodes are missing in the network. Please read in a network structure where the number of nodes matches the number of household agents. Nr household agents:', self.nr_households)
        # Create a dictionary with None values for all nodes
        attributes = {node: {'household': None} for node in self.social_network.nodes}
        # Set the attributes
        nx.set_node_attributes(self.social_network, attributes)
        
        #print('sn:', self.social_network)
        self.place_HHagents_on_network()
        self.update_node_id()
        self.count_within_cell_edges()
        self.add_homophily_to_network()
        # self.count_homophily_edges()  --> # TODO check if we need to do this
        self.assign_family_and_friends_weight()
        self.get_network_descriptives()
            
        if self.collect_agent_data == 1:
            # print('collecting model and agent')
            agent_collector = household_datacollector(self.nr_households)
            model_collector = model_datacollector()
            self.datacollector = DataCollector(agent_reporters = agent_collector,
                                           model_reporters = model_collector)
        else:
            # print('collecting model data')
            model_collector = model_datacollector()
            self.datacollector = DataCollector(model_reporters = model_collector)

        


    def make_grid(self):
        # create a spatial grid
        self.grid = MultiGrid(width = self.grid_width, height = self.grid_height, # dimensions of the grid
                              torus = False # no connecting of the edges
                              )
        # Dictionary to store cell properties
        self.cell_properties = {}

        # Add properties to grid
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                # Define cell type based on its coordinates
                cell_type = "water" if y == 0 else "residential" # have one waterfront at y = 0
                cell_flood_depth = self.flood_depth_in_m[y]
                
                if cell_type == 'residential':
                    cell_population = self.population_density[x][y]
                else:
                    cell_population = 0
                # Store the property in the dictionary
                self.cell_properties[(x, y)] = {'type': cell_type,
                                                'population': cell_population,
                                                'flood_depth': cell_flood_depth}
        #print(self.cell_properties)
        
    def add_HHagents_to_grid(self):
        # Check whether the desnity adds up to the number of households
        population_sum = 0
        # Iterate through the outer dictionary
        for key, inner_dict in  self.cell_properties.items():
            # Add the 'density' value from the inner dictionary to the total
            population_sum += inner_dict['population']
        if population_sum != self.nr_households:
            raise ValueError('Error: ', self.nr_households - population_sum, 
                             'households not considered in density. Density sum: ', population_sum, 
                             'Nr agents:', self.nr_households)
        
        # place households on cells
        # no need to check for residential again because for everything non-residential, the density is 0
        for cell_id, cell_props in self.cell_properties.items():
            (x, y) = cell_id
            cell_pop = cell_props['population']
            # Add 'density' number of households to the cell
            for _ in range(cell_pop):
                household = Households(self)
                self.grid.place_agent(household, (x, y))
                
        # keeping dictionary of which households are in which cell
        self.cell_households = {}
        for (content, (x, y)) in self.grid.coord_iter():
            household = [obj.unique_id for obj in content if isinstance(obj, Households)]
            self.cell_households[(x,y)] = household
        #print(self.cell_households)
        
    def place_HHagents_on_network(self):
        """
        Placing the household agents on the network while making sure that fraction_neighbors 
        amount of connections are within the same grid cell.
        Process:
            1. Get the edges in the network.
            2. select fraction_neighbors edges from it randomly
            3. Place two households on the nodes with the same cell value
            4. Assign households to the rest of the nodes randomly.
        """
        random.seed(self.seed)
        
        self.households_without_node = []
        for agent in self.agents:
            self.households_without_node.append(agent.unique_id)
        # print(len(self.households_without_node))
        
        # Calculate the number of within-cell edges
        num_within_cell_edges = int(self.fraction_neighbors * self.social_network.number_of_edges())
        #print('Edges needded within cell ie neighbors: ', num_within_cell_edges)
        edges = list(self.social_network.edges())
        neighbor_edges = random.sample(edges, num_within_cell_edges)       
        # print(neighbor_edges)
        
        cells_with_min_2_households = {cell: agents for cell, agents in self.cell_households.items() if len(agents) >= 2}
        if not cells_with_min_2_households:
            print('Cannot make neighbor connection because there are no households that share a cell')
        #print(cells_with_min_2_households)
        
        for edge in neighbor_edges: 
            # print('edge', edge)
            if (self.social_network.nodes[edge[0]].get("household") is None) and (self.social_network.nodes[edge[1]].get("household") is None):
                #print('no households placed on any of these nodes')
                # select two agents randomly that are in households_without_node and share the same location. 
                random.seed(self.seed + int(edge[0]))
                selected_cell = random.choice(list(cells_with_min_2_households.keys()))# select a random cell
                households_in_cell = self.cell_households[selected_cell] # get the households in that cell
                # check which households do not have a node assigned within that cell
                eligible_households = [household for household in households_in_cell if household in self.households_without_node] 
                # if there are more than 2 households, select 2 randomly from the list
                if len(eligible_households) >= 2:
                    selected_households = random.sample(eligible_households,2)
                    # add the households to the nodes
                    self.social_network.nodes[edge[0]]['household'] = selected_households[0]
                    self.social_network.nodes[edge[1]]['household'] = selected_households[1]
                    # remove the household from the available nodes
                    self.households_without_node.remove(selected_households[0])
                    self.households_without_node.remove(selected_households[1])
                    #print(len(self.households_without_node))
                else:
                    print('Not enough households left in this cell, skip')
            
            # One of the households is already assigned to a node
            if (self.social_network.nodes[edge[0]].get("household") is None) and (self.social_network.nodes[edge[1]].get("household") is not None):
                # print('household 2 set, need to place housheold 1 from that edge')
                set_household = self.social_network.nodes[edge[1]].get("household")
                #print('id of household that has node;', set_household)
                # get the position
                for (x,y), agent_list in self.cell_households.items():
                    if set_household in agent_list:
                        cell = (x,y)
                #print('set household position', cell)
                households_in_cell = self.cell_households[cell]
                # check which households do not have a ndoe assigned within that cell
                eligible_households = [household for household in households_in_cell if household in self.households_without_node] 
                if len(eligible_households) >= 1:
                    selected_household = random.sample(eligible_households,1)[0]
                    self.social_network.nodes[edge[0]]['household'] = selected_household
                    self.households_without_node.remove(selected_household)
                    # print(len(self.households_without_node))
                else:
                    print('No households left available, skip')
            
            if (self.social_network.nodes[edge[0]].get("household") is not None) and (self.social_network.nodes[edge[1]].get("household") is  None):
                #print('shousehold 1 set, need to place housheold 2 from that edge')
                set_household = self.social_network.nodes[edge[0]].get("household")
                # print('id of household that has node;', set_household)
                # get the position
                for (x,y), agent_list in self.cell_households.items():
                    if set_household in agent_list:
                        cell = (x,y)
                # print('set household position', cell)
                households_in_cell = self.cell_households[cell]
                # check which households do not have a ndoe assigned within that cell
                eligible_households = [household for household in households_in_cell if household in self.households_without_node] 
                if len(eligible_households) >= 1:
                    selected_household = random.sample(eligible_households,1)[0]
                    self.social_network.nodes[edge[1]]['household'] = selected_household
                    self.households_without_node.remove(selected_household)
                    # print(len(self.households_without_node))
                else:
                    print('No households left available, skip')
            else:
                continue
        # assign rest of households households_without_node to rest of nodes
        for node in self.social_network.nodes:
            if self.social_network.nodes[node].get("household") is None: 
                # assign household to node
                selected_household = random.choice(self.households_without_node)
                self.social_network.nodes[node]['household'] = selected_household
                self.households_without_node.remove(selected_household)
                # print(len(self.households_without_node))
    
    def update_node_id(self):
        """ 
        Update node id of the network so it's the same as the household id
        Create a mapping of the current node IDs to the 'household' values
        
        Returns
        -------
        self.social_network: Social network with relabelled values
        """
        node_relabel_map = {}
        for node in list(self.social_network.nodes()):
            household_value = self.social_network.nodes[node].get('household')
            if household_value is not None:
                # Remove the node with the old ID and add it with the new ID (household value)
                node_relabel_map[node] = household_value       
        self.social_network = nx.relabel_nodes(self.social_network, node_relabel_map)
        self.edges = list(self.social_network.edges())

        #print(self.social_network)
    def add_homophily_to_network(self):
        """

        Returns
        -------
        self.social_network: Social network with assigned income, age, and education level

        """
        random.seed(self.seed)
        
        self.households_without_sociodemographics = []
        for agent in self.agents:
            self.households_without_sociodemographics.append(agent.unique_id)
        
        # Calculate the number of homophily edges 
        num_homophily_edges = int(self.fraction_homophily * self.social_network.number_of_edges())
        #print('Edges needded with homophily: ', num_homophily_edges)
    
        homophily_edges = random.sample(self.edges, num_homophily_edges)       
        # print(homophily_edges)
        # check how many unique values ie how many hh ids are in there
        flattened_list = [item for tuple in homophily_edges for item in tuple]
        unique_values = set(flattened_list)
        # print('unique values in homophily edges', num_unique_values)
        
        # potential pairs that are homophilic in terms of age, income, education from the sociodemographics data
        potential_pairs = find_homophily_indexes(sociodemographics)
        

        for edge in homophily_edges: 
            #print(edge)
            random.seed(self.seed + int(edge[0]))
            if (edge[0] in self.households_without_sociodemographics) and (edge[1] in self.households_without_sociodemographics):
                #print('both nodes not assigned')
                selected_pair = random.choice(potential_pairs)
                #print(selected_pair)
                #print(sociodemographics.loc[list(selected_pair)])
                for agent in self.agents:
                    if agent.unique_id == edge[0]:
                        agent.income_category = sociodemographics.loc[selected_pair[0], 'income_category']
                        agent.income = sociodemographics.loc[selected_pair[0], 'annual_income']
                        agent.age = sociodemographics.loc[selected_pair[0], 'age']
                        agent.savings = sociodemographics.loc[selected_pair[0], 'savings']
                        agent.level_of_education = sociodemographics.loc[selected_pair[0], 'Level_Education']
                        agent.house_size = sociodemographics.loc[selected_pair[1], 'house_size'] 
                        agent.sociodemographics_index = selected_pair[0]
                        # print(agent.income, agent.age, agent.savings)
                    if agent.unique_id == edge[1]:
                        agent.income_category = sociodemographics.loc[selected_pair[1], 'income_category']
                        agent.income = sociodemographics.loc[selected_pair[1], 'annual_income']
                        agent.age = sociodemographics.loc[selected_pair[1], 'age']
                        agent.savings = sociodemographics.loc[selected_pair[1], 'savings']
                        agent.level_of_education = sociodemographics.loc[selected_pair[1], 'Level_Education']
                        agent.house_size = sociodemographics.loc[selected_pair[1], 'house_size'] 
                        agent.sociodemographics_index = selected_pair[0]
                # remove both households from the households without sociodemopgraphics list
                self.households_without_sociodemographics.remove(edge[0])
                self.households_without_sociodemographics.remove(edge[1])
            if (edge[0] in self.households_without_sociodemographics) and (edge[1] not in self.households_without_sociodemographics):
                #print('Edge[0] assigned, edge[1] not')
                # find index from sociodemogrpahic data for the HH agent that already has sociodemographic data
                for agent in self.agents:
                    if agent.unique_id == edge[1]:
                        given_index = agent.sociodemographics_index
                        #print(given_index)
                # Find all pairs where the given index is present
                matching_pairs = [pair for pair in potential_pairs if given_index in pair]
                #print(matching_pairs)
                # select random pair from the matching pairs and get the other index from it
                random_matching_pair = random.choice(matching_pairs)
                other_index = random_matching_pair[0] if random_matching_pair[1] == given_index else random_matching_pair[1]
                #print(random_matching_pair, other_index)
                # assign the sociodemographics for the other HH agent
                for agent in self.agents:
                    if agent.unique_id == edge[0]:
                        agent.income_category = sociodemographics.loc[other_index, 'income_category']
                        agent.income = sociodemographics.loc[other_index, 'annual_income']
                        agent.age = sociodemographics.loc[other_index, 'age']
                        agent.savings = sociodemographics.loc[other_index, 'savings']
                        agent.level_of_education = sociodemographics.loc[other_index, 'Level_Education']
                        agent.house_size = sociodemographics.loc[selected_pair[1], 'house_size'] 
                        agent.sociodemographics_index = other_index
                        #print(agent.income, agent.age, agent.savings)
                # remove household1 from the households without sociodemopgraphics list
                self.households_without_sociodemographics.remove(edge[0])
            if (edge[0] not in self.households_without_sociodemographics) and (edge[1] in self.households_without_sociodemographics):
                #print('Edge[1] assigned, edge[0] not')
                # find index from sociodemogrpahic data for the HH agent that already has sociodemographic data
                for agent in self.agents:
                    if agent.unique_id == edge[0]:
                        given_index = agent.sociodemographics_index
                        #print(given_index)
                # Find all pairs where the given index is present
                matching_pairs = [pair for pair in potential_pairs if given_index in pair]
                #print(matching_pairs)
                # select random pair from the matching pairs and get the other index from it
                random_matching_pair = random.choice(matching_pairs)
                other_index = random_matching_pair[0] if random_matching_pair[1] == given_index else random_matching_pair[1]
                #print(random_matching_pair, other_index)
                # assign the sociodemographics for the other HH agent
                for agent in self.agents:
                    if agent.unique_id == edge[1]:
                        agent.income_category = sociodemographics.loc[other_index, 'income_category']
                        agent.income = sociodemographics.loc[other_index, 'annual_income']
                        agent.age = sociodemographics.loc[other_index, 'age']
                        agent.savings = sociodemographics.loc[other_index, 'savings']
                        agent.level_of_education = sociodemographics.loc[other_index, 'Level_Education']
                        agent.house_size = sociodemographics.loc[selected_pair[1], 'house_size'] 
                        agent.sociodemographics_index = other_index
                        #print(agent.income, agent.age, agent.savings)
                # remove household2 from the households without sociodemopgraphics list
                self.households_without_sociodemographics.remove(edge[1])
            if (edge[0] not in self.households_without_sociodemographics) and (edge[1] not in self.households_without_sociodemographics):
                continue
                #print('both nodes assigned')
        #print('len hh without sociodemographics', len(self.households_without_sociodemographics))
        
        # loop through the HH agents without sociodemographics
        agent_without_homophily = 0
        for agent in self.agents:
            for HH_without_sociodemographics in self.households_without_sociodemographics:
                if HH_without_sociodemographics == agent.unique_id:
                    agent_without_homophily += 1
                    # pick random line from the sociodemographics data
                    random.seed(self.seed + int(agent.unique_id))
                    index = random.randrange(len(sociodemographics))
                    agent.income_category = sociodemographics.loc[index, 'income_category']
                    agent.income = sociodemographics.loc[index, 'annual_income']
                    agent.age = sociodemographics.loc[index, 'age']
                    agent.savings = sociodemographics.loc[index, 'savings']
                    agent.level_of_education = sociodemographics.loc[index, 'Level_Education']
                    agent.house_size = sociodemographics.loc[selected_pair[1], 'house_size'] 
                    agent.sociodemographics_index = index
                    # remove household from the households without sociodemopgraphics list
                    self.households_without_sociodemographics.remove(agent.unique_id)
        #print('len hh without sociodemographics', len(self.households_without_sociodemographics))
        #print('all edges, homophily edges', self.social_network.number_of_edges(), num_homophily_edges)
        #print(num_homophily_edges/self.social_network.number_of_edges())
                    
    def assign_family_and_friends_weight(self):
        random.seed(self.seed)
        num_family_and_friends_edges = int(self.fraction_family_and_friends * self.social_network.number_of_edges())
        family_and_friends_edges = random.sample(self.edges, num_family_and_friends_edges)
        
        for u,v in self.edges:
            if (u,v) in family_and_friends_edges or (v,u) in family_and_friends_edges:
                # more weight for all the family_and_friends edges
                self.social_network[u][v]['weight'] = 2
            else:
                self.social_network[u][v]['weight'] = 1
        
        # for u, v, weight in self.social_network.edges(data='weight'):
        #     print(f"Edge ({u}, {v}) has weight {weight}")
        
            
    def count_within_cell_edges(self):
        """ 
        Function to count how many edges are within a district ie cell
        """ 
        self.district_edges = []
        for edge in self.edges:
            for (x,y), agent_list in self.cell_households.items():
                if (self.social_network.nodes[edge[0]].get("household") in agent_list) and (self.social_network.nodes[edge[1]].get("household") in agent_list):
                    self.district_edges.append(edge)
                else:
                    continue
        #print('all edges, district_edges edges', len(self.edges), len(self.district_edges))
        self.proportion_district_edges = len(self.district_edges)/len(self.edges)
        #print(self.proportion_district_edges)
        
    
    def get_network_descriptives(self):
        # avg degree = 2 nr_edges/nr_nodes --> collect nr_edges and calculate this afterwards
        # Clustering coefficient
        self.avg_clustering_nw = round(nx.average_clustering(self.social_network), 5)
        
        # Connected components
        is_directed = nx.is_directed(self.social_network)
        connected_components = list(nx.connected_components(self.social_network)) if not is_directed else list(nx.strongly_connected_components(self.social_network))

        # Diameter (for each connected component)
        diameters = []
        for component in connected_components:
            subgraph = self.social_network.subgraph(component)
            if nx.is_connected(subgraph):  # Ensure the subgraph is connected
                diameters.append(nx.diameter(subgraph))
        self.diameters_nw_counter = collections.Counter(diameters)
        # sum over counter of diameters of connected components gives nr of connected components

        # Density
        self.density_nw = round(nx.density(self.social_network), 5)
        
        # Centrality measures (example: degree centrality), potentially collect later
        # degree_centrality = nx.degree_centrality(self.social_network)
        # betweenness_centrality = nx.betweenness_centrality(self.social_network)


    
    def remove_network_ties(self, proportion_edges_to_remove):
        # function to remove some edges
        #print('nr_edges_to_remove:', int(round(proportion_edges_to_remove * self.nr_households, 0)))
        #print(self.edges)
        nr_edges_to_remove = int(round(proportion_edges_to_remove * self.nr_households, 0))
        if self.social_network.number_of_edges() >= nr_edges_to_remove:
            random.seed(self.seed)
            edges_to_remove = random.sample(self.edges, nr_edges_to_remove)
        else: 
            edges_to_remove = self.edges
        #print(edges_to_remove)
        for edge in edges_to_remove:
            (u,v) = edge
            self.social_network.remove_edge(u,v)    
            self.edges.remove(edge)
        #print('len edges', len(self.edges))
    
    def get_proportion_new_ties(self):
        # calculate the proportion of new ties based on whethere a flood shock just happened or not
        for flood_time in self.flood_time:
            if (self.steps >= flood_time) and (self.steps <= flood_time + self.flood_remembrance_period):
                self.proportion_added_ties_flood_dependent = round(self.fraction_ties_added_at_flood/(self.steps - flood_time + 1), 2)
                return self.proportion_added_ties_flood_dependent
        return self.proportion_ties_added_per_step
    
    def add_network_ties(self, proportion_new_edges):
        # function to add some ties
        #print(proportion_new_edges)
        if nx.is_connected(self.social_network): # check if already fully connected network
            pass
        else:
            # select households that aim to get a new connection randomly
            random.seed(self.seed)
            nr_hh_seeking_new_connection = int(round(proportion_new_edges*self.nr_households, 0))
            hh_seeking_new_connection = random.sample(list(self.social_network.nodes()), nr_hh_seeking_new_connection)
            #print(hh_seeking_new_connection)
            
            for hh in hh_seeking_new_connection:
                #print(hh)
                # get the list of potential connections for that household agent
                current_ties = set(self.social_network.neighbors(hh))
                #print(current_ties)
                possible_connections = set(self.social_network.nodes()) - current_ties - {hh}
                #print('possible connections', possible_connections)
                
                # split the connections: 
                neighbor_group = [] # connections within the same cell
                homophily_group = [] # connections fulfilling homophily
                other_group = [] # rest
                
                # looping through the agents and connecting agents toa dd them to the right group
                for agent in self.agents:
                    if agent.unique_id == hh:
                        for potential_connection in self.agents:
                            if potential_connection.unique_id in possible_connections:
                                #print('hh', agent, agent.pos)
                                #print('connection', potential_connection, potential_connection.pos)
                                if potential_connection.pos == agent.pos:
                                    neighbor_group.append(potential_connection.unique_id)
                                if (potential_connection.income_category == agent.income_category) and (potential_connection.level_of_education == agent.level_of_education) and (abs(potential_connection.age - agent.age) < 10):
                                    homophily_group.append(potential_connection.unique_id)
                                if (potential_connection.unique_id not in neighbor_group) and (potential_connection.unique_id not in homophily_group):
                                    other_group.append(potential_connection.unique_id)
                #print('neighbor_group', neighbor_group)
                #print('homophily_group', homophily_group)
                #print('other_group', other_group)
                
                # generate random number and in 15% of cases, select preferably from neighbors,
                # in 25% of case from homophily subgroup and else from the rest
                random.seed(self.seed + hh)
                random_group_selection = random.uniform(0,1)
                #print(random_group_selection)
                if (len(neighbor_group) > 0) and (random_group_selection < self.fraction_neighbors):
                    selected_connection = random.sample(neighbor_group, 1)
                    #print('neighbor selected')
                elif (len(homophily_group) > 0) and (self.fraction_neighbors <= random_group_selection < (self.fraction_homophily + self.fraction_neighbors)):
                    selected_connection = random.sample(homophily_group, 1)
                    #print('homophily selected')
                else:
                    selected_connection = random.sample(other_group, 1)
                #print(selected_connection)
                
                self.social_network.add_edge(hh, selected_connection[0])
                
                random.seed(self.seed + hh + hh)
                random_family_friends = random.uniform(0,1)
                if random_family_friends < self.fraction_family_and_friends:
                    self.social_network[hh][selected_connection[0]]['weight'] = 2
                else:
                    self.social_network[hh][selected_connection[0]]['weight'] = 1
                    
                #print(self.social_network[hh][selected_connection[0]]['weight'])

        
    def update_model_variables(self):
        
        self.get_network_descriptives()

        cumulative_damage = 0
        cumulative_damage_no_adaptation = 0
        total_damage_experienced = 0
        HH_adapted_wp = 0
        HH_adapted_dp = 0
        HH_adapted_any = 0
        HH_adapted_both = 0
        
        sum_intention_SN = 0
        sum_intention_PMT_DP = 0
        sum_intention_PMT_WP = 0
        sum_intention_overall_DP = 0
        sum_intention_overall_WP = 0

        sum_worry = 0
        sum_perceived_risk = 0
        sum_savings = 0
        sum_flood_damage = 0

        # Additional state-space variables
        households_with_expiry = 0

        active_measure_ages = []
        active_measure_remaining_lifetimes = []

        sum_fraction_connections_adapted = 0
        agents_with_connections = 0

        HH_severe_flood_exp = 0
        HH_severe_adapted_any = 0

        liquidity_constrained = 0
        unmet_adaptation_demand = 0

        # Including initial adaptation value
        relative_burdens = []
        relative_burdens_cumulative_income = []

        # Simulation period only
        relative_burdens_simulation_only = []
        relative_burdens_cumulative_income_simulation_only = []

        low = [a for a in self.agents if a.income_category in [1,2]]
        mid = [a for a in self.agents if a.income_category == 3]
        high = [a for a in self.agents if a.income_category in [4,5]]

        def adapted_rate(group):
            if len(group) == 0:
                return 0
            return sum(1 for a in group if 1 in a.measures_taken.values()) / len(group)

        self.adaptation_rate_low = adapted_rate(low)
        self.adaptation_rate_middle = adapted_rate(mid)
        self.adaptation_rate_high = adapted_rate(high)
        self.adaptation_gap_high_low = self.adaptation_rate_high - self.adaptation_rate_low

        agents_by_id = {
            agent.unique_id: agent
            for agent in self.agents
        }

        for agent in self.agents:

            agent.calculate_optimal_adaptation()

            actually_adapted = 1 in agent.measures_taken.values()

            agent.adaptation_deficit = (
                    agent.optimal_to_adapt and not actually_adapted
            )

            agent.over_adapted = (
                    (not agent.optimal_to_adapt) and actually_adapted
            )

            min_cost_remaining = min(
                self.CCA_costs[m] * self.adaptation_cost_multiplier
                for m in self.CCA_costs
                if agent.measures_taken[m] == 0
            ) if any(agent.measures_taken[m] == 0 for m in self.CCA_costs) else 0

            # damage
            cumulative_damage += round(agent.flood_damage, 0)
            cumulative_damage_no_adaptation += round(agent.flood_damage_no_measures, 0)
            total_damage_experienced += agent.damage_experienced

            if agent.measure_expired_this_step:
                households_with_expiry += 1


            if agent.nr_connected_HHagents > 0:
                adapted_connections = sum(
                    1
                    for connection_id in agent.connected_HHagents
                    if 1 in agents_by_id[
                        connection_id
                    ].measures_taken.values()
                )

                fraction_connections_adapted = (
                        adapted_connections
                        / agent.nr_connected_HHagents
                )

                sum_fraction_connections_adapted += (
                    fraction_connections_adapted
                )

                agents_with_connections += 1



            for measure, taken in agent.measures_taken.items():
                if taken == 1:
                    lifetime = self.measures_aging[measure]

                    if lifetime > 0:
                        age = agent.age_of_measures[measure]

                        active_measure_ages.append(age)

                        active_measure_remaining_lifetimes.append(
                            max(lifetime - age, 0)
                        )



            # adaptation intention
            sum_intention_SN += agent.prob_from_social_norm
            sum_intention_PMT_DP += agent.protection_motivation['dry-proofing']
            sum_intention_PMT_WP += agent.protection_motivation['wet-proofing']
            sum_intention_overall_DP += agent.probability_to_take_measure['dry-proofing']
            sum_intention_overall_WP += agent.probability_to_take_measure['wet-proofing']

            # Additional agent vars
            sum_worry += agent.worry
            sum_perceived_risk += agent.perceived_risk
            sum_savings += agent.savings
            sum_flood_damage += agent.flood_damage

            # adaptation
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 1):
                HH_adapted_dp += 1
                HH_adapted_wp += 1
                HH_adapted_any += 1
                HH_adapted_both += 1
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 0):
                HH_adapted_dp += 1
                HH_adapted_any += 1
            if (agent.measures_taken['dry-proofing'] == 0) and (agent.measures_taken['wet-proofing'] == 1):
                HH_adapted_wp += 1
                HH_adapted_any += 1

            if agent.flood_depth > self.threshold_severe_flood_exp:
                HH_severe_flood_exp += 1
                if (agent.measures_taken['dry-proofing'] == 1) or (agent.measures_taken['wet-proofing'] == 1):
                    HH_severe_adapted_any += 1

            if min_cost_remaining > 0 and agent.savings < min_cost_remaining:
                liquidity_constrained += 1

            if agent.could_not_afford > 0:
                unmet_adaptation_demand += 1

            annual_income = max(agent.income, 1)

            # Costs during simulation only
            simulation_costs = (
                    agent.damage_experienced
                    + agent.adaptation_cost_paid
            )

            # Costs including the remaining value of measures already present at t=0
            total_costs = (
                    simulation_costs
                    + agent.initial_adaptation_value
            )

            years_elapsed = self.steps + 1
            cumulative_income = annual_income * years_elapsed

            # Annual-income burden
            relative_burdens.append(total_costs / annual_income)
            relative_burdens_simulation_only.append(simulation_costs / annual_income)

            # Cumulative-income burden
            relative_burdens_cumulative_income.append(
                total_costs / cumulative_income
            )

            relative_burdens_cumulative_income_simulation_only.append(
                simulation_costs / cumulative_income
            )

        # Experienced flood damage
        self.total_damage_experienced = total_damage_experienced

        self.average_damage_experienced = (
                total_damage_experienced / self.nr_households
        )

        # Expired adaptation
        self.share_households_with_expiry = (
                households_with_expiry / self.nr_households
        )

        # Age of currently active adaptation measures
        self.average_active_measure_age = (
            np.mean(active_measure_ages)
            if active_measure_ages
            else 0
        )

        self.average_remaining_measure_lifetime = (
            np.mean(active_measure_remaining_lifetimes)
            if active_measure_remaining_lifetimes
            else 0
        )

        # Network exposure to adapted households
        self.average_fraction_connections_adapted = (
            sum_fraction_connections_adapted
            / agents_with_connections
            if agents_with_connections > 0
            else 0
        )

        self.share_optimal_to_adapt = (
                sum(a.optimal_to_adapt for a in self.agents)
                / self.nr_households
        )

        self.share_adaptation_deficit = (
                sum(a.adaptation_deficit for a in self.agents)
                / self.nr_households
        )

        self.share_over_adapted = (
                sum(a.over_adapted for a in self.agents)
                / self.nr_households
        )

        self.cumulative_damage = cumulative_damage
        self.cumulative_damage_no_adaptation = cumulative_damage_no_adaptation
        self.fraction_wet_proofed = HH_adapted_wp/self.nr_households
        self.fraction_dry_proofed = HH_adapted_dp/self.nr_households
        self.fraction_adapted_any = HH_adapted_any/self.nr_households
        self.fraction_adapted_both = HH_adapted_both/self.nr_households
        
        self.HH_severe_fl_exp = HH_severe_flood_exp
        # moderately affected = self.nr_households - self.HH_severe_fl_exp
        self.HH_severe_fl_adapted_any = HH_severe_adapted_any
        # moderately affected adapted = HH_adapted - HH_severe_adapted 
        # = self.fraction_adapted_any * self.nr_households - HH_severe_adapted

        self.avg_intention_SN = round(sum_intention_SN/self.nr_households, 4)
        self.avg_intention_PMT_DP = round(sum_intention_PMT_DP/self.nr_households, 4)
        self.avg_intention_PMT_WP = round(sum_intention_PMT_WP/self.nr_households, 4)
        self.avg_intention_overall_DP = round(sum_intention_overall_DP/self.nr_households, 4)
        self.avg_intention_overall_WP = round(sum_intention_overall_WP/self.nr_households, 4)

        self.average_worry = sum_worry / self.nr_households
        self.average_perceived_risk = sum_perceived_risk / self.nr_households
        self.average_savings = sum_savings / self.nr_households
        self.average_flood_damage = sum_flood_damage / self.nr_households

        # check if adaptation transformative
        if self.fraction_adapted_any >= self.transformative_threshold:
            self.transformative_adaptation = 1
        else: 
            self.transformative_adaptation = 0
        # print(self.fraction_adapted_any, self.transformative_adaptation)

        self.edges = list(self.social_network.edges())

        # district-level adaptatation
        agents_in_11 = self.grid.get_cell_list_contents((1, 1))
        self.HH_adapted_any_11 = 0
        self.HH_adapted_dp_11 = 0
        self.HH_adapted_wp_11 = 0
        self.HH_adapted_both_11 = 0
        for agent in agents_in_11:
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_dp_11 += 1
                self.HH_adapted_wp_11 += 1
                self.HH_adapted_any_11 += 1
                self.HH_adapted_both_11 += 1
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 0):
                self.HH_adapted_dp_11 += 1
                self.HH_adapted_any_11 += 1
            if (agent.measures_taken['dry-proofing'] == 0) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_wp_11 += 1
                self.HH_adapted_any_11 += 1

        agents_in_12 = self.grid.get_cell_list_contents((1, 2))
        self.HH_adapted_any_12 = 0
        self.HH_adapted_dp_12 = 0
        self.HH_adapted_wp_12 = 0
        self.HH_adapted_both_12 = 0
        for agent in agents_in_12:
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_dp_12 += 1
                self.HH_adapted_wp_12 += 1
                self.HH_adapted_any_12 += 1
                self.HH_adapted_both_12 += 1
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 0):
                self.HH_adapted_dp_12 += 1
                self.HH_adapted_any_12 += 1
            if (agent.measures_taken['dry-proofing'] == 0) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_wp_12 += 1
                self.HH_adapted_any_12 += 1

        agents_in_21 = self.grid.get_cell_list_contents((2, 1))
        self.HH_adapted_any_21 = 0
        self.HH_adapted_dp_21 = 0
        self.HH_adapted_wp_21 = 0
        self.HH_adapted_both_21 = 0
        for agent in agents_in_21:
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_dp_21 += 1
                self.HH_adapted_wp_21 += 1
                self.HH_adapted_any_21 += 1
                self.HH_adapted_both_21 += 1
            if (agent.measures_taken['dry-proofing'] == 1) and (agent.measures_taken['wet-proofing'] == 0):
                self.HH_adapted_dp_21 += 1
                self.HH_adapted_any_21 += 1
            if (agent.measures_taken['dry-proofing'] == 0) and (agent.measures_taken['wet-proofing'] == 1):
                self.HH_adapted_wp_21 += 1
                self.HH_adapted_any_21 += 1

        self.share_liquidity_constrained = liquidity_constrained / self.nr_households
        self.share_unmet_adaptation_demand = unmet_adaptation_demand / self.nr_households
        # print("relative_burdens len:", len(relative_burdens))
        # print("relative_burdens min:", min(relative_burdens))
        # print("relative_burdens max:", max(relative_burdens))
        # print("relative_burdens first 10:", relative_burdens[:10])
        # print("liquidity constrained count:", liquidity_constrained)
        # print("unmet demand count:", unmet_adaptation_demand)
        self.gini_relative_burden = gini(
            relative_burdens
        )

        self.gini_relative_burden_cumulative_income = gini(
            relative_burdens_cumulative_income
        )

        self.gini_relative_burden_simulation_only = gini(
            relative_burdens_simulation_only
        )

        self.gini_relative_burden_cumulative_income_simulation_only = gini(
            relative_burdens_cumulative_income_simulation_only
        )

        self.average_relative_burden = np.mean(
            relative_burdens
        )

        self.average_relative_burden_cumulative_income = np.mean(
            relative_burdens_cumulative_income
        )
 
    def step(self):
        #print('step:', self.steps)
        # print('adapted', self.fraction_adapted_any)
        
        self.flood_shock = False
        if self.steps in self.flood_time:
            
            #print('Flood happening')
            self.flood_shock = True
            
        self.agents.shuffle_do("step") # shuffle_do: random activation, do: base scheduler
        
        self.update_model_variables()
        
        # add and remove network ties
        self.remove_network_ties(proportion_edges_to_remove = self.proportion_ties_removed_per_step)
        self.add_network_ties(proportion_new_edges = self.get_proportion_new_ties())
        
        #print(self.social_network)
        #self.count_within_cell_edges()
        
        # collect data
        self.datacollector.collect(self)
            

            
    def run_model(self, step_count = steps):
        
        for i in range(step_count):
            #self.datacollector.collect(self)
            self.step()
                
            #print('step', i, 'done')
        
    