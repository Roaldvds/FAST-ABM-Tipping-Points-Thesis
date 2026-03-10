# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Contains the model class for the flood adaptation model.
"""
import numpy as np
import random
from mesa import Model
from mesa.space import MultiGrid
import networkx as nx
from agent import Households
from functions import get_connections_distribution_halfnormal, construct_network

steps = 100


class AdaptationModel(Model):
    """
    Class representing the household adaptation model. 
    Models the uptake of flood adaptation measures of households under the 
    influence of social networks and opinion diffusion.
    """
    def __init__(self,
                 seed = None,
                 nr_households = 10, # number of household agents
                 savings_rate = 0.05, # how much each household saves of their monthly income
                 
                 # what HH "talk" about and hence get influenced in by their connections
                 exchange_what = ['threat', 'coping', 'threat+coping'],
                 
                 # Grid related
                 grid_width = 5,
                 grid_height = 5,
                 flood_depth_in_m = [99, 1, 0.6, 0.4, 0.1], # same length as grid_height, first 99 (water)
                 population_density = [[0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5],
                                       [0,5,5,5,5]], # this has to add up to the number of households
                                       
                 
                 # network structure related
                 max_nr_connections = 8,
                 perc_until_what_nr_conn = 0.8,
                 most_conn_below = 3,
                 
                 fraction_neighbors = 0.15,
                 # flood damage related: from Huizinga, de Moel --> damage factor
                 # in dollar and adjusted for inflation to 2020 value
                 max_damage_dol_per_sqm = 1216.65
                 ):
        
        super().__init__(seed = seed)
        
        # defining the variables and setting the values
        self.nr_households = nr_households
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.flood_depth_in_m = flood_depth_in_m
        self.population_density = population_density
        if sum(sum(inner_list) for inner_list in self.population_density) != self.nr_households:
            raise ValueError('Population distirbution is', self.population_density, 
                             ' and does not match the number of household agents. nr household agents:', 
                             self.nr_households)
        
        self.max_nr_connections = max_nr_connections
        self.perc_until_what_nr_conn = perc_until_what_nr_conn
        self.most_conn_below = most_conn_below
        self.fraction_neighbors = fraction_neighbors
        
        self.seed = seed
        self.max_damage_dol_per_sqm = max_damage_dol_per_sqm
        self.exchange_what = exchange_what
        

        
        '''
        Making the space
        '''
        self.make_grid()
        self.add_HHagents_to_grid()
        # TODO make grid visualisation
        
        '''
        making the social network structure
        '''
        self.make_network_structure()
        self.place_HHagents_on_network()
        self.update_node_id()
        self.count_within_cell_edges()
        self.get_network_descriptives()



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
        print(self.cell_properties)
        
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
                
    def make_network_structure(self):
        self.pdf, self.x_linspace = get_connections_distribution_halfnormal(self.perc_until_what_nr_conn,
                                                                            self.most_conn_below, 
                                                                            self.max_nr_connections)
        self.social_network, self.degrees = construct_network(self.pdf, 
                                                              self.max_nr_connections, 
                                                              self.nr_households)
        # if not enough nodes, add a random, unconnected node
        if self.social_network.number_of_nodes() < self.nr_households:
            nodes_missing = self.nr_households - self.social_network.number_of_nodes()
            for n in range(nodes_missing):
                self.social_network.add_node(self.social_network.number_of_nodes()+1)
        if self.social_network.number_of_nodes() != self.nr_households:
            raise ValueError('The number of households and number of nodes of the network are not equal.')
        print(self.social_network)
        # Create a dictionary with None values for all nodes
        attributes = {node: {'household': None} for node in self.social_network.nodes}
        # Set the attributes
        nx.set_node_attributes(self.social_network, attributes)
        
    def place_HHagents_on_network(self):
        # other idea:
            # get the edges in the network
            # select fraction_neighbors edges from it randomly
            # place always to houwholds on the nodes with the same cell value
            # set the rest of the nodes randomly
            
        self.households_without_node = []
        for agent in self.agents:
            self.households_without_node.append(agent.unique_id)
        # print(len(self.households_without_node))
        
        # Calculate the number of within-cell edges (15% of total edges)
        num_within_cell_edges = int(self.fraction_neighbors * self.social_network.number_of_edges())
        print('Edges needded within cell ie neighbors: ', num_within_cell_edges)
        edges = list(self.social_network.edges())
        neighbor_edges = random.sample(edges, num_within_cell_edges)       
        # print(neighbor_edges)
        
        cells_with_min_2_households = {cell: agents for cell, agents in self.cell_households.items() if len(agents) >= 2}
        if not cells_with_min_2_households:
            print('Cannot make niehgbor connection because there are no households that share a cell')
        #print(cells_with_min_2_households)

        for edge in neighbor_edges: 
            # print('edge', edge)
            if (self.social_network.nodes[edge[0]].get("household") is None) and (self.social_network.nodes[edge[1]].get("household") is None):
                #print('no households placed on any of these nodes')
                # select two agents randomly that are in households_without_node and share the same location. 
                selected_cell = random.choice(list(cells_with_min_2_households.keys()))# select a random cell
                households_in_cell = self.cell_households[selected_cell] # get the households in that cell
                # check which households do not have a ndoe assigned within that cell
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
        # Update node id of the network so it's the same as the household id
        # Create a mapping of the current node IDs to the 'household' values
        node_relabel_map = {}
        for node in list(self.social_network.nodes()):
            household_value = self.social_network.nodes[node].get('household')
            if household_value is not None:
                # Remove the node with the old ID and add it with the new ID (household value)
                node_relabel_map[node] = household_value       
        self.social_network = nx.relabel_nodes(self.social_network, node_relabel_map)

        #print(self.social_network)
        
    def count_within_cell_edges(self):
        # function to count how many edges are within a cell ie neighbors
        all_edges = list(self.social_network.edges())
        self.neighborhood_edges = []
        for edge in all_edges:
            for (x,y), agent_list in self.cell_households.items():
                if (self.social_network.nodes[edge[0]].get("household") in agent_list) and (self.social_network.nodes[edge[1]].get("household") in agent_list):
                    self.neighborhood_edges.append(edge)
                else:
                    continue
        print('all edges, neighborhood edges', len(all_edges), len(self.neighborhood_edges))
        self.proportion_neighborhood_edges = len(self.neighborhood_edges)/len(all_edges)
        print(self.proportion_neighborhood_edges)
    
    def get_network_descriptives(self):
        # Basic properties
        num_nodes = self.social_network.number_of_nodes()
        num_edges = self.social_network.number_of_edges()
        is_directed = nx.is_directed(self.social_network)
        
        # Connected components
        num_connected_components = nx.number_connected_components(self.social_network) if not is_directed else nx.number_strongly_connected_components(self.social_network)
        connected_components = list(nx.connected_components(self.social_network)) if not is_directed else list(nx.strongly_connected_components(self.social_network))
        
        # Degree distribution
        degree_sequence = [d for n, d in self.social_network.degree()]
        avg_degree = sum(degree_sequence) / num_nodes
        
        # Clustering coefficient
        avg_clustering = nx.average_clustering(self.social_network)
        
        # Diameter (for each connected component)
        diameters = []
        for component in connected_components:
            subgraph = self.social_network.subgraph(component)
            if nx.is_connected(subgraph):  # Ensure the subgraph is connected
                diameters.append(nx.diameter(subgraph))
        
        # Density
        density = nx.density(self.social_network)
        
        # Centrality measures (example: degree centrality)
        degree_centrality = nx.degree_centrality(self.social_network)
        betweenness_centrality = nx.betweenness_centrality(self.social_network)
        # Print the results
        print()
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Is directed: {is_directed}")
        print(f"Number of connected components: {num_connected_components}")
        #print(f"Connected components: {connected_components}")
        print(f"Average degree: {avg_degree}")
        #print(f"Degree sequence: {degree_sequence}")
        print(f"Average clustering coefficient: {avg_clustering}")
        print(f"Diameter of connected components: {diameters}")
        print(f"Density of the graph: {density}")
        #print(f"Degree centrality: {degree_centrality}")
        print(f"Betweenness centrality: {betweenness_centrality}")
        print()

    
    def remove_network_ties(self):
        # function to remove some ties
        pass
    
    def add_network_ties(self):
        # function to add some ties
        pass
        
 
    def step(self):
        self.agents.shuffle_do("step") # shuffle_do: random activation, do: base scheduler
        self.remove_network_ties()
        self.add_network_ties()
        #self.count_within_cell_edges()
        # check https://mesa.readthedocs.io/latest/migration_guide.html#replacing-schedulers-with-agentset-functionality
        print('step:', self.steps)
        # collect data
      #  self.datacollector.collect(self)
            

            
    def run_model(self, step_count = steps):
        
        for i in range(step_count):
            #self.datacollector.collect(self)
            self.step()
                
            #print('step', i, 'done')
        
    