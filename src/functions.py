# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Functions that are used in the model_file.py and agent.py for the running of the Flood Adaptation Model.
Functions get called by the Model and Agent class.
"""
import numpy as np
import random
from scipy.stats import halfnorm
import networkx as nx


def get_connections_distribution_halfnormal(target_cdf, upper_bound, end_value):
    """
    Function to get a distribution for the number of connections. The underlying assumption is that most households have few connections, but there are some outliers. 
    This uses a half-normal distribution. 
    
    target_cdf: fraction of households that are supposed to have upper_bound or fewer connections
    upper_bound: value where target_cdf % of connections should be in.
    end_value: end of the tail. Maximum number of connections
    """
    
    # Solve for the scale parameter of the half-normal distribution
    scale = upper_bound / halfnorm.ppf(target_cdf)

    # Generate data points from the half-normal distribution
    x = np.linspace(0, end_value+1, end_value+1)  # Extend to the upper bound, +1 because we want it to still have the end_value (because it starts with 0)
    pdf = halfnorm.pdf(x, scale=scale)
    pdf = np.array(pdf)
    #probabilities /= pdf.sum()
    return pdf, x

def construct_network(probability_distribution, max_connections, nr_nodes):
    """
    Function to make the network graph given a probability distribution. 
    
    probability_distribution: probability distribution of the node degree
    max_connections: maximun number of connections an agent can have
    nr_nodes: number of nodes wanted for the network
    """
    # get connection values from the probability distribution
    p = np.array(probability_distribution)
    p /= p.sum()  # normalize
    values = np.random.choice(max_connections+1, nr_nodes, p = p) # get values in that distribution
    if sum(values) % 2 == 0:
        pass
    elif sum(values) % 2 != 0: # remove a random odd number from the array so it gets an even sum
        odd_indices = np.where(values % 2 != 0)[0] # find all odd numbers in array
        values = np.delete(values, random.choice(odd_indices)) # remove the a random odd number from the vlaues
    # generate graph based on the values
    G = nx.configuration_model(values)
    G = nx.Graph(G)  # Remove parallel edges and self-loops created by the configuration model
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
    # Verify the degree distribution
    actual_degrees = [d for _, d in G.degree()]
    return G, actual_degrees


def rewire_for_preferential_attachment(graph, iterations=5000):
    # Rewire edges to enforce preferential attachment based on degree similarity
    for _ in range(iterations):
        # Randomly select two edges (u1, v1) and (u2, v2)
        edge1 = random.choice(list(graph.edges()))
        edge2 = random.choice(list(graph.edges()))
        
        u1, v1 = edge1
        u2, v2 = edge2
        
        # Ensure the edges are distinct 
        if u1 != u2 and u1 != v2 and v1 != u2 and v1 != v2:
            # Calculate the degree similarity for the current and potential edges
            current_similarity = abs(graph.degree(u1) - graph.degree(v1)) + abs(graph.degree(u2) - graph.degree(v2))
            potential_similarity = abs(graph.degree(u1) - graph.degree(u2)) + abs(graph.degree(v1) - graph.degree(v2))
            
            # Rewire if the proposed edges have a better (ie more similar) degree similarity
            if potential_similarity < current_similarity:
                graph.remove_edges_from([edge1, edge2])
                graph.add_edges_from([(u1, u2), (v1, v2)])

def find_homophily_indexes(df):
    # Function to find two random indexes meeting the conditions
    # Group by income and level of education
    grouped = df.groupby(['income_category', 'Level_Education'])
    
    valid_pairs = []
    # Iterate through each group
    for _, group in grouped:
        if len(group) >= 2:  # Ensure there are at least 2 rows in the group
            # Sort by 'age' to easier check the difference
            group = group.sort_values(by='age')
            # Check for pairs where the age difference is <= 10
            for i in range(len(group) - 1):
                for j in range(i + 1, len(group)):
                    if abs(group.iloc[i]['age'] - group.iloc[j]['age']) <= 10:
                        # Store the pair in sorted order to avoid duplicates
                        pair = tuple(sorted((group.index[i], group.index[j])))
                        if pair not in valid_pairs:  # Avoid adding duplicates
                            valid_pairs.append(pair)
    return valid_pairs

def logistic_adapted_to_SN(x, midpoint, k=15):
    """Logistic curve scaled to [0,1] with adjustable midpoint."""
    # Shift and scale to enforce f(0)=0, f(1)=1, f(midpoint)=0.5
    x_scaled = (x - midpoint) * k  # Center at midpoint, scale by steepness
    sigmoid = 1 / (1 + np.exp(-x_scaled))  # Standard sigmoid
    # Linear transformation to map sigmoid's range (0,1) to [0,1]
    return sigmoid

def decrease_rate_worry_in_flood_remembrance_period(worry_pre_flood, 
                                                    worry_flood, 
                                                    increase_worry, 
                                                    min_worry_after_flood,
                                                    flood_remembrance_period):
    """ 
    Calculating the decrease rate for the personal flood remembrance period.
    It is decreasing until the nd of the flood remembrance period. After flood experience,
    it will always be 10% higher than the worry before the flood.
    For all worry > 0.9 before the flood, we set the max worry afterwards to 0.95.
    """
    if worry_pre_flood > 1 - increase_worry:
        # end target: 0.95
        worry_decrease_rate = 1 - (0.95/worry_flood) ** (1/flood_remembrance_period)
    else:
        # end target: 1.1 worry_pre_flood
        target_ratio = (1.1 * worry_pre_flood)/worry_flood
        if target_ratio == 0:
            target_ratio = min_worry_after_flood
        if target_ratio > 1:
            return None # shouldn't happen for worry_pre_flood <= 1 - increase_worry
        worry_decrease_rate = 1 - target_ratio ** (1/flood_remembrance_period)
    return worry_decrease_rate