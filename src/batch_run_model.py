# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 13:35:23 2022

@author: thoridwagenblast
"""

from src.model import AdaptationModel
import pandas as pd
import numpy as np
#from functions import household_datacollector, model_datacollector
#from mesa.batchrunner import BatchRunner
from mesa.batchrunner import batch_run

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

if __name__ == '__main__':
    parameter_sweep = {'seed': range(12345, 12345+3),
                       'nr_households': [1000],
                       'set_hh_adapted_at_start': [0.04],
                       #'savings_rate': [0.05], 
                       'exchange_what': ['threat+coping'], #'threat' or 'coping'
                       'social_norm_calculation': ['all'], #'connections', # 'neighbors', 'all'
                       'per_HH_adapted_to_SN_prob_midpoint': [0.5],
                       'weight_SN_vs_PMT': [0.5], #'empirical'
                       
                       'opinion_dynamics_model': ['similarity_biased'], #'assimilative'
                       'weight_opinion_others': [0.5],
                       'threshold_similarity_bias': [0.2],
                       
                       # Grid related
                       # grid_width = 5,
                       # grid_height = 5,
                       'flood_depth_in_m': [[99, 3, 1, 1, 1]], # same length as grid_height, first 99 (water)
                       'flood_time':  [[10]],# [20], [30], [40], [50]],
                       'population_density': [[[0,50,50,50,50],
                                             [0,50,50,50,50],
                                             [0,50,50,50,50],
                                             [0,50,50,50,50],
                                             [0,50,50,50,50]]],

                       'network_structure_file_path': ['data/processed/network_structures/Network_empirical.graphml'],
                       
                       # fraction_neighbors = 0.15,
                       # fraction_homophily = 0.25,
                       # fraction_family_and_friends = 0.55,
                       'proportion_ties_added_per_step': [0],
                       'proportion_ties_removed_per_step': [0],
                       'fraction_ties_added_at_flood': [0],
                       'worry_increase_with_flood': [0.4],
                       'min_worry_after_flood': [0.1],
                       'flood_remembrance_period': [5], # how long the experience of flood leads to more connection in the network
                       'consider_measure_every_x_steps': [5],
                       # flood damage related: from Huizinga, de Moel --> damage factor
                       # in dollar and adjusted for inflation to 2020 value
                       # max_damage_per_sqm = 950,
                       # CCA_costs = {'dry-proofing': 10300, 
                       #              'wet-proofing': 8000},
                       # damage_reduction = {'dry-proofing': 0.5, 
                       #              'wet-proofing': 0.5},
                       'measures_aging': [{'dry-proofing': 20, # 0 if no aging, int for number of years they age
                                    'wet-proofing': 20}],           
                        'transformative_threshold': [0.7],
                        'collect_agent_data': [1]
                       }
    
    num_iterations = 1
    steps = 50
    
    #agent_collector = household_datacollector()
    #model reporters : how many percent have taken each measure
    #model_collector = model_datacollector()
    
    results = batch_run(AdaptationModel,
                        number_processes=None,
                        parameters=parameter_sweep,
                        display_progress=True,
                        iterations = num_iterations,
                        max_steps=steps,
                        data_collection_period=1)
    
    results_df = pd.DataFrame(results)
    
    results_df['flood_time'] = results_df['flood_time'].astype(str)
    results_df['flood_depth_in_m'] = results_df['flood_depth_in_m'].astype(str)
    results_df['measures_aging'] =  results_df['measures_aging'].astype(str)
    
    model_results = results_df.drop(columns = ['RunId', 'iteration', 'population_density', 'AgentID', 'network_structure_file_path', 'savings', 'income', 
                                               'risk_perception', 'worry', 'PMT_intention_DP', 'PMT_intention_WP',
                                               'nr_connections', 'SN_intention', 'overall_intention_DP', 'overall_intention_WP',
                                               'DP_taken', 'WP_taken'], 
                                    axis = 1)
    model_results = model_results.drop_duplicates(ignore_index = True)
    model_results = model_results.groupby(['Step', 
                                           'flood_time', 
                                           'flood_depth_in_m',
                                           'measures_aging',
                                           'opinion_dynamics_model', 
                                           'exchange_what', 
                                           'social_norm_calculation']).agg([np.mean, np.std, np.min, np.max, np.median, q1, q3])
    
    model_results.to_csv('results/model_data_250716.csv')
    
    agent_results = results_df.drop(columns = ['RunId', 'iteration','population_density', 'measures_aging', 'network_structure_file_path', 'edges', 
                                               'cumulative_damage', 'cumulative_damage_no_adaptation', 
                                            'fraction_wet_proofed', 'fraction_dry_proofed', 'fraction_adapted_any', 'fraction_adapted_both'],
                                 axis = 1)
    agent_results = agent_results[agent_results.Step != 0]
    #print(agent_results.dtypes)
    agent_results = agent_results.groupby(['Step', 
                                           'flood_time', 
                                           'flood_depth_in_m',
                                           'opinion_dynamics_model', 
                                           'exchange_what', 
                                           'social_norm_calculation']).agg([np.mean, np.std, np.min, np.max, np.median, q1, q3])
    # probably need to only collect stats when running more
    agent_results.to_csv('results/agent_data_250716.csv')
    
    #results_df.to_csv('../output_data/data_test_batch_run.csv')
    





 