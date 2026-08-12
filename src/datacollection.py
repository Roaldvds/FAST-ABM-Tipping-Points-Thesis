#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Stores the model and agent variables for the datacollection.

"""

def household_datacollector(nr_households):
    """
    Function for the datacollection of the household agent parameters.
    Comment/uncomment those lines / parameters that you do (not) want to collect.
    Data collection happens each step for every household (depending on the number of 
    agents these can be huge datasets, so select what you want to collect with caution).
    
    Parameters
    ----------
    nr_households: number of household agents
    
    Returns
    -------
    agent_datacollector    
    """
    agent_datacollector = {
        # Money
        # 'savings': lambda x: x.savings,
        # 'income': lambda x: x.income, # does not change
        # Damage
        #'flood_damage_no_measures': lambda x: x.flood_damage_no_measures,
        #'flood_damage_w_measures': lambda x: x.flood_damage,
        #'flood_damage_experienced': lambda x: x.damage_experienced,
            
        # PMT related
        # 'risk_perception': lambda x: x.perceived_risk,
        # 'worry': lambda x: x.worry,
        # 'PMT_intention_DP': lambda x: x.protection_motivation['dry-proofing'],
        # 'PMT_intention_WP': lambda x: x.protection_motivation['wet-proofing'],
        # Social influence
        # 'nr_connections': lambda x: x.nr_connected_HHagents,
        # 'SN_intention': lambda x: x.prob_from_social_norm,
        # adaptation 
        # 'overall_intention_DP': lambda x: x.probability_to_take_measure['dry-proofing'],
        # 'overall_intention_WP': lambda x: x.probability_to_take_measure['wet-proofing'],
        'DP_taken': lambda x: x.measures_taken['dry-proofing'],
        'WP_taken': lambda x: x.measures_taken['wet-proofing'],
        # location
        'location': lambda x: x.pos
        }
    
    return agent_datacollector

def model_datacollector():
    """
    Function for the datacollection of the model parameters.
    Comment/uncomment those lines / parameters that you do (not) want to collect.

    Returns
    -------
    model_datacollector
    
    """
    
    model_datacollector = { 
        # Network
        'edges': lambda m: m.social_network.number_of_edges(),
        'avg_clustering': lambda m: m.avg_clustering_nw,
        'density_nw': lambda m: m.density_nw,
        #'diameters_counter': lambda m: m.diameters_nw_counter,

         # Damage
         'cumulative_damage': lambda m: m.cumulative_damage,
         'cumulative_damage_no_adaptation': lambda m: m.cumulative_damage_no_adaptation, 
         # Adaptation
         'fraction_wet_proofed': lambda m: m.fraction_wet_proofed,
         'fraction_dry_proofed': lambda m: m.fraction_dry_proofed,
         'fraction_adapted_any': lambda m: m.fraction_adapted_any,
         'fraction_adapted_both': lambda m: m.fraction_adapted_both,

         # severly affected HH
         'nr_HH_severely_affected': lambda m: m.HH_severe_fl_exp,
         'nr_HH_severely_affected_adapted_any': lambda m: m.HH_severe_fl_adapted_any,

         'transformative_adaptation': lambda m: m.transformative_adaptation,
         # Adaptation Intentions:
         'avg_intention_SN': lambda m: m.avg_intention_SN,
         'avg_intention_PMT_DP': lambda m: m.avg_intention_PMT_DP,
         'avg_intention_PMT_WP': lambda m: m.avg_intention_PMT_WP,
         'avg_intention_overall_DP': lambda m: m.avg_intention_overall_DP,
         'avg_intention_overall_WP': lambda m: m.avg_intention_overall_WP,

        "average_worry": lambda m: m.average_worry,
        "average_perceived_risk": lambda m: m.average_perceived_risk,
        "average_savings": lambda m: m.average_savings,
        "average_flood_damage": lambda m: m.average_flood_damage,

        "total_damage_experienced": lambda m: m.total_damage_experienced,
        "average_damage_experienced": lambda m: m.average_damage_experienced,
        "share_households_with_expiry": lambda m: m.share_households_with_expiry,
        "average_active_measure_age": lambda m: m.average_active_measure_age,
        "average_remaining_measure_lifetime": lambda m: m.average_remaining_measure_lifetime,
        "average_fraction_connections_adapted": lambda m: m.average_fraction_connections_adapted,

        "share_liquidity_constrained": lambda m: m.share_liquidity_constrained,
        "share_unmet_adaptation_demand": lambda m: m.share_unmet_adaptation_demand,
        "gini_relative_burden": lambda m: m.gini_relative_burden,
        "adaptation_gap_high_low": lambda m: m.adaptation_gap_high_low,
        "gini_relative_burden_cumulative_income": lambda m: m.gini_relative_burden_cumulative_income,
        "average_relative_burden": lambda m: m.average_relative_burden,
        "average_relative_burden_cumulative_income": lambda m: m.average_relative_burden_cumulative_income,
        "share_optimal_to_adapt": lambda m: m.share_optimal_to_adapt,
        "share_adaptation_deficit": lambda m: m.share_adaptation_deficit,
        "share_over_adapted": lambda m: m.share_over_adapted,
        "gini_relative_burden_simulation_only": lambda m: m.gini_relative_burden_simulation_only,
        "gini_relative_burden_cumulative_income_simulation_only": lambda m: m.gini_relative_burden_cumulative_income_simulation_only,
        }
    
    return model_datacollector
