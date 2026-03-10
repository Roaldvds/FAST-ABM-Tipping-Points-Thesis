# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

File to generate the household agents
"""

from mesa import Agent
import pandas as pd
import numpy as np
import random
from src.functions import logistic_adapted_to_SN, decrease_rate_worry_in_flood_remembrance_period
#import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

"""


"""
HH_PMTrelated = pd.read_csv('data/processed/PMT_variables.csv')
Logistic_reg_PMT = pd.read_csv('data/processed/logistic_regression_PMT/Logistic_regression_PMT_done_NLUKw5.csv')



class Households(Agent):
    """
    Class representing the household agents. 
    Initiated by the FloodAdaptationModel in model_file.py
    Initialising the household agents happens in multiple steps
        1. assigning the majority of their parameters 
    After the initialisation, each step the household agents
        1. update their parameters
        2. based on the updated parameters, adapt their likelihood to take a measure
        3. try to take a measure if "wanted"
        4. calculate the new perceptions based on their connections influence
        5. add a proportion of their income to their savings (whith which they buy measures)
    """
    def __init__(self, 
                 model):
        super().__init__(model)
        #print(self.unique_id, 'initialisation starts')
        
        # setting seed
        if model.seed is None:
            self.seed = model.seed
        else:
            self.seed = model.seed + self.unique_id
        random.seed(self.seed)
        
        """
        setting the household variables
        selecting random row in the dataframe and assigning these values
        depending on how many and which measures we select, make everything related to coping appraisal and measures into a dictionary
        """
        PMTrelated_index = random.randrange(len(HH_PMTrelated))
        # threat appraisal
        self.perceived_probability = HH_PMTrelated.iloc[PMTrelated_index]['Perc_probability']
        self.perceived_severity = HH_PMTrelated.iloc[PMTrelated_index]['perc_damage']
        self.perceived_risk = self.perceived_probability * self.perceived_severity
        self.worry = HH_PMTrelated.iloc[PMTrelated_index]['worry']
        if self.model.weight_SN_vs_PMT == 'empirical':
            #print('SN weight empirical')
            self.weight_PMT = HH_PMTrelated.iloc[PMTrelated_index]['weight_PMT']
        else:
            self.weight_PMT = 1 - self.model.weight_SN_vs_PMT
            # print('set SN weight, weight PMT:', self.weight_PMT)
        
        # coping appraisal
        self.perceived_response_efficacy = {'dry-proofing': HH_PMTrelated.iloc[PMTrelated_index]['resp_efficacy_DP'],
                                            'wet-proofing': HH_PMTrelated.iloc[PMTrelated_index]['resp_efficacy_WP']}
        self.self_efficacy = {'dry-proofing': HH_PMTrelated.iloc[PMTrelated_index]['self_efficacy_DP'],
                              'wet-proofing': HH_PMTrelated.iloc[PMTrelated_index]['self_efficacy_WP']}
        self.perceived_cost = {'dry-proofing': HH_PMTrelated.iloc[PMTrelated_index]['perc_cost_DP'],
                              'wet-proofing': HH_PMTrelated.iloc[PMTrelated_index]['perc_cost_WP']}
        self.measures_taken = {'dry-proofing': HH_PMTrelated.iloc[PMTrelated_index]['implemented_DP'],
                              'wet-proofing': HH_PMTrelated.iloc[PMTrelated_index]['implemented_WP']}
        
        if self.model.set_hh_adapted_at_start is not False:
            num_set_measures = random.uniform(0, 1)
            # print(num_set_measures)
            if num_set_measures <= self.model.set_hh_adapted_at_start:
                self.measures_taken = {'dry-proofing': 1,'wet-proofing': 1}
            else:
                self.measures_taken = {'dry-proofing': 0,'wet-proofing': 0}
            # print(self.measures_taken)

        self.flood_experience = HH_PMTrelated.iloc[PMTrelated_index]['flood_experience']
        self.flood_damage = 0
        self.damage_experienced = 0
        
        # how long it affects worry (drawing from truncated normal dist with given mean +- 2)
        self.personal_flood_remembrance_period = int(np.round(np.clip(np.random.normal(loc=self.model.flood_remembrance_period, scale=1.0), 
                                                                      self.model.flood_remembrance_period-2, 
                                                                      self.model.flood_remembrance_period+2))) 
        # print(self.personal_flood_remembrance_period)
        # social influence/norms related
        self.fraction_in_network_adapted = 0 # update later
        if self.model.per_HH_adapted_to_SN_prob_midpoint == 'uniform':
            print('uniform')
            self.own_per_HH_adapted_to_SN_prob_midpoint = round(random.uniform(0.3, 0.8), 2)
        else:
            self.own_per_HH_adapted_to_SN_prob_midpoint = self.model.per_HH_adapted_to_SN_prob_midpoint
        
        print('midpoint', self.own_per_HH_adapted_to_SN_prob_midpoint)

        self.protection_motivation = {'dry-proofing': 0,
                                      'wet-proofing': 0} # calculate later
        self.prob_from_social_norm = 0
        self.probability_to_take_measure = {'dry-proofing': 0,
                                            'wet-proofing': 0} # calculate later
        # if measure taken, make it anything between 0 and age measure
        self.age_of_measures = {'dry-proofing': 0,
                                'wet-proofing': 0}
        for measure in self.measures_taken.keys():
            if self.measures_taken[measure] == 1:
                age = random.randint(0, self.model.measures_aging[measure])
                # print(measure, 'age', age)
                self.age_of_measures[measure] = age
        # print(self.measures_taken)
        #print(self.age_of_measures)
        
        self.steps_since_tried_to_take_measure = 0

        
        
    def get_connected_HHagents(self):
        # get a list of the household agents they are connected to
        own_edges = self.model.social_network.edges(self.unique_id)
        self.connected_HHagents = []
        for edge in own_edges:
            connected_HH = edge[1]
            self.connected_HHagents.append(connected_HH)
        #print(self.connected_HHagents)
        self.nr_connected_HHagents = len(self.connected_HHagents)
        
    def get_own_district(self):
        # get a list of the household agents that are in the same cell
        self.district_HHagents = []
        for agent in self.model._agents:
            if agent.pos == self.pos:
                self.district_HHagents.append(agent.unique_id)
        self.nr_HH_in_district = len(self.district_HHagents)
        #print('district', self.nr_HH_in_district, self.district_HHagents)            

    def calculate_probability_to_take_measure_PMT(self, measure, other_measure):
        """
        Function to determine the probability to take a measure based PMT on logistic regression
        The factors for logistic regression were determined with the survey data
        For further info, check the .ipynb notebook in the input_data file
        """
        # get the weights for the measure
        PMT_weights = Logistic_reg_PMT[measure]
        #print(PMT_weights)
        PMT_values = [1, #intercept
                      self.worry,
                      self.perceived_severity,
                      self.perceived_probability,
                      self.flood_experience,
                      self.self_efficacy[measure],
                      self.perceived_response_efficacy[measure],
                      self.perceived_cost[measure],
                      self.measures_taken[other_measure]
                      ]
        #print(PMT_values)
        # get weighted average of attributes
        y_hat = np.dot(PMT_weights, PMT_values)
        # Take inverse logit function for adaptation (intention) probability
        y_hat_inv = round(min(1, np.exp(y_hat)/(1 + np.exp(y_hat))), 2)
        self.protection_motivation[measure] = round(y_hat_inv, 4)
        #print(self.protection_motivation)

    def influence_of_social_norm(self):
        
        #print(self.model.social_norm_calculation)
        
        if self.model.social_norm_calculation == 'connections':
            if self.nr_connected_HHagents == 0: # if no connections, SN from connections is 0
                self.prob_from_social_norm = 0
            else:
                # calcuate the social pressure to adapt ie how many of the connections have adapated and how that pressures you to adapt
                nr_connections_adapted = 0
                # loop through agents to find connected HH agent
                for agent in self.model._agents:
                    if agent.unique_id in self.connected_HHagents:
                        if 1 in agent.measures_taken.values(): #check if an adaptation measures is taken by that connecting agen
                            nr_connections_adapted += 1
                # print(self.nr_connected_HHagents, nr_connections_adapted)
                self.prob_from_social_norm = logistic_adapted_to_SN(nr_connections_adapted/self.nr_connected_HHagents, self.own_per_HH_adapted_to_SN_prob_midpoint)
            #print(self.prob_from_social_norm)
            
        if self.model.social_norm_calculation == 'all':
            # calculate social pressure to adapt ie how many have adapted overall
            self.prob_from_social_norm = logistic_adapted_to_SN(self.model.fraction_adapted_any, self.own_per_HH_adapted_to_SN_prob_midpoint)
            # print(self.prob_from_social_norm, self.model.fraction_adapted_any)
        if self.model.social_norm_calculation == 'district':
            # calculate social pressure to adapt ie how many HH in the district ie. with same location have adapted
            nr_adapted_in_district = 0
            for agent in self.model._agents:
                if agent.unique_id in self.district_HHagents:
                    if 1 in agent.measures_taken.values(): #check if an adaptation measures is taken by that agent
                        nr_adapted_in_district += 1
            # print(self.district_HHagents, nr_adapted_in_district)
            self.prob_from_social_norm = logistic_adapted_to_SN(nr_adapted_in_district/self.nr_HH_in_district, self.own_per_HH_adapted_to_SN_prob_midpoint)
            #print(self.prob_from_social_norm)
        # round to 4 decimals
        self.prob_from_social_norm = round(self.prob_from_social_norm, 4)

    def calculate_probability_to_take_measure_overall(self, measure):
        """

        Function to combine probabilities to take measure from PMT with social norm

        """
        self.probability_to_take_measure[measure] = (self.protection_motivation[measure] * self.weight_PMT) + (self.prob_from_social_norm * (1 - self.weight_PMT))
        # print('probs', self.protection_motivation[measure], self.prob_from_social_norm, self.probability_to_take_measure[measure])
    
    def try_to_take_measure(self, measure):
        """
        Function to take a measure
        A random number is calculated and compared with the probability to take a measure. This determines, whether the HH will take the measure or not. 
        The measure is "taken" meaning that the corresponding values in the dataframe are adapted and the damage is reduced.

        """
        random.seed(self.seed)
        
        if random.random() < self.probability_to_take_measure[measure]:
            #print('want to take measure')
            #print('savings', self.savings, 'costs', CCA_costs[measure])
            if self.savings > self.model.CCA_costs[measure]:
                #print('will take measure')
                self.flood_damage = self.calculate_flood_damage() # update flood damage after taking measure
                self.savings -=  self.model.CCA_costs[measure]
                self.measures_taken[measure] = 1
                #print(self.measures_taken)
                
    def age_measures(self):
        """
        Aging measures and removing them if their end-of-life is reached

        """
        for measure in self.age_of_measures.keys():
            #print(measure)
            if self.model.measures_aging[measure] == 0: # no aging
                #print('no aging of', measure)
                continue
            else: # aging of measures
                if self.measures_taken[measure] == 1:
                    self.age_of_measures[measure] += 1 # age measures by 1 
                #print(self.age_of_measures[measure], self.model.measures_aging[measure])
                if self.age_of_measures[measure] == self.model.measures_aging[measure]:
                    # if measures age same as end of life age, remove the measure and set the age back to 0
                    #print(measure, 'reached end of life')
                    self.measures_taken[measure] = 0
                    self.age_of_measures[measure] = 0
        #print(self.measures_taken, self.age_of_measures)
            
        
    def calculate_flood_damage(self):
        """
        Calculates the flood damage based on the location of the HH and the flood depth there.
        Uses Huizinga's depth-damage function:
            1. calculates damage factr based on flood depth
            2. calculate damage factor with measures based on reduction through measures taken
            3. calculate flood damage based on damage factors (with and without measures), the max damage per sqm and the house size
        """
        self.flood_depth = self.model.flood_depth_in_m[self.pos[1]] # get it via the y value position
        if self.flood_depth >= 6 or self.flood_depth == 0:
            if self.flood_depth >= 6:
                self.flood_damage_factor = 1
            else:
                self.flood_damage_factor = 0
        else:
            self.flood_damage_factor = round(0.0065 * (self.flood_depth**self.flood_depth) - 0.0896 * (self.flood_depth*self.flood_depth) + 0.457 * self.flood_depth, 2) # see damage function excel for this equation
        if self.flood_depth == 0:
            self.flood_damage_factor = 0
        self.flood_damage_factor_with_measures = self.flood_damage_factor
        # TODO check if this makes sense (order)
        if self.measures_taken['dry-proofing'] == 1:
            self.flood_damage_factor_with_measures = (1-self.model.damage_reduction['dry-proofing']) * self.flood_damage_factor
        if self.measures_taken['wet-proofing'] == 1:
            self.flood_damage_factor_with_measures = (1-self.model.damage_reduction['wet-proofing']) * self.flood_damage_factor
        
        self.flood_damage = round(self.flood_damage_factor_with_measures * self.model.max_damage_per_sqm * self.house_size, 2)
        self.flood_damage_no_measures = round(self.flood_damage_factor * self.model.max_damage_per_sqm * self.house_size, 2)
        # print('damage', self.flood_damage, self.flood_damage_no_measures)
    
    def influence_of_network_on_PMT_perceptions(self):
        """
        Function to determine the influence of the network connections.
        People take in the opinions of their connections with a certain weight and keep their own opinion for the rest.
        Some relations ahve an extra weight assigned to them --> they are considered more in the formation of a new opinion.
        The new values with the influence of others are determined based on either
        - an assimilative opinion dynamics model (Degroot) where all opinions are considered.
        - a similarity-biased opinion dynamics model (inspired by Hegselmann & Krause) where only those opinions are considered, that are less than a threshold different. 
        
        """
        #creating empty lists to store the perceptions of the connections in
        perceived_severity_of_connections = []
        perceived_probability_of_connections = []
        perceived_costs_of_connections = {'dry-proofing': [],
                                          'wet-proofing': []}
        perceived_response_efficacy_of_connections = {'dry-proofing': [],
                                                      'wet-proofing': []}
        # no exchange of self-efficacy as this is highly individual

        # loop through all agents and store the relevant perception values of the connections
        for agent in self.model._agents:
            if agent.unique_id in self.connected_HHagents:
                # only consider the perceptions as specified in the model input
                # and append the connecting hh agents' values to the lists
                # if self.model.opinion_dynamics_model == 'similarity_biased'
                # check the weight of the edge between the two HH agents
                weight = self.model.social_network.get_edge_data(self.unique_id, agent.unique_id).get('weight')
                # print(weight)
                if (self.model.exchange_what == 'threat') or (self.model.exchange_what == 'threat+coping'): # check if threat exchanged
                    # calculate opinion difference
                    opinion_difference_threat = abs((self.worry + self.perceived_severity + self.perceived_probability) - 
                                                (agent.worry + agent.perceived_severity + agent.perceived_probability))
                    # print('threat OD diff', opinion_difference_threat)
                    if self.model.opinion_dynamics_model == 'assimilative' or (self.model.opinion_dynamics_model == 'similarity_biased' and opinion_difference_threat < self.model.threshold_similarity_bias): # only consider opinion of assimilative or if similarity bias threshold kept
                        perceived_severity_of_connections.append(agent.perceived_severity)
                        perceived_probability_of_connections.append(agent.perceived_probability)
                        
                        if weight == 2: # if the weight on the edge connecting the two HH agents is 2, add the value another time
                            perceived_severity_of_connections.append(agent.perceived_severity)
                            perceived_probability_of_connections.append(agent.perceived_probability)
                            
                if (self.model.exchange_what == 'coping') or (self.model.exchange_what == 'threat+coping'): #check if coping exchanged
                    for key in agent.perceived_cost:
                        # keys the same for perceived costs and perceived response efficacy so can do in one go
                        # calculate opinion difference
                        opinion_difference_coping = abs((self.perceived_cost[key] + self.perceived_response_efficacy[key] + self.self_efficacy[key]) -
                                                (agent.perceived_cost[key] + agent.perceived_response_efficacy[key] + agent.self_efficacy[key]))
                        # print('coping OD diff', opinion_difference_coping)
                        #perceived costs
                        if self.model.opinion_dynamics_model == 'assimilative' or (self.model.opinion_dynamics_model == 'similarity_biased' and opinion_difference_coping < self.model.threshold_similarity_bias):
                            perceived_costs_of_connections[key].append(agent.perceived_cost[key])
                            perceived_response_efficacy_of_connections[key].append(agent.perceived_response_efficacy[key])
                            if weight == 2:
                                perceived_costs_of_connections[key].append(agent.perceived_cost[key])
                                perceived_response_efficacy_of_connections[key].append(agent.perceived_response_efficacy[key])
                                
        
        # print('worry', worry_of_connections, 'costs', perceived_costs_of_connections)
        sum_severity = 0
        sum_probability = 0

        # now updated immediately --> agents that are activated later, get already adjusted opinion
        if len(perceived_severity_of_connections) > 0:
            for i in range(len(perceived_severity_of_connections)):
                sum_severity += perceived_severity_of_connections[i]
            self.perceived_severity = round((1-self.model.weight_opinion_others) * self.perceived_severity + self.model.weight_opinion_others * sum_severity/len(perceived_severity_of_connections), 4)
        
        if len(perceived_probability_of_connections) > 0:
            for i in range(len(perceived_probability_of_connections)):
                sum_probability += perceived_probability_of_connections[i]
            self.perceived_probability = round((1-self.model.weight_opinion_others) * self.perceived_probability + self.model.weight_opinion_others * sum_probability/len(perceived_probability_of_connections), 4)
        
            
        for key in perceived_costs_of_connections:
            
            # perceied cost
            sum_perceived_costs_of_connections_m = 0
            if len(perceived_costs_of_connections[key]) > 0:
                #print(perceived_costs_of_connections[key])
                for i in range(len(perceived_costs_of_connections[key])):
                    sum_perceived_costs_of_connections_m += perceived_costs_of_connections[key][i]
                #print('before', self.perceived_cost[key])
                self.perceived_cost[key] = round((1-self.model.weight_opinion_others) * self.perceived_cost[key] + self.model.weight_opinion_others *  sum_perceived_costs_of_connections_m/len(perceived_costs_of_connections[key]), 4)
            
            # perceived response efficacy
            sum_perceived_response_efficacy_of_connections_m = 0
            if len(perceived_response_efficacy_of_connections[key]) > 0:
                for i in range(len(perceived_response_efficacy_of_connections[key])):
                    sum_perceived_response_efficacy_of_connections_m += perceived_response_efficacy_of_connections[key][i]
                self.perceived_response_efficacy[key] = round((1-self.model.weight_opinion_others) * self.perceived_response_efficacy[key] + self.model.weight_opinion_others *  sum_perceived_response_efficacy_of_connections_m/len(perceived_response_efficacy_of_connections[key]), 4)

       #print('after', self.perceived_cost) #self.worry,
        
       
    def save_money(self):
        # add a perceptage of the income to the savings
        self.savings = round(self.savings + self.income * self.model.prop_inc_for_damage_adaptation, 0)
        
    def experience_flood_shock(self):
        # experiencing a flood
        #print('pre flood: damage:', self.flood_damage, 'savings left:', self.savings)
        # print('flood')
        self.damage_experienced = self.damage_experienced + self.flood_damage
        self.savings = round(self.savings - self.flood_damage, 0)
        self.flood_experience = 1
        # print(self.flood_damage)
        if self.model.worry_increase_with_flood > 0:
            self.worry_pre_flood = self.worry
            self.worry = min(round(self.worry + self.model.worry_increase_with_flood, 2), 1) # worry max value is 1
            self.worry_decrease_rate = decrease_rate_worry_in_flood_remembrance_period(self.worry_pre_flood, 
                                                                                        self.worry, 
                                                                                        self.model.worry_increase_with_flood, 
                                                                                        self.model.min_worry_after_flood,
                                                                                        self.personal_flood_remembrance_period)
        # print(self.worry_pre_flood, 'after', self.worry, 'decrease rate', self.worry_decrease_rate)
        #print('post flood: damage:', self.damage_experienced, 'savings left:', self.savings)
    
    def calculate_worry(self):
        # while in personal flood remembrance period, decrease worry with worry_decrease_rate
        # print('before decrease', self.worry)
        print(self.worry_decrease_rate)
        if self.worry_decrease_rate:
            self.worry -= self.worry * self.worry_decrease_rate
            # worry should not be 0
            if self.worry > self.model.min_worry_after_flood:
                self.worry = self.model.min_worry_after_flood
            # print('after decrease', self.worry)
    
    def step(self):
        self.save_money()
        self.get_connected_HHagents()
        self.get_own_district() # not moving so could also only do this once
        
        # while in flood remembrance period, calculate worry
        if self.damage_experienced > 0:
            for flood_time in self.model.flood_time:
                if (self.model.steps > flood_time) and (self.model.steps <= flood_time + self.personal_flood_remembrance_period):
                    # print('update worry')
                    print(self.worry_decrease_rate)
                    self.calculate_worry()

        if self.nr_connected_HHagents > 0:# if no connections, no influence on perceptions from network and social norm
            self.influence_of_network_on_PMT_perceptions()
        self.influence_of_social_norm()
        # print(self.unique_id, 'savings', self.savings)
        self.calculate_probability_to_take_measure_PMT('dry-proofing', 'wet-proofing')
        self.calculate_probability_to_take_measure_PMT('wet-proofing', 'dry-proofing')
        self.calculate_probability_to_take_measure_overall('wet-proofing')
        self.calculate_probability_to_take_measure_overall('dry-proofing')
        if self.model.steps % self.model.consider_taking_measure_every_x_steps == 0:
            #print(self.model.steps, 'taking measures potentially')
            #print(self.probability_to_take_measure)
            if self.measures_taken['dry-proofing'] == 0 and (self.probability_to_take_measure['dry-proofing'] > self.model.minimum_intention_to_consider_measure):
                #print('trying to take DP measure')
                self.try_to_take_measure('dry-proofing')
                #print('damage DP', self.flood_damage)
            if self.measures_taken['wet-proofing'] == 0 and (self.probability_to_take_measure['wet-proofing'] > self.model.minimum_intention_to_consider_measure):
                #print('trying to take WP measure')
                self.try_to_take_measure('wet-proofing')
                    #print('damage WP', self.flood_damage)
        self.age_measures()
        
        self.calculate_flood_damage()
        if self.model.flood_shock:
            if self.flood_damage > 0:
                self.experience_flood_shock()
                
        
        
        

    
        