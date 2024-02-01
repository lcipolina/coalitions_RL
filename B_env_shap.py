'''Environment to calculate Coalition Structure Generation Game
   - Agents are selected in round-robin fashion and offered coalitions to join by the env
   - Agents don't need to know the characteristic function as it is given as the reward by the env ('deltas')
   - Since the deltas are marginal contributions, the Shapley val is obtained as a byproduct....


   The steps are as follows:
    1. Agents are selected in round-robin fashion
    2. The environment offers a coalition to the agent
    3. The agent accepts or rejects the coalition
    4. Environment calculates delta = V(S U {i}) - V(S) and assigns it as reward to the agent
    if accept: reward = delta, if reject, reward = -delta
    5. Environment updates the coalition structure (state) of the agent
    6. Environment selects the next agent in round-robin fashion
    etc

    4. The env calculates the marginal contribution of the coalition (delta) - this is the reward for the agent
    --> THESE ARE NOT THE MARGINAL CONTRIB OF THE PLAYING AGENT! for Shapley it needs to be reallocated!! (problem)

   Works on Ray 2.6
'''

import os
import itertools
import gymnasium as gym
from gym.spaces import Box, Dict
from gym.utils import seeding
import numpy as np
from itertools import product
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict #To avoid empty policy
from itertools import permutations
import logging

#logging.basicConfig(filename='ray_info.log', level=logging.INFO, format='%(message)s')

#RLLIB divides the total number of steps (i.e. batch size) across the workers
NUM_CPUS  = 20 # os.cpu_count() #NUM_CPUS-1 = num_rollour _workers. to calculate the num_steps on each batch (for the cv learning)


# Define the characteristic function as a class
class CharacteristicFunction:
    def __init__(self, config_dict):
        self.mode  = config_dict.get('mode', 'closed_form')
        self.alpha = config_dict.get('alpha', 1)
        self.k     = config_dict.get('k', 1)
        # Non-superadditive game
        self.partition_values = {
            frozenset([0, 1]): 50,
            frozenset([0, 2]): 0,
            frozenset([1, 2]): 0,
            frozenset([0, 1, 2]): 10,
            frozenset([0]): 0,
            frozenset([1]): 0,
            frozenset([2]): 50
        }

    def value(self, coalition, distances, distance):
        if self.mode == 'closed_form':  # superadditive game
            return np.sum(coalition)**3
        elif self.mode == 'partition_form': #non-superadditive game
            # Convert the coalition array to a set for easier lookup
            agents_in_coalition = frozenset([i for i, x in enumerate(coalition) if x == 1])
            return self.partition_values.get(agents_in_coalition, 0)
        elif self.mode == 'ridesharing':
            # V(S, d, alpha) = k * sum(d) - alpha * |S|^2 #cost function: we want: more people, less cost (and reward is changed to -cost)
            #return self.k * np.sum(distances) - (self.alpha * np.sum(coalition)**2)
            #print('coalition:',coalition)
            #print('distances before:', distances)
            distances = [x for x in distances if x != 0]
            #print('distances later:', distances)
            #print('std dev of distances:', np.std(distances))
            #return self.k * np.std(distances) - (self.alpha * np.sum(coalition)**2) #std dev of distances
            #return self.k * np.std(distances) /(self.alpha * np.sum(coalition)**2) #std dev of distances
            if len(distances) == 1:
                return distance
            else:
                return (self.k * np.var(distances)/(self.alpha * np.sum(coalition))) #sign consistent



# Define the coalition formation environment
class ShapleyEnv(MultiAgentEnv):
    def __init__(self, config_dict):
        super().__init__()
        self.step_count            = 0
        self.num_agents            = config_dict.get('num_agents',2)
        self.batch_size            = config_dict.get('batch_size', 2000)       # for the CV learning - update the CV when a batch is full
        char_config_dict           = config_dict.get('char_func_dict',{})      # characteristic function config dict
        self.manual_distance_lst   = config_dict.get('manual_distances',None)  # for the curriculum learning
        self.cyclic_iter           = itertools.cycle(self.manual_distance_lst)
        self.char_func             = CharacteristicFunction(char_config_dict)
        self.agent_lst             = list(range(self.num_agents))              # [0, 1, 2, ..., num_agents-1]
        self._agent_ids            = set(self.agent_lst)                       # Required by RLLIB (key word- don't change)
        self.reward_dict           = {}
        self.current_coalitions    = {i: np.array([1 if i == j else 0 for j in range(self.num_agents)]) for i in range(self.num_agents)} #start on singletons - will be overriden later
        self.max_steps             = config_dict.get('max_steps',4000)
        self.shapley_values        = {agent: 0 for agent in self.agent_lst}    # for the shapley value

        self.accumulated_rewards = {agent: 0 for agent in self.agent_lst}     # To match RLLIB's per-agent output


        # SHAPLEY COMMENTED OUT FOR NOW
        #self.required_deltas = self.generate_required_deltas() #for the shapley value

        self.reset() # Resets time,Resets valid coals. Stores the current coalition for each agent. Selects current agent

        self.action_space      = gym.spaces.Discrete(2) #accept/reject
        self.observation_space = gym.spaces.Dict({
            'coalitions': gym.spaces.Box(low=0, high=1, shape=(2, self.num_agents), dtype=np.int32), #[current coal,proposed coal]
            'distances': gym.spaces.Box(low=0, high=10, shape=(2, self.num_agents), dtype=np.float32) #distance vectors
        })


    # RLLIB Methods to avoid the empty action space!
    def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {id: self.observation_space.sample() for id in agent_ids} # Observation space is the same for all agents

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
           agent_ids = self._agent_ids
        return {id: self.action_space.sample() for id in agent_ids} #Action space is a dict

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
           return False
        return all(self.action_space.contains(val) for id, val in x.items()) #Action space is a dict

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
           return False
        return all(self.observation_space.contains(val) for id, val in x.items()) # Observation space is the same for all agents

    def seed(self, seed=None):
        '''returns a seed for the env'''
        if seed is None:
            _, seed = seeding.np_random()
        return [seed]

    # For ridesharing
    def generate_valid_coalitions(self):
        '''# Generate all possible coalitions for each agent as list of 0's and 1's. 1 means the agent is in the coalition.'''
        self.valid_coalitions = {}
        for agent in range(self.num_agents):
            agent_coalitions = []  # Initialize as an empty list
            for coalition in product([0, 1], repeat=self.num_agents):
                if coalition[agent] == 1:  # Include only coalitions that contain the agent
                    if np.sum(coalition) > 0:  # Skip coalitions that are all zeros
                        agent_coalitions.append(np.array(coalition))
            self.valid_coalitions[agent] = agent_coalitions
        return self.valid_coalitions  # Used for testing

    def generate_required_deltas(self):
        ''' List of required Marginal Contribs (difference of coalitions - not values-) for each agent needed for Shapley
            Only works for Grand Coalitions, as we need to know the coalition structure in advance'''
        self.required_deltas = {agent: [] for agent in range(self.num_agents)}
        self.perms = list(permutations(self.agent_lst)) # Generate all permutations of agent indices
        for perm in self.perms:
            for idx, agent in enumerate(perm):
                coalition_without_agent = [0] * self.num_agents
                coalition_with_agent = [0] * self.num_agents
                for a in perm[:idx]:
                    coalition_without_agent[a] = 1
                for a in perm[:idx + 1]:
                    coalition_with_agent[a] = 1
                delta = (coalition_with_agent, coalition_without_agent)
                self.required_deltas[agent].append(delta)
        return self.required_deltas

    def generate_initial_distances(self):
        '''Generate an INITIAL random distance vector for each agent
           Used as a starting point for the curriculum learning. Only makes sense on the rnd generation
           Only used ONCE at the beginning of the training
           OR when distances have been all clipped
           Considers both manually passed list (for parallel envs) or rnd generated list (for num_workers = 1)
        '''
        if self.manual_distance_lst is not None:
           self.distance_lst = next(self.cyclic_iter) #lst of list - Passed manually from outside the class or in the curriculum
        else:
            dist = np.random.uniform(0.1, 0.5, self.num_agents).flatten().astype(np.float16) #generate first distance
            round_dist = [round(x, 2) for x in dist] #round
            self.distance_lst = round_dist

    def update_distances(self):
        '''If manual distance list -> move to next item on list
           If rnd generated list -> Add a small random number to each distance
           Legacy method from the rnd generated distances.
        '''
        if self.manual_distance_lst is not None:
            self.distance_lst = next(self.cyclic_iter) #lst of list - Passed manually from outside the class or in the curriculum
        else: #generate rnd distances
            add = np.random.uniform(0.01, 0.015, self.num_agents) # add a rnd number between 0.1 and 0.13 to each distance
            dist = [x + y for x, y in zip(self.distance_lst, add)]
            # Check if all distances are greater than 0.5, regenerate
            if all(d >= 0.5 for d in self.distance_lst):
                self.generate_initial_distances() # Generate a new random distance vector for each agent
            else: #check if any distance is greater than 0.5 clip it
                dist = np.clip(dist, 0, 0.5)
                round_dist        = [round(x, 2) for x in dist] #round
                self.distance_lst = round_dist
                # Save to a logging text file - note that RLLIB resets the env several times before training for real
        #log_entry = f"Step: {self.step_count}, Distances: {self.distance_lst}"
        #logging.info(log_entry)
        #print('Step:', self.step_count, 'Distances:', self.distance_lst)


    def _get_observation(self):
        ''' After an agent has been selected -
            Generate a new coalition proposal for the current agent - randomly without replacement - from the list of valid coalitions
            And delete it from the list of valid coalitions for the current agent
        '''
        # Check if there are valid coalitions available for the current agent
        if len(self.valid_coalitions[self.current_agent]) == 0:
            # No valid coalitions left, return a 'null' observation in the same format
               return {
                'coalitions': np.zeros((2, self.num_agents), dtype=int),
                'distances': np.zeros((2, self.num_agents), dtype=float)
                 }
        ''''
        # Check if there's only one coalition left for the current agent - make sure is not the same as it current
        if len(self.valid_coalitions[self.current_agent]) == 1: #EXPERIMENTAL
            # If the last coalition is the same as the current coalition, represent it as an empty coalition and the coalition that's left
            if np.array_equal(self.valid_coalitions[self.current_agent][0], self.current_coalitions[self.current_agent]):
               self.new_coalition = self.valid_coalitions[self.current_agent][0]
               # delete the coalition from the list of valid coalitions for the current agent
               self.valid_coalitions[self.current_agent] = [coal for coal in self.valid_coalitions[self.current_agent] if not np.array_equal(coal, self.new_coalition)]
               return {
            'coalitions': np.vstack([np.zeros(self.num_agents, dtype=int), self.valid_coalitions[self.current_agent][0]]),
            'distances': np.vstack([self.distance_lst * np.zeros(self.num_agents, dtype=int), self.distance_lst * self.valid_coalitions[self.current_agent][0]])
        }
        '''
        # Generate a new coalition proposal - avoid proposing the current coalition and the empty coalition
        '''
        while True:
            coalition_id       = np.random.choice(len(self.valid_coalitions[self.current_agent])) #avoid proposing the current coalition
            self.new_coalition = self.valid_coalitions[self.current_agent][coalition_id]
            if not np.array_equal(self.new_coalition, self.current_coalitions[self.current_agent]):
                break
        '''
        #COALITIONS
        # Last coalition might be repeated, but that's ok, better than an infinite loop or obscure coding.
        coalition_id       = np.random.choice(len(self.valid_coalitions[self.current_agent])) #avoid proposing the current coalition
        self.new_coalition = self.valid_coalitions[self.current_agent][coalition_id]
        # Proposed coalitions can't repeat - Remove the proposed coalition from the list of valid coalitions for the current agent
        self.valid_coalitions[self.current_agent] = [coal for coal in self.valid_coalitions[self.current_agent] if not np.array_equal(coal, self.new_coalition)]
        # DISTANCES
        # Update distances based on Curriculum Learning
        if self.step_count == np.round((self.batch_size/NUM_CPUS-1),0): # CPUS-1 = num_rollout_workers. Approx nbr of steps per one batch - update only when batch is full. RLLIB divides the steps by the num_workers
           self.update_distances()    # Either add randomness to distances - or select next distance from the curriculum list

        return {
            'coalitions': np.vstack([self.current_coalitions[self.current_agent], self.new_coalition]),
            'distances': np.vstack([self.distance_lst*self.current_coalitions[self.current_agent] , self.distance_lst*self.new_coalition])
        }


    def _calculate_reward(self, action,current_agent = None, current_coal = None, new_coal = None, distance_lst = None, mode = None):
        '''Reward for the playing agent. Coaltion value is calculated on-the-fly
           Reward was shaped to encourage agent to choose the coalition with more Mg Contribution
           Accepts coalitions and mode from outside the class used to test the env
        '''
        # Allowing to pass the observations and mode from outside the class - for inference
        if all(var is not None for var in [current_agent,current_coal, new_coal, distance_lst, mode]):
           self.current_agent = current_agent
           prev_distances     = distance_lst*current_coal
           new_distances      = distance_lst*new_coal
           distance           = distance_lst[self.current_agent] #distance of the current agent
           prev_value         = self.char_func.value(current_coal, prev_distances,distance)
           new_value          = self.char_func.value(new_coal,new_distances,distance)
        else: # take variables from env  - for training
            # Calculate the value of the current coalition
            prev_distances = self.distance_lst*self.current_coalitions[self.current_agent]
            new_distances  = self.distance_lst*self.new_coalition
            distance       = self.distance_lst[self.current_agent]
            prev_value     = self.char_func.value(self.current_coalitions[self.current_agent], prev_distances,distance)
            new_value      = self.char_func.value(self.new_coalition,new_distances,distance )
        # Calculate Reward  - Whether we are on a value game or a cost game - "delta or PROPOSED coalition"
        multiplier = 1
        if self.char_func.mode == 'ridesharing': multiplier = -1 # cost game: - new_value + prev_value. Cost = -Value. We want to join coalition with min cost
        delta_value =  ((multiplier)*new_value - (multiplier)*prev_value) # for ridesharing

        # Assign Reward depending on action
        if delta_value <0: # if proposed coalition has less value (or more cost) - stay in old
            if action ==1:
               self.reward_dict[self.current_agent] = -100 #delta_value *100 # big negative reward  #-10
            else: #if we stay where we are - repeat the reward from the current state
                self.reward_dict[self.current_agent] = 10 #-delta_value*100 #big positive reward #10
                # if it is the first time on this state, and it gets rejected, dict is empty and reward = 0. This is buggy.
        else: #delta new coalition >0: new coal has more value (or less cost) --> accept moving to new coalition
            if action ==1:
               self.reward_dict[self.current_agent] = 10 #delta_value #1
            else: #if we stay where we are - repeat the reward from the current state
                self.reward_dict[self.current_agent] =  -100 # -delta_value #-1
                # if it is the first time on this state, and it gets rejected, dict is empty and reward = 0. This is buggy.

        # Shapley for each agent - only meaningful for Grand Coalitions - meaningless if there is a coalition structure
        delta_coalitions = (self.new_coalition.tolist(), self.current_coalitions[self.current_agent].tolist()) #which coals are involved

        # SHAPLEY NOT USED FOR NOW
        #self._calculate_shapley(delta_coalitions, delta_value)

        return self.reward_dict[self.current_agent]

    def _update_coalitions(self, action):
        '''If coalition accepted - Update agent's current coalition (i.e. state)'''
        if action == 1:
           self.current_coalitions[self.current_agent] = self.new_coalition  # Update the current coalition for this agent
           #print('Current coalition - acting agent:', self.current_coalitions[self.current_agent])

    def _calculate_shapley(self, delta_coalitions, delta_value):
        '''Assign the delta value to the appropriate agent'''
        # Whether all needed deltas have been already assigned (as the method calculates more than needed)
        all_deltas = all(len(deltas) == 0 for deltas in self.required_deltas.values())
        if not all_deltas:
           for agent in range(self.num_agents):
               if delta_coalitions in self.required_deltas[agent]:
                self.shapley_values[agent] += delta_value

                #print('Agent:', agent)
                #print('delta_coalitions:', delta_coalitions)
                #print('Delta value:', delta_value)
                self.required_deltas[agent].remove(delta_coalitions)
                #print('shapleys on the fly:', self.shapley_values)

    def calculate_final_shapley(self):
        for agent in self.shapley_values:
            self.shapley_values[agent] /= len(self.perms)
        #print('Final Shapley values:', self.shapley_values)
        return self.shapley_values

    def _select_playing_agent(self):
        '''Selects the next agent to play
           agents selected in round-robin fashion is more efficient than random selection
        '''
        # Select agent for the incoming round
        self.current_agent = (self.current_agent + 1) % self.num_agents
        # If no more available coalitions for the current agent - Terminate the agent
        if len(self.valid_coalitions[self.current_agent]) == 0:
            self.terminated_dict[self.current_agent] = True
            self._select_playing_agent() #pick another agent
        # Skip terminated agents
        elif self.terminated_dict[self.current_agent]:
            if self.terminated_dict['__all__']:   # All agents are terminated, stop the recursion
                return

    def set_first_coalitions(self):
        '''Set the first 'current' state for each agent'''
        # Set current coalition as the Singleton for each agent - Don't remove it as it needs to be proposed as new state later.
        self.current_coalitions = {}
        for agent in range(self.num_agents):
            coalition_id                   = np.random.choice(len(self.valid_coalitions[agent]))
            self.current_coalitions[agent] = self.valid_coalitions[agent][coalition_id]

    ##################################################################
    # RESET
    ##################################################################
    def reset(self, *, seed=None, options=None):
        ''' Reset and start a new episode
            Number of resets = number of episodes per iteration
        '''

        self.truncated_dict             = {agent: False for agent in self.agent_lst}   # whether they have been placed in a coalition
        self.truncated_dict['__all__']  = False
        self.terminated_dict            = self.truncated_dict.copy()
        self.truncated_dict             = {agent: False for agent in self.agent_lst}   # whether they have been placed in a coalition
        self.truncated_dict['__all__']  = False
        self.terminated_dict            = self.truncated_dict.copy()
        self.reward_dict                = {agent: 0 for agent in self.agent_lst}        # each episode is an independent trial
        self.accumulated_rewards = {agent: 0 for agent in self.agent_lst}     # Reset accumulated rewards for a new episode - to match RLLIB's per-agent output

        #COALITIONS - Initial coalitions for each agent.
        self.generate_valid_coalitions()      # populates self.valid_coalitions. This list shrinks as coalitions are proposed by the env.
        self.set_first_coalitions()           # set the first 'current' state for each agent

        #DISTANCES - Initial distance vector for each agent (manual or rnd)
        if self.step_count == 0: #first time (only once after initializing the class - not after every reset)
           self.generate_initial_distances()   # includes manual lst or rnd

        # Select the first agent to play
        #self.current_agent    = 0 # next agent in play
        # Choose the next agent
        self.current_agent = np.random.randint(0, self.num_agents - 1)
        self.next_observation = self._get_observation() #obs for next agent in play {[coalitions][distances]}
        return  {self.current_agent:self.next_observation}, {}


    ##################################################################
    # STEP
    ##################################################################
    def step(self, action_dict):
        self.step_count += 1                             # for the curriculum learning. Max num steps = batch_size/NUM_CPUS-1

        self.current_agent = list(action_dict.keys())[0] # this should be the same as before
        action = action_dict[self.current_agent]         # action is a dict

        self._calculate_reward(action)                   # reward needs to be calculated before updating the coalition

        # TO TEST ALL COALITIONS PRESENT
        #print({'current agent': self.current_agent,'action':action, 'Reward': self.reward_dict[self.current_agent]})

        self._update_coalitions(action) # If the agent accepts the proposal - Update the coalition agent is in. Else pass

        # Prepare next obs: generate a new coalition proposal for the next agent
        # Termination conditions
        #if self.step_count >= self.max_steps:  # this doesn't work with CV learning
        #   self.truncated_dict['__all__'] = True # sends to reset
        #   self.step_count = 0
           #print('ALL AGENTS TRUNCATED!!')

        if all(len(self.valid_coalitions[agent]) == 0 for agent in range(self.num_agents)):
            self.terminated_dict['__all__'] = True
           # print('ALL AGENTS TERMINATED!!')
        else:
             # Choose the next agent
             self._select_playing_agent()
             # Obs for next agent in play {[coalitions][distances]}
             self.next_observation = self._get_observation() #CV learning - distances are updated at the end of the batch

        # Update the reward only for the active agent
        self.accumulated_rewards[self.current_agent] += self.reward_dict[self.current_agent] # calculate the accumulated reward for the active agent

        return {self.current_agent:self.next_observation}, self.reward_dict , self.terminated_dict, self.truncated_dict,{}


    def render(self,acting_agent):
        # Only used for the testing of the Env
        print('step_count:', self.step_count)
        print("Acting agent:", acting_agent)
        print('Reward - acting agent:', self.reward_dict[acting_agent])
        print("Next agent:", self.current_agent)
        print("Next agent - obs [current, proposed]:", self.next_observation)



#####################################################################
# TEST THE ENVIRONMENT
#####################################################################


def test_env(custom_env_config):
    # Test the environment
    # Just pass a dummy action so we can run the step method


    env = ShapleyEnv(custom_env_config)
    reward_accumulators = {i: 0 for i in range(env.num_agents)}
    num_episodes = 1

    for episode in range(num_episodes):
        obs = env.reset()
        #print("Initial agent:", env.current_agent)
        acting_agent = env.current_agent
        #print("Initial obs [current, proposed]:", obs)
        for i in range(env.num_agents**4):  # 3 agents * 3 turns
            action_dict = {acting_agent :1} #just pass an action so we can run the step method
            new_obs, reward_dict, terminated, truncated, info = env.step(action_dict)
            #env.render(acting_agent) #prints the current state of the environment
            reward_accumulators[acting_agent] += reward_dict[acting_agent]
            #print('Reward accumulators:', reward_accumulators)
            acting_agent = env.current_agent

    average_rewards = {agent: total_reward / num_episodes for agent, total_reward in reward_accumulators.items()}
    print('avg rewards per agent:', average_rewards)
    #print('Initial required deltas:', env.required_deltas)
    print('Calculated Shapley values:', env.calculate_final_shapley())



##################################
# RUNNER
##################################
if __name__ == '__main__':

    char_func_dict = {
        'mode'  : 'ridesharing',# 'closed_form', #'partition_form', #'closed_form','ridesharing'
        'alpha' : 1,
        'k'     : 7,
    }

    custom_env_config = {
        'num_agents'     : 3,
        'char_func_dict' : char_func_dict,
        'max_steps'      : 4000,
        'batch_size'     : 1000 # for the CV learning - one update per batch size
          }

    test_env(custom_env_config)
