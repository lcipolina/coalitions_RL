import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

''' Attempt at replicating:
http://researchers.lille.inria.fr/~lazaric/Webpage/Publications_files/munoz2007learning.pdf'''

'''  COULDNT REPLICATE THE PAPER - Agent's don't reach the max return and they don't form the optimal coalition

Coalitions should form as follows:
- 1 cooker and 2 helpers can make a cake worth 10 points
- 4 cookers alone can make a cake worth 10 points


Implements metod by Cote et al for Coalition Formation
It's just a simple TABULAR Qlearning with multiple agents

Let's consider a simplified example where there are three possible coalitions: Coalition 0, Coalition 1, and Coalition 2.
Rows represent the coalition the agent is currently in.
Each agent can choose an action to move to any other coalition or stay in the same one.

Agents have "types" (e.g., 'cooker' or 'helper') that determine their rewards based on the coalition they are in.
Each type of agent has one Q-table like this

          +-----------+-----------+-----------+
          | Move to 0 | Move to 1 | Move to 2 |
+---------+-----------+-----------+-----------+
| In 0    |   Q[0,0]  |   Q[0,1]  |   Q[0,2]  |
+---------+-----------+-----------+-----------+
| In 1    |   Q[1,0]  |   Q[1,1]  |   Q[1,2]  |
+---------+-----------+-----------+-----------+
| In 2    |   Q[2,0]  |   Q[2,1]  |   Q[2,2]  |
+---------+-----------+-----------+-----------+


'''

#==============================================================================
# Characteristic Function Class
#==============================================================================
class CharacteristicFunction:
    def __init__(self, value_function):
        """
        Initialize the characteristic function with a user-defined coalition value function.
        :param value_function: Function that calculates the value of a coalition (list of agent types).
        """
        self.value_function = value_function

    def calculate_coalition_value(self, coalition):
        """
        Calculate the total value of the coalition using the defined value function.
        :param coalition: List of agent types in the coalition.
        :return: Value of the coalition.
        """
        return self.value_function(coalition)

    def calculate_marginal_contribution(self, coalition, agent):
        """ This is to assign rewards to agents based on their marginal contribution to the coalition.
        Calculate the marginal contribution of a single agent to a coalition.
        :param coalition: List of agent types representing the current coalition.
        :param agent: The agent whose marginal contribution is to be calculated.
        :return: The agent's marginal contribution to the coalition.
        """
        coalition_without_agent = [a for a in coalition if a != agent]
        value_without_agent = self.calculate_coalition_value(coalition_without_agent)
        value_with_agent = self.calculate_coalition_value(coalition)

        return value_with_agent - value_without_agent


# Value function for coalition
def value_function(coalition):
    '''The characteristic function captures the synergy between agents by assigning a higher reward for certain agent combinations
    (e.g., 1 cooker and 2 helpers yield a "cake" worth 10 points).
    '''
    num_cookers = sum(1 for agent in coalition if agent == 'cooker')
    num_helpers = sum(1 for agent in coalition if agent == 'helper')
    value = 0

    # Rule (i): 1 cooker and 2 helpers can make a cake worth 10 points
    while num_cookers >= 1 and num_helpers >= 2:
        value += 10  # Cake value
        num_cookers -= 1  # count the number of cookers
        num_helpers -= 2  # count the number of helpers

    # Rule (ii): 4 cookers alone can make a cake worth 10 points
    while num_cookers >= 4:
        value += 10  # Cake value
        num_cookers -= 4

    # Rule (iii): A helper alone can make a cookie worth 1 point
    value += num_helpers  # Each remaining helper can make a cookie

    # Rule (iv): A cooker alone can do nothing (no additional points)

    return value


#==============================================================================
# Gymnasium Environment Class
#==============================================================================
class CoalitionFormationEnv(gym.Env):
    def __init__(self, num_agents=6, agent_types=None, policy=None):
        """
        Initialize the Gymnasium environment.

        :param num_agents: Total number of agents in the environment.
        :param agent_types: List of agent types (optional).
        :param policy: An instance of the policy/learning agent class responsible for managing Q-values. Needed to check termination conditions (based on the difference of Q values)
        """
        super(CoalitionFormationEnv, self).__init__()
        self.characteristic_function = CharacteristicFunction(value_function)
        self.num_agents = num_agents
        self.agent_types = agent_types if agent_types else ['cooker'] * (num_agents // 2) + ['helper'] * (num_agents // 2)
        self.agents = list(range(num_agents))
        self.coalition_structures = [[] for _ in range(num_agents // 2 + 1)]
        self.agent_coalitions = np.zeros(self.num_agents, dtype=int)
        self.rewards = {}

        self.policy = policy  # Reference to the multi-agent policy instance

        self.action_space = spaces.Discrete(len(self.coalition_structures))
        self.observation_space = spaces.Dict({
            "old_coalition": spaces.Discrete(len(self.coalition_structures)),
            "new_coalition": spaces.Discrete(len(self.coalition_structures)),
            "agent_coalitions": spaces.MultiDiscrete([len(self.coalition_structures)] * self.num_agents),
            "next_agent": spaces.Discrete(self.num_agents),
            "next_agent_type": spaces.Discrete(len(set(self.agent_types)))
        })

        # Call reset to initialize the environment structures
        self.reset()


    def reset(self, seed=None, options=None):
        '''
            The reset method initializes the environment to a consistent state at the start of a new episode:
            1. Reinitialize Coalition Structures:
            - Clears all coalition structures to ensure no leftover data from prior episodes.

            2. Randomly Assign Agents:
            - Randomly assigns each agent to an initial coalition for diverse starting configurations.
            - This provides different scenarios for exploration and learning.

            3. Return Initial State:
            - Returns an array representing agent-coalition assignments to give a baseline state before agent policy-based decisions.
        '''

        super().reset(seed=seed)  # New gymnasium requirement - Call parent class's reset function to ensure proper initialization

        # Dummy state - Initialize all agents to be part of coalition 0 initially
        self.agent_coalitions = np.zeros(self.num_agents, dtype=int)

        # Create empty coalition structures, with each coalition represented as a list of agents
        # Adding one extra coalition slot for flexibility
        self.coalition_structures = [[] for _ in range(self.num_agents // 2 + 1)]

        # Randomly assign each agent to an initial coalition
        for agent in range(self.num_agents):
            coalition = random.choice(range(len(self.coalition_structures)))     # Randomly choose a coalition index from the available coalition structures
            self.coalition_structures[coalition].append(self.agent_types[agent]) # Append the agent type to the chosen coalition structure
            self.agent_coalitions[agent] = coalition                             # Record the agent's coalition assignment in the state array

        # Initialize the rewards dictionary for all agents
        self.rewards = {agent_id: 0 for agent_id in range(self.num_agents)}
        # Construct the initial observation with plausible values
        observation = self._get_observation(old_coalition=None, new_coalition=None)


        return observation, {}


    def step(self, action_dict):
        """
        Perform one step in the environment by moving an agent to a specified coalition.

        :param agent_id: Index of the agent performing the action.
        :param action: Index of the coalition to join (new coalition).

        :return: Tuple containing the updated state (agent-coalition assignments), reward, done flag, truncated flag, and an empty info dictionary.
        """

        # Process action
        agent_id = action_dict["agent_id"]
        action = action_dict["action"]
        old_coalition = self.agent_coalitions[agent_id]    # Get the current and target coalition index of the agent
        new_coalition = action                             # The target coalition index specified by the agent's action

        # If the agent is moving to a new coalition, update the coalition structures
        if old_coalition != new_coalition:
            self.coalition_structures[old_coalition].remove(self.agent_types[agent_id]) # Remove the agent's type from the old coalition
            self.coalition_structures[new_coalition].append(self.agent_types[agent_id]) # Add the agent's type to the new coalition
            self.agent_coalitions[agent_id] = new_coalition                             # Update the agent's coalition assignment in the state array

        # Calculate reward
        current_coalition      = self.coalition_structures[new_coalition] # Retrieve the agent's current coalition to calculate the marginal contribution
        reward                 = self.characteristic_function.calculate_marginal_contribution(current_coalition, self.agent_types[agent_id]) # Calculate the agent's reward based on its marginal contribution to the coalition
        self.rewards[agent_id] = reward                                  # Update the agent's specific reward in the rewards dictionary

        # Check for learning termination through the policy's termination condition
        terminated = self.policy.check_termination()
        truncated = False  # Optionally set a truncated flag if you have other criteria to stop the episode early

        # Get the updated observation for the next agent
        observation = self._get_observation(old_coalition, new_coalition)

        return  observation,self.rewards, terminated, truncated, {}

    def _get_observation(self, old_coalition, new_coalition):
        """
        Construct the observation dictionary to pass to the policy.
        :param old_coalition: The previous coalition index of the acting agent.
        :param new_coalition: The new coalition index of the acting agent.
        :return: An observation dictionary containing the old/new coalition, next agent ID, and type.
        """
        if old_coalition is None:
            old_coalition = random.choice(range(len(self.coalition_structures)))
        if new_coalition is None:
            new_coalition = random.choice(range(len(self.coalition_structures)))

        next_agent_id = random.choice(self.agents)
        next_agent_type = self.agent_types[next_agent_id]

        return {
            "old_coalition": old_coalition,
            "new_coalition": new_coalition,
            "agent_coalitions": self.agent_coalitions,
            "next_agent": next_agent_id,
            "next_agent_type": self.agent_types.index(next_agent_type)
        }


    def render(self, mode='human'):
        print("Coalition Structures:")
        for idx, coalition in enumerate(self.coalition_structures):
            print(f"Coalition {idx}: {coalition}")


#================================================================================================
#+--------------------------------- Q-Learning Agent Class ----------------------------------+
#================================================================================================
class MultiAgentQLearning:
    def __init__(self, agent_types, num_actions, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.1, decay_rate=0.99):
        """
        Initialize the Q-learning agent structure for each agent type.

        :param agent_types: List of agent types.
        :param num_actions: Total number of available actions.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Initial exploration rate.
        :param min_epsilon: Minimum exploration rate.
        :param decay_rate: Rate at which to decay the epsilon value.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.num_actions = num_actions # Number of possible actions

        # Initialize Q-tables for each unique agent type
        self.q_tables = {atype: np.zeros((num_actions, num_actions)) for atype in set(agent_types)}

        # For the termination condition
        self.no_improvement_counter = {atype: 0 for atype in set(agent_types)}
        self.improvement_threshold = 0.01
        self.patience = 5

    def update_q_value(self, agent_type, old_coalition, action, reward, new_coalition):
        """
        Update the Q-table based on the received reward.

        :param agent_type: Type of agent (e.g., 'cooker' or 'helper').
        :param old_coalition: Previous coalition index.
        :param action: Selected action (index of the coalition to move to).
        :param reward: Reward received for the action.
        :param new_coalition: Index of the coalition the agent moved to.
        """

        # Retrieve the current Q-value and calculate the expected future value
        old_value = self.q_tables[agent_type][old_coalition, action]
        future_value = np.max(self.q_tables[agent_type][new_coalition])
        # Apply the Bellman equation to update the Q-value: Q(s, a) = (1 - α) * Q(s, a) + α * (R + γ * max(Q(s', a')))
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_value)

        # Update the Q-table with the new value
        self.q_tables[agent_type][old_coalition, action] = new_value

        # Check for significant improvement
        if abs(new_value - old_value) < self.improvement_threshold:
            self.no_improvement_counter[agent_type] += 1
        else:
            self.no_improvement_counter[agent_type] = 0



    def select_action(self, agent_type, current_coalition):
        """
        Select an action for the given agent type based on an epsilon-greedy policy.

        :param agent_type: Type of agent (e.g., 'cooker' or 'helper').
        :param current_coalition: Index of the current coalition.

        :return: The selected action index.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            return np.argmax(self.q_tables[agent_type][current_coalition])

    def decay_epsilon(self):
        """
        Decay the epsilon value over time to gradually shift from exploration to exploitation.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def check_termination(self):
        """
        Check if the learning process should terminate due to lack of significant improvement.
        :return: Boolean indicating whether learning should stop.
        """
        # If any agent type has exceeded the patience limit for no improvement, terminate
        return any(count >= self.patience for count in self.no_improvement_counter.values())


# Function to calculate discounted sum of rewards
def discounted_returns(rewards, gamma=0.9):
    discounted = np.zeros(len(rewards))
    cumulative_sum = 0
    # Bellman returns: Gt = Rt + γ * Gt+1
    for t in reversed(range(len(rewards))):
        cumulative_sum = rewards[t] + gamma * cumulative_sum
        discounted[t] = cumulative_sum
    return discounted

def plot_cumulative_returns(returns_per_agent, num_episodes):
    """
    Plot the cumulative discounted returns per agent.
    :param returns_per_agent: Dictionary mapping agent IDs to their cumulative returns over episodes.
    :param num_episodes: Number of episodes for which to plot the data.
    """
    plt.figure(figsize=(10, 6))
    for agent_id, returns in returns_per_agent.items():
        plt.plot(range(num_episodes), returns, label=f"Agent {agent_id}")

    plt.xlabel("Episode")
    plt.ylabel("Discounted Returns")
    plt.legend()
    plt.title("Cumulative Discounted Returns per Agent")
    #plt.show()
    plt.savefig('cumulative_returns.png')



#================================================================================================
#+--------------------------------- Simulation and Training ----------------------------------+
#================================================================================================

if __name__ == "__main__":
    # Initialize the environment and policy
    num_agents     = 6 #More than 4, run it on Colab  - needs to be even number - ideally 2 cookers and 4 helpers - but with more than 4 agents it won't run on my PC
    agent_types    = ['cooker'] * (num_agents // 2) + ['helper'] * (num_agents // 2)
    num_coalitions = (num_agents // 2) + 1
    num_actions    = num_coalitions  # Each agent can move to any of the coalition indices
    policy         = MultiAgentQLearning(agent_types, num_actions)

    # Initialize returns storage
    returns_per_agent = {agent_id: [] for agent_id in range(num_agents)}

    # Pass the policy instance to the environment
    env = CoalitionFormationEnv(num_agents=num_agents, agent_types=agent_types, policy=policy)

    # Simulation and Training Loop
    num_episodes = 13  #70 to converge
    gamma        = 0.9  # Discount factor

    for episode in range(num_episodes):
        # Reset the environment and obtain the initial observation
        observation, _ = env.reset()
        terminated = False

        # Initialize cumulative rewards per agent for this episode
        episode_rewards = {agent_id: [] for agent_id in range(num_agents)}

        while not terminated:
            # Extract the next agent and other necessary information from the observation
            next_agent      = observation["next_agent"]
            next_agent_type = agent_types[next_agent]
            old_coalition   = observation["old_coalition"]

            # Use the policy to select the next action
            action = policy.select_action(next_agent_type, old_coalition)
            action_dict = {"agent_id": next_agent, "action": action} # Create an action dictionary to pass to the environment's step function

            # Step through the environment with the selected action and receive updated information
            observation, rewards, terminated, truncated, _ = env.step(action_dict)

            # Retrieve the reward for the last acting agent (next_agent is now the current agent)
            reward = rewards[next_agent]
            episode_rewards[next_agent].append(reward)

            # Update the Q-table for the last acting agent
            new_coalition = observation["new_coalition"]
            policy.update_q_value(next_agent_type, old_coalition, action, reward, new_coalition)


        # Calculate discounted returns for each agent
        for agent_id, rewards in episode_rewards.items():
            returns = discounted_returns(rewards, gamma)
            returns_per_agent[agent_id].append(np.sum(returns))


        policy.decay_epsilon()  # Decay epsilon after each episode

        env.render()
        print("Episode", episode, "completed.")


    # Final results
    print("Returns per agent:", {k: np.sum(v) for k, v in episode_rewards.items()})
    # Plot the cumulative returns per agent
    plot_cumulative_returns(returns_per_agent, num_episodes)
