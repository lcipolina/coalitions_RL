'''Implements metod by Cote et al for Coalition Formation - It's just a simple Qlearning
   Used in my paper
'''




import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class TabularQLearning:
    def __init__(self, num_states, num_actions, alpha=0.01, gamma=0.5, epsilon=0.1):
        self.q_table = np.zeros((num_states, num_actions))  # Initialize Q-table
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state_index, evaluate=False):
        # If evaluating and the state_index is out of bounds, choose a random action - otherwise Q table doesnt have this obs
        # if evaluate and (state_index >= self.q_table.shape[0] or state_index < 0):
        #    return np.random.randint(0, self.q_table.shape[1])

        if evaluate :
           return np.random.randint(0, self.q_table.shape[1])

        # Existing epsilon-greedy policy for action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])  # Explore
        else:
            # Ensure state_index is within bounds for exploitation
            state_index = max(0, min(state_index, self.q_table.shape[0] - 1))
            return np.argmax(self.q_table[state_index])         # Exploit


    def update_q_table(self, state_index, action, reward, next_state_index):
        # Bellman equation for Q-learning update
        best_next_action = np.argmax(self.q_table[next_state_index])                        # Best action at next state
        td_target = reward + self.gamma * self.q_table[next_state_index][best_next_action]  # Target for TD error
        td_error = td_target - self.q_table[state_index][action]                            # Temporal Difference error
        self.q_table[state_index][action] += self.alpha * td_error  # Update Q-value


class SimpleProductionEnv(gym.Env):
    def __init__(self, num_cookers, num_helpers):
        super(SimpleProductionEnv, self).__init__()
        self.num_cookers = num_cookers
        self.num_helpers = num_helpers
        self.action_space = spaces.Discrete(4)  # Four actions: 0-make cake with helpers, 1-make cake with cookers, 2-make cookie, 3-no action
        self.observation_space = spaces.MultiDiscrete([num_cookers + 1, num_helpers + 1])  # Observations are counts of available cookers and helpers

    def reset(self):
        self.available_cookers = self.num_cookers
        self.available_helpers = self.num_helpers
        return self._get_state()

    def _get_state(self):
        return (self.available_cookers, self.available_helpers)

    def step(self, action):
        reward = 0
        done = False

        # TODO: separate function for reward calculation
        # TODO: As per the paper, reward for each agent is the marginal contribution of the coalition to the total reward
        # currently, the agent receives as reward the total reward of the coalition
        # Need to calculate the marginal contribution of the coalition to the total reward
        # Need to assign rewards to each agent in the coalition
        # it seems like we need a dictionary to store the rewards of each agent in the coalition
        # and then calculate the marginal contribution of the coalition to the total reward

        if action == 0 and self.available_cookers >= 1 and self.available_helpers >= 2:  # Make cake with helpers
            self.available_cookers -= 1
            self.available_helpers -= 2
            reward = 10
        elif action == 1 and self.available_cookers >= 4:  # Make cake with cookers
            self.available_cookers -= 4
            reward = 10
        elif action == 2 and self.available_helpers >= 1:  # Make cookie
            self.available_helpers -= 1
            reward = 1

        # TODO: have a separate function for the characteristic function
            # A cake can be made by either 1 cooker and 2 helpers or 4 cookers alone.
            # A cookie can be made by 1 helper alone. Given these rules, the condition to check for the end of an episode (done) should ensure that:
            # It's not possible to form another cake or cookie with the remaining agents.
            # No more productive teams can be formed
        if not ((self.available_cookers >= 1 and self.available_helpers >= 2) or  # Can still form a cake with 1 cooker and 2 helpers
            (self.available_cookers >= 4) or  # Can still form a cake with 4 cookers
            (self.available_helpers >= 1)):  # Can still make a cookie with 1 helper
            done = True  # No further productive teams can be formed

        return self._get_state(), reward, done, {}

def train_agent(env, num_episodes=100):
    num_states = (env.num_cookers + 1) * (env.num_helpers + 1)  # Assuming a discrete state space where each state is unique
    num_actions = env.action_space.n
    agent = TabularQLearning(num_states, num_actions, alpha=0.01, gamma=0.99, epsilon=0.1)

    cumulative_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # The environment directly maps states to indices for simplicity
            state_index = state[0] * (env.num_helpers + 1) + state[1]
            action = agent.choose_action(state_index)
            next_state, reward, done, _ = env.step(action)
            next_state_index = next_state[0] * (env.num_helpers + 1) + next_state[1]
            agent.update_q_table(state_index, action, reward, next_state_index)
            total_reward += reward
            state = next_state

        cumulative_rewards.append(total_reward)

    return agent, cumulative_rewards

def evaluate_agent(env, agent, num_episodes=100, familiar_ratio=0.5, evaluate=True):
    total_reward = 0
    num_familiar_episodes = int(num_episodes * familiar_ratio)
    num_unfamiliar_episodes = num_episodes - num_familiar_episodes

    # Evaluate on familiar states
    for _ in range(num_familiar_episodes):
        state = env.reset()
        done = False
        while not done:
            state_index = state[0] * (env.num_helpers + 1) + state[1]
            action = agent.choose_action(state_index)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

    # Evaluate on unfamiliar states
    for _ in range(num_unfamiliar_episodes):
        # Generate an unfamiliar state by altering the number of cooks or helpers
        # This example assumes you have a method to set the state directly in the env, which may require modification of the env class
        unfamiliar_cookers = np.random.randint(1, env.num_cookers + 3)  # Assuming a range that can go beyond the training setup
        unfamiliar_helpers = np.random.randint(1, env.num_helpers + 3)  # Similarly, assuming a range beyond the training setup
        env.available_cookers = unfamiliar_cookers
        env.available_helpers = unfamiliar_helpers
        state = (unfamiliar_cookers, unfamiliar_helpers)

        done = False
        while not done:
            state_index = state[0] * (env.num_helpers + 1) + state[1]  # Might need adjustment for direct state setting
            action = agent.choose_action(state_index, evaluate=evaluate)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

    average_reward = total_reward / num_episodes
    return average_reward

def get_optimal_action(available_cookers, available_helpers):
    if available_cookers >= 1 and available_helpers >= 2:
        return 0  # Action 0: Make a cake with helpers
    elif available_cookers >= 4:
        return 1  # Action 1: Make a cake with cookers
    elif available_helpers >= 1:
        return 2  # Action 2: Make a cookie
    else:
        return 3  # Action 3: No action possible

def evaluate_accuracy(env, agent, num_episodes=100):
    correct_actions = 0
    total_actions = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            optimal_action = get_optimal_action(env.available_cookers, env.available_helpers)
            state_index = state[0] * (env.num_helpers + 1) + state[1]
            chosen_action = agent.choose_action(state_index, evaluate=True)

            if chosen_action == optimal_action:
                correct_actions += 1
            total_actions += 1

            _, _, done, _ = env.step(chosen_action)
            state = env._get_state()

    accuracy = correct_actions / total_actions if total_actions > 0 else 0
    return accuracy

def run_evaluation_across_familiar_ratios(env, agent, num_episodes=100, familiar_ratios=[0.0, 0.25, 0.5, 0.75, 1.0]):
    results = []
    for ratio in familiar_ratios:
        average_reward = evaluate_agent(env, agent, num_episodes, familiar_ratio=ratio)
        results.append((ratio, average_reward))
    return results

import matplotlib.pyplot as plt

def plot_familiar_ratios_vs_rewards(results):
    familiar_ratios, average_rewards = zip(*results)
    plt.plot(familiar_ratios, average_rewards, marker='o', linestyle='-')
    plt.xlabel('Coalitions Seen During Training (%)')
    plt.ylabel('Accurate Actions (%)')
    plt.title('Agent Performance vs. Generalization')
    plt.grid(True)
    plt.xticks(familiar_ratios)  # Ensure there's a tick for each tested ratio
    plt.show()


def plot_rewards(cumulative_rewards):
    """
    Plot the cumulative rewards per episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards, label='Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_moving_average_rewards(cumulative_rewards, window_size=50):
    """
    Plot a moving average of the cumulative rewards for a smoother visualization of the learning progress.
    """
    moving_avg = np.convolve(cumulative_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f'Moving Average (window size = {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Moving Average of Cumulative Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()


### Run the training process and plot the learning progress
if __name__ == "__main__":
    # Initialize the environment
    num_cookers = 4
    num_helpers = 4
    env = SimpleProductionEnv(num_cookers=num_cookers, num_helpers=num_helpers)

    # Parameters from the paper
    num_episodes = 5000  # Total number of episodes for training

    # Train the agent
    agent, cumulative_rewards = train_agent(env, num_episodes)

    # Save the Q-table - i.e. trained agent
    np.save("q_table.npy", agent.q_table)

    # Plot the learning progress using both regular and moving average plots
    #plot_rewards(cumulative_rewards)
    #plot_moving_average_rewards(cumulative_rewards, window_size=50)

    # Load the Q-table - trained agent
    q_table = np.load("q_table.npy")

    # Initialize the agent with the loaded Q-table
    num_states = (env.num_cookers + 1) * (env.num_helpers + 1)  # Recalculate based on your env setup
    num_actions = env.action_space.n
    agent = TabularQLearning(num_states, num_actions, alpha=0.01, gamma=0.5, epsilon=0.1)
    agent.q_table = q_table

    # Specify the number of evaluation episodes
    num_evaluation_episodes = 100  # Adjust as needed
    familiar_ratio = 0.2  # % of familiar coalitions (seen at training time)

    # Run the evaluation
    average_reward = evaluate_agent(env, agent, num_evaluation_episodes, familiar_ratio)

    print(f"Average reward over {num_evaluation_episodes} evaluation episodes: {average_reward}")

    # Run the action correctness evaluation
    num_evaluation_episodes = 100  # Define how many episodes you want to evaluate
    action_correctness = evaluate_accuracy(env, agent, num_evaluation_episodes)

   # print(f"Action accuracy: {action_correctness:.2%}")

    familiar_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = run_evaluation_across_familiar_ratios(env, agent, num_episodes=100, familiar_ratios=familiar_ratios)

    # Plot the results
    plot_familiar_ratios_vs_rewards(results)


    #TODO: Implement this as DQN
