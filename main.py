import tqdm
import numpy as np
import matplotlib.pyplot as plt

from environment import RLEnv
from agent import IMAgent, RLAgent
from graph import GraphGenerator


def run_baseline(env, agent_id, episode_length, num_episodes):
    rewards = np.array([])
    for episode in range(num_episodes):
        env.reset()
        for _ in range(episode_length):
            _, _, _, done = env.step()
            if done:
                reward = env.get_agent_influence_ratio(agent_id)
                rewards = np.append(rewards, reward)
    return rewards

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train(env, episode_length, num_episodes, save_model=False):
    average_rewards_train = []
    average_rewards_validation = []
    rl_agent = env.agents[env.get_rlagent_id()]
    for episode in range(num_episodes):
        episode_rewards = []
        for g in env.train_graphs:
            state = env.reset(new_graph=g)
            for _ in range(episode_length):
                next_state, action, reward, done = env.step()
                if action != []:
                    rl_agent.remember(state, action, reward, next_state, done)
                loss = rl_agent.optimize_model()
                state = next_state
                if done:
                    episode_rewards.append(reward)
                    break
        average_rewards_train.append(np.mean(np.array(episode_rewards)))
        average_reward_validation = env.check_validation(episode_length)
        if average_reward_validation:
            average_rewards_validation.append(average_reward_validation)

    if save_model:
        rl_agent.save_best_model("test")
    return average_rewards_train, average_rewards_validation


def plot_results(rewards, baseline_average, baseline):
    plt.plot(rewards)
    plt.axhline(y=baseline_average, color='r', linestyle='--', label=f'Baseline Average ({baseline})= {baseline_average:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.show()

def plot_results_multiple_graphs(average_rewards_train, average_rewards_validation, baseline_average, baseline):
    plt.plot(average_rewards_train, linewidth=1, color="red",label=f"Average Reward Testing")
    if len(average_rewards_validation) > 0:
        plt.plot(average_rewards_validation, linewidth=1, color="green",label=f"Average Reward Validation")
    plt.axhline(y=baseline_average, color='black', linestyle='--', label=f'Baseline Average ({baseline})= {baseline_average:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    episode_length = 10
    budget = 1
    n_episodes = 1500

    rl_agent = RLAgent(2, budget, 3, color="red")
    im_agent = IMAgent(1, "degree_centrality", budget, color="blue")
    agents = {1: im_agent, 2:rl_agent}
    environment = RLEnv(agents, diffusion_method="stochastic")
    graph_gen = GraphGenerator(0.7, 0.3, 0.)
    train_graphs, validation_graphs, test_graphs = graph_gen.generate_ba(50, [100], [3])
    environment.setup_graphs(train_graphs, validation_graphs, test_graphs)

    average_rewards_train, average_rewards_validation = train(environment, episode_length, n_episodes, save_model=True)

    print(rl_agent.best_validation_performance)
    # plot_results_multiple_graphs(average_rewards_train, average_rewards_validation, 0.138, "random")

    train_moving_average = moving_average(average_rewards_train, 50)
    validation_moving_average = moving_average(average_rewards_validation, 50)
    plot_results_multiple_graphs(train_moving_average, validation_moving_average, 0.138, "random")
