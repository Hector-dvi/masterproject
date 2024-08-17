import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from heapq import nlargest
from collections import deque
import random
import os

MODELS_DIR = "./Models"

from network import DQN
from memory import NstepReplayMem

class IMAgent:
    def __init__(self, agent_id, agent_type, budget, transmission_probability=0.1, infection_rate=0, color="red"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.budget = budget
        self.num_nodes_selected = 0
        self.transmission_probability = transmission_probability
        self.rate = infection_rate
        self.color = color
        self.is_trainable = False

    def reset(self):
        self.num_nodes_selected = 0

    def select_nodes(self, graph, eligible_nodes, num_nodes):
        match self.agent_type:
            case "random":
                nodes = random.sample(eligible_nodes, num_nodes)
            case "degree_centrality":
                centrality = nx.degree_centrality(graph)
                eligible_centrality = {node: centrality[node] for node in eligible_nodes}
                nodes = nlargest(num_nodes, eligible_centrality, key=eligible_centrality.get)
            case "closeness_centrality":
                centrality = nx.closeness_centrality(graph)
                eligible_centrality = {node: centrality[node] for node in eligible_nodes}
                nodes = nlargest(num_nodes, eligible_centrality, key=eligible_centrality.get)
            case "betweenness_centrality":
                centrality = nx.betweenness_centrality(graph)
                eligible_centrality = {node: centrality[node] for node in eligible_nodes}
                nodes = nlargest(num_nodes, eligible_centrality, key=eligible_centrality.get)
            case "eigenvector_centrality":
                centrality = nx.eigenvector_centrality(graph, max_iter=10000, tol=1e-04)
                eligible_centrality = {node: centrality[node] for node in eligible_nodes}
                nodes = nlargest(num_nodes, eligible_centrality, key=eligible_centrality.get)
            case "LIR":
                LI = {}
                for vi in list(graph.nodes.keys()):
                    LI[vi] = 0
                    neighbors = list(graph.neighbors(vi))
                    for vj in neighbors:
                        degree_vi = graph.degree(vi)
                        degree_vj = graph.degree(vj)
                        if degree_vj > degree_vi:
                            LI[vi] += 1
                LI_zero_nodes = [vi for vi in eligible_nodes if LI[vi] == 0]
                LI_zero_nodes.sort(key=lambda vi: -graph.degree(vi))
                if len(LI_zero_nodes) >= num_nodes:
                    nodes = LI_zero_nodes[:num_nodes]
                else:
                    nodes = LI_zero_nodes
                    LI_one_nodes = [vi for vi in eligible_nodes if LI[vi] == 1]
                    LI_one_nodes.sort(key=lambda vi: -graph.degree(vi))
                    remaining = num_nodes - len(nodes)
                    nodes.extend(LI_one_nodes[:remaining])
            case _:
                raise ValueError(f"'{self.agent_type}' agent type does not exist.")
            
        return nodes
    
    def step(self, env, num_selection=None):
        graph = env.graph
        actions = env.get_susceptible_nodes()
        if not num_selection or self.num_nodes_selected + num_selection > self.budget:
            num_selection = self.budget - self.num_nodes_selected
        num_selection = min(len(actions), num_selection)
        selected_nodes = self.select_nodes(graph, actions, num_selection)
        self.num_nodes_selected += num_selection
        return selected_nodes
    
class RandomAgent:
    def __init__(self, agent_id, transmission_probability=0, recovery_rate=0, color="blue"):
        self.name = agent_id
        self.transmission_probability = transmission_probability
        self.rate = recovery_rate
        self.is_trainable = False
        self.color = color

    def step(self, eligible_nodes):
        if eligible_nodes != []: 
            return np.random.choice(eligible_nodes)

class RLAgent:
    def __init__(self, agent_id, budget, num_states, device="CPU", color="blue"):
        self.agent_id = agent_id
        self.budget = budget
        self.num_nodes_selected = 0
        self.current_step = 0

        if device == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.input_dim = num_states
        self.policy_net = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)
        self.target_net = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)

        self.lr = 0.001
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_decay = 0.999925
        self.epsilon_min = 0.01
        self.color = color
        self.is_trainable = True
        self.net_copy_interval = 50

        self.best_checkpoint = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)
        self.best_validation_performance = 0.
        self.validation_check_interval = 100

        
    
    def reset(self):
        self.num_nodes_selected = 0
        self.current_step = 0

    def setup_graphs(train_graphs, validation_graphs):
        pass

    def get_default_hyperparameters(self):
        hyperparams = {'hidden_gnn': 32,
                       'latent_dim': 64,
                       'hidden_dqn': 32}
        return hyperparams

    def select_action(self, state, actions, num_selection):

        if random.random() < self.epsilon:
            return random.sample(actions, num_selection)
        else:
            with torch.no_grad():      
                edge_index, node_states, _ = state
                data = Data(x=node_states, edge_index=edge_index)
                q_values = self.policy_net(data).squeeze()
                possible_actions = torch.tensor(actions, dtype=torch.long).to(self.device)
                possible_q_values = q_values[torch.tensor(actions)]
                _, topk_indices = torch.topk(possible_q_values, num_selection)
                selected_actions = possible_actions[topk_indices].tolist()
                return selected_actions
            
    def step(self, env, num_selection=None):
        self.current_step += 1
        state = env.get_state()
        actions = env.get_susceptible_nodes()
        if not num_selection or self.num_nodes_selected + num_selection > self.budget:
            num_selection = self.budget - self.num_nodes_selected
        num_selection = min(len(actions), num_selection)
        selected_nodes = self.select_action(state, actions, num_selection)
        self.num_nodes_selected += num_selection
        if self.current_step % self.net_copy_interval == 0:
                self.take_snapshot()
        return selected_nodes
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        data_policy, data_target = [], []
        _, actions, rewards, _, dones = zip(*batch)
        for state, _, _, next_state, _ in batch:
            edge_index, node_states, _ = state
            _, node_next_states, _ = next_state
            data_policy.append(Data(x=node_states, edge_index=edge_index))
            data_target.append(Data(x=node_next_states, edge_index=edge_index))

        data_policy = Batch.from_data_list(data_policy).to(self.device)
        data_target = Batch.from_data_list(data_target).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # q_values = self.policy_net(data_policy)
        # q_values = q_values.view(self.batch_size, -1).gather(1, actions).squeeze()
        # next_q_values = self.target_net(data_target).view(self.batch_size, -1).max(1)[0]
        
        q_values_raw = self.policy_net(data_policy)
        q_values_dense, _ = to_dense_batch(q_values_raw, data_policy.batch)
        q_values = q_values_dense.gather(1, actions.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        next_q_values_raw = self.target_net(data_target)
        next_q_values_dense, _ = to_dense_batch(next_q_values_raw, data_target.batch)
        next_q_values = next_q_values_dense.squeeze(-1).max(1)[0]

        exp_q_values = rewards + self.gamma * next_q_values
        loss = F.mse_loss(q_values, exp_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def take_snapshot(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_best_model(self, model_name):
        torch.save(self.best_checkpoint, f"{MODELS_DIR}/{model_name}.pt")

    def load_model(self, model_name):
        self.policy_net.load_state_dict(torch.load(f"{MODELS_DIR}/{model_name}.pt", map_location=self.device))
        self.take_snapshot()
    
    def save_checkpoint(self, value):
        self.best_checkpoint.load_state_dict(self.policy_net.state_dict())
        self.best_validation_performance = value

    def restore_last_checkpoint(self):
        self.policy_net.load_state_dict(self.best_checkpoint.state_dict())
        self.take_snapshot()


# class RLAgent2:

#     def __init__(self, agent_id, budget, color="blue"):
#         self.enable_assertions = True
#         self.hist_out = None
#         self.id = agent_id
#         self.budget = budget
#         self.color = "blue"

#         self.batch_size = 50
#         self.is_deterministic = False
#         self.is_trainable = True

#         self.validation_change_threshold = 1e-5
#         self.best_validation_changed_step = -1
#         self.best_validation_loss = float("inf")

#         self.pos = 0
#         self.step = 0
#         self.hyperparams = self.get_default_hyperparameters()
#         self.net = QNet(self.hyperparams)
#         self.old_net = QNet(self.hyperparams)

#     def get_default_hyperparameters(self):
#         hyperparams = {'learning_rate': 0.0001,
#                        'epsilon_start': 1,
#                        'mem_pool_to_steps_ratio': 1,
#                        'latent_dim': 64,
#                        'hidden': 32,
#                        'embedding_method': 'mean_field',
#                        'max_lv': 3,
#                        'eps_step_denominator': 10}
#         return hyperparams
    
#     def take_snapshot(self):
#         self.old_net.load_state_dict(self.net.state_dict())

#     def train(self, max_steps):

#         if len(self.memory) < self.batch_size:
#             return

#         # Setup
#         self.setup_mem_pool(max_steps, self.hyperparams['mem_pool_to_steps_ratio'])
#         self.setup_training_parameters(max_steps)

#         # Burn-in
#         pbar = tqdm(range(self.burn_in), unit='batch', disable=None)
#         for p in pbar:
#             with torch.no_grad():
#                 self.run_simulation()

#         pbar = tqdm(range(max_steps + 1), unit='steps', disable=None)
#         optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

#         for self.step in pbar:
#             with torch.no_grad():
#                 self.run_simulation()
#             if self.step % self.net_copy_interval == 0:
#                 self.take_snapshot()
#             # self.check_validation_loss(self.step, max_steps)

#             batch = random.sample(self.memory, self.batch_size)
#             states, actions, rewards, next_states, dones = zip(*batch)

#             cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
#                 batch_size=self.batch_size)
#             list_target = torch.Tensor(list_rt)
#             cleaned_sp = []
#             nonterms = []
#             for i in range(len(list_st)):
#                 if not list_term[i]:
#                     cleaned_sp.append(list_s_primes[i])
#                     nonterms.append(i)

#             if len(cleaned_sp):
#                 _, _, banned = zip(*cleaned_sp)
#                 _, q_t_plus_1, prefix_sum_prime = self.old_net((cur_time + 1) % 2, cleaned_sp, None)
#                 _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
#                 list_target[nonterms] = q_rhs

#             list_target = Variable(list_target.view(-1, 1))
#             _, q_sa, _ = self.net(cur_time % 2, list_st, list_at)

#             loss = F.mse_loss(q_sa, list_target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss))

#             # should_stop = self.check_stopping_condition(self.step, max_steps)
#             # if should_stop:
#             #     break

#     def setup_mem_pool(self, num_steps, mem_pool_to_steps_ratio):
#         exp_replay_size = int(num_steps * mem_pool_to_steps_ratio)
#         self.memory = deque(maxlen=exp_replay_size)

#     def setup_training_parameters(self, max_steps):
#         self.learning_rate = self.hyperparams['learning_rate']
#         self.eps_start = self.hyperparams['epsilon_start']

#         eps_step_denominator = self.hyperparams['eps_step_denominator'] if 'eps_step_denominator' in self.hyperparams else 2
#         self.eps_step = max_steps / eps_step_denominator
#         self.eps_end = 0.1
#         self.burn_in = 50
#         self.net_copy_interval = 50

#     def make_actions(self, t):

#         self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
#                                         * (self.eps_step - max(0., self.step)) / self.eps_step)
#         if random.random() < self.eps: # Explore

#             # do explore action
#             exploration_actions_t0, exploration_actions_t1 = self.environment.exploratory_actions(self.agent_exploration_policy)
#             self.next_exploration_actions = exploration_actions_t1
#             return exploration_actions_t0
#         else: # Greedy

#             # do greedy action
#             greedy_acts = self.do_greedy_actions(t)
#             self.next_exploration_actions = None
#             return greedy_acts

#     def do_greedy_actions(self, time_t):
#         cur_state = self.environment.get_state_ref()
#         actions, _, _ = self.net(time_t % 2, cur_state, None, greedy_acts=True)
#         actions = list(actions.cpu().numpy())
#         return actions

#     def run_simulation(self):
#         selected_idx = self.advance_pos_and_sample_indices()
#         self.environment.setup([self.train_g_list[idx] for idx in selected_idx],
#                            [self.train_initial_obj_values[idx] for idx in selected_idx],
#                            training=True)
#         self.post_env_setup()

#         final_st = [None] * len(selected_idx)
#         final_acts = np.empty(len(selected_idx), dtype=np.int); final_acts.fill(-1)

#         t = 0
#         while not self.environment.is_terminal():
#             list_at = self.make_actions(t, greedy=False)

#             non_exhausted_before, = np.where(~self.environment.exhausted_budgets)
#             list_st = self.environment.clone_state(non_exhausted_before)
#             self.environment.step(list_at)

#             non_exhausted_after, = np.where(~self.environment.exhausted_budgets)
#             exhausted_after, = np.where(self.environment.exhausted_budgets)

#             nonterm_indices = np.flatnonzero(np.isin(non_exhausted_before, non_exhausted_after))
#             nonterm_st = [list_st[i] for i in nonterm_indices]
#             nonterm_at = [list_at[i] for i in non_exhausted_after]
#             rewards = np.zeros(len(nonterm_at), dtype=np.float)
#             nonterm_s_prime = self.environment.clone_state(non_exhausted_after)

#             now_term_indices = np.flatnonzero(np.isin(non_exhausted_before, exhausted_after))
#             term_st = [list_st[i] for i in now_term_indices]
#             for i in range(len(term_st)):
#                 g_list_index = non_exhausted_before[now_term_indices[i]]

#                 final_st[g_list_index] = term_st[i]
#                 final_acts[g_list_index] = list_at[g_list_index]

#             if len(nonterm_at) > 0:
#                 self.mem_pool.add_list(nonterm_st, nonterm_at, rewards, nonterm_s_prime, [False] * len(nonterm_at), t % 2)

#             t += 1

#         final_at = list(final_acts)
#         rewards = self.environment.rewards
#         final_s_prime = None
#         self.mem_pool.add_list(final_st, final_at, rewards, final_s_prime, [True] * len(final_at), (t - 1) % 2)

#     def get_default_hyperparameters(self):
#         hyperparams = {'learning_rate': 0.0001,
#                        'epsilon_start': 1,
#                        'mem_pool_to_steps_ratio': 1,
#                        'latent_dim': 64,
#                        'hidden': 32,
#                        'embedding_method': 'mean_field',
#                        'max_lv': 3,
#                        'eps_step_denominator': 10}
#         return hyperparams







