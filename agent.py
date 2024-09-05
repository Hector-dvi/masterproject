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
        self.color = color
        self.is_trainable = True

        if device == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.input_dim = num_states
        self.policy_net = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)
        self.target_net = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)

        self.get_default_training_parameters()
        
        self.net_copy_interval = 50
        self.best_checkpoint = DQN(self.input_dim, self.get_default_hyperparameters()).to(self.device)
        self.best_validation_performance = 0.
        self.validation_check_interval = 100
    
    def reset(self):
        self.num_nodes_selected = 0
        self.current_step = 0

    def get_default_training_parameters(self):
        self.lr = 0.001 # 0.001 for first experiments, 0.00005
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_decay = 0.995 # 0.99925 0.9995
        self.epsilon_min = 0.01

    def set_optimizer(self, optim):
        self.optimizer = optim

    def set_default_training_parameters(self, lr, memory_size, batch_size, epsilon_decay, epsilon_min):
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon_min =epsilon_min

    def get_default_hyperparameters(self):
        self.hyperparams = {'hidden_gnn': 32,
                       'latent_dim': 64,
                       'hidden_dqn': 32}
        return self.hyperparams
    
    def set_default_hyperparameters(self, hyperparams):
        self.hyperparams = hyperparams
        self.policy_net = DQN(self.input_dim, hyperparams).to(self.device)
        self.target_net = DQN(self.input_dim, hyperparams).to(self.device)
        self.best_checkpoint = DQN(self.input_dim, hyperparams).to(self.device)

    def select_action(self, state, actions, num_selection):

        if random.random() < self.epsilon:
            return random.sample(actions, num_selection)
        else:
            with torch.no_grad():      
                edge_index, node_states, _ = state
                data = Data(x=node_states, edge_index=edge_index).to(self.device)
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


