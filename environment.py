import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random

from graph import Graph

class RLEnv(Graph): # Multi Agent Competitive Diffusion

    def __init__(self, agents, diffusion_method="deterministic", num_steps=10):
        super().__init__()
        self.agents = agents
        self.diffusion_method = diffusion_method
        self.num_steps = num_steps
        self.current_step = 0
        self.influenced = {}

    def reset(self, new_graph=None):
        for _, agent in self.agents.items():
            agent.reset()
        self.current_step = 0
        self.influenced = {}
        if new_graph:
            self.load_gaph(new_graph)
        self.init_graph()
        return self.get_state()
    
    def setup_graphs(self, train_graphs, validation_graphs, test_graphs):
        self.train_graphs = train_graphs
        self.validation_graphs = validation_graphs
        self.test_graphs = test_graphs
    
    def get_rlagent_id(self):
        for id, agent in self.agents.items():
            if agent.is_trainable: return id
        return None
    
    def get_state(self):
        return self.extract_graph_data()
    
    def get_agent_influence_ratio(self, agent_id):
        infected_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["state"] == agent_id]
        num_nodes = self.get_num_nodes()
        return len(infected_nodes) / num_nodes
    
    def get_influence_ratios(self):
        influence_ratios = {}
        for agent_id in self.agents:
            influence_ratios[agent_id] = self.get_agent_influence_ratio(agent_id)
        return influence_ratios
    
    def set_node_attribute(self, node_id, name, value):
        super().set_node_attribute(node_id, name, value)
        if name == "state" and value != "S": self.influenced[node_id] = value
    
    def generate_barabasi_albert_graph(self, n, m, seed=None, initial_graph=None):
        super().generate_barabasi_albert_graph(n, m, seed, initial_graph)
        self.init_graph()

    def generate_watts_strogatz_graph(self, n, m, p, tries=100, seed=None):
        super().generate_watts_strogatz_graph(n, m, p, tries, seed)
        self.init_graph()
    
    def load_karate_club_graph(self):
        super().load_karate_club_graph()
        self.init_graph()
    
    def load_facebook_graph(self, graph_id):
        super().load_facebook_graph(graph_id)
        self.init_graph()

    def load_gaph(self, graph):
        super().load_gaph(graph)
        self.init_graph()

    def init_graph(self):
        self.set_attribute_to_all_nodes("state", "S")

    def merge_graph(self, new_graph):
        new_nodes = super().merge_graph(new_graph)
        for node in new_nodes:
            if self.graph.nodes[node]["state"] != "S": 
                self.influenced[node] = self.graph.nodes[node]["state"]

    def extract_graph_data(self):
        edge_index = torch.tensor(list(self.graph.edges)).t().contiguous()
        state_to_label = {"S": 0}
        current_label = 1
        for agent_id in self.agents:
            state_to_label[agent_id] = current_label
            current_label += 1
        x = []
        for node in self.graph.nodes:
            state = self.graph.nodes[node]["state"]
            one_hot = [0] * len(state_to_label)
            one_hot[state_to_label[state]] = 1
            x.append(one_hot)
        x = torch.tensor(x,dtype=torch.float32)
        return edge_index, x, state_to_label
    
    def display(self, with_labels=False, width=0.5, edge_color="gray", edgecolors="black"):

        if self.node_positions is None:
            raise ValueError("Node positions not initialized.")
        
        node_colors = [
            "white" if self.graph.nodes[node]["state"] == "S" 
            else "grey" if self.graph.nodes[node]["state"] == "N"
            else self.agents[self.graph.nodes[node]["state"]].color
            for node in self.graph.nodes
        ]
        k = 250
        num_nodes = self.get_num_nodes()
        node_size = k / (num_nodes ** 0.35)
        nx.draw(self.graph, self.node_positions, with_labels=with_labels, 
                node_size=node_size, node_color=node_colors, 
                width=width, edge_color=edge_color, edgecolors=edgecolors)
        plt.show()

    def apply_action(self, agent_id, nodes_target):
        for node in nodes_target:
            self.set_node_attribute(node, "state", agent_id)

    def check_termination(self):
        for node, attributes in self.graph.nodes.items():
            if attributes["state"] == "S":
                return False
        return True

    def get_new_influenced_nodes(self):
        new_influenced_nodes = {}
        for node_id in self.influenced:
            node_state = self.influenced[node_id]
            for neighbor_id in self.neighbors(node_id):
                if self.graph.nodes[neighbor_id]["state"] == "S":
                    if neighbor_id not in new_influenced_nodes: 
                        new_influenced_nodes[neighbor_id] = {node_state: 1}
                    else:
                        if node_state not in new_influenced_nodes[neighbor_id]:
                            new_influenced_nodes[neighbor_id][node_state] = 1
                        else:
                            new_influenced_nodes[neighbor_id][node_state] += 1
        return new_influenced_nodes
    
    def update_nodes_state(self, new_influenced_nodes):
        for node, influencing_neigbors in new_influenced_nodes.items():
            match self.diffusion_method:
                case "deterministic":
                    majority_value = max(influencing_neigbors.values())
                    majority_key = [key for key, value in influencing_neigbors.items() if value == majority_value]
                    if len(majority_key) == 1: 
                        final_state = majority_key[0]
                        self.set_node_attribute(node, "state", final_state)
                        self.influenced[node] = final_state
                case "stochastic":
                    sampled_state = random.choices(list(influencing_neigbors.keys()), 
                                         weights=list(influencing_neigbors.values()), k=1)[0]
                    self.set_node_attribute(node, "state", sampled_state)
                    self.influenced[node] = sampled_state

    def agent_step(self, num_selection):
        action = []

        for id, agent in self.agents.items():
            if self.check_termination():
                return action
            nodes_target = agent.step(self, num_selection=num_selection) 
            if agent.is_trainable: 
                action = nodes_target
            self.apply_action(id, nodes_target)
        return action

    def step(self, verbose=False):
        self.current_step += 1
        action = self.agent_step(1)
        if verbose: self.display()
        new_influenced_nodes = self.get_new_influenced_nodes()
        self.update_nodes_state(new_influenced_nodes)
        done = (self.current_step == self.num_steps) or self.check_termination()
        reward = self.get_agent_influence_ratio(self.get_rlagent_id()) if done else 0
        
        return self.get_state(), action, reward, done
    
    def check_validation(self, episode_length):
        if self.validation_graphs == []:
            return None
        rewards = []
        for g in self.validation_graphs:
            self.reset(new_graph=g)
            for _ in range(episode_length):
                _, _, reward, done = self.step()
                if done:
                    rewards.append(reward)
                    break
        average_reward = np.mean(np.array(rewards))
        rl_agent = self.agents[self.get_rlagent_id()]
        if average_reward > rl_agent.best_validation_performance:
            rl_agent.save_checkpoint(average_reward)
        return average_reward

    