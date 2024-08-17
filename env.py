import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random

from graph import Network

class RLEnv(Network): # Multi Agent Competitive Diffusion

    def __init__(self, agents, diffusion_method="deterministic", num_steps=10):
        super().__init__()
        self.agents = agents
        self.diffusion_method = diffusion_method
        self.num_steps = num_steps
        self.influenced = {}

    def reset(self):
        self.current_step = 0
        self.graph.init_graph()
        return self._get_state()
    
    def step(self, action):
        self.graph.im_step(action)
        self.graph.diffusion_step()
        self.current_step += 1
        
        done = (self.current_step == self.num_steps)
        reward = self.graph.get_influence_ratio() if done else 0
        
        return self._get_state(), reward, done, {}
    
    def get_agent_influence_ratio(self, agent_id):
        pass
    
    def get_influence_ratios(self):
        pass
    
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

    def init_graph(self):
        self.set_attribute_to_all_nodes("state", "S")

    def merge_graph(self, new_graph):
        new_nodes = super().merge_graph(new_graph)
        for node in new_nodes:
            if self.graph.nodes[node]["state"] != "S": 
                self.influenced[node] = self.graph.nodes[node]["state"]
    
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

    