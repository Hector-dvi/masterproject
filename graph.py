import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random

GRAPHS_DIR = "./Graphs"

class Graph:
    def __init__(self):
        self.node_positions = None
        self.graph = nx.Graph()

    def generate_barabasi_albert_graph(self, n, m, seed=None, initial_graph=None):
        self.seed = seed
        self.initial_graph = initial_graph
        self.graph = nx.barabasi_albert_graph(n, m, 
                                              seed=self.seed, 
                                              initial_graph=self.initial_graph)
        self.node_positions = nx.spring_layout(self.graph)

    def generate_watts_strogatz_graph(self, n, m, p, tries=100, seed=None):
        self.seed = seed
        self.graph = nx.connected_watts_strogatz_graph(n, m, p,
                                                     tries=tries,
                                                     seed=self.seed)
        self.node_positions = nx.circular_layout(self.graph)
    
    def load_karate_club_graph(self):
        self.graph = nx.karate_club_graph()
        self.node_positions = nx.spring_layout(self.graph)

    def load_facebook_graph(self, graph_id):
        path = f'{GRAPHS_DIR}/{graph_id}.csv'
        edges_df = pd.read_csv(path + "/edges.csv")
        # nodes_df = pd.read_csv(path + "/nodes.csv")
        self.graph = nx.from_pandas_edgelist(edges_df, '# source', ' target')
        self.node_positions = nx.spring_layout(self.graph)

    def load_gaph(self, graph):
        self.graph = graph

    def get_num_nodes(self):
        return len(self.graph.nodes)
    
    def add_node(self, node_id, **attributes):
        self.graph.add_node(node_id, **attributes)
    
    def add_edge(self, node_id1, node_id2, **attributes):
        self.graph.add_edge(node_id1, node_id2, **attributes)
    
    def remove_node(self, node_id):
        self.graph.remove_node(node_id)
    
    def remove_edge(self, node_id1, node_id2):
        self.graph.remove_edge(node_id1, node_id2)
    
    def get_node_attributes(self, node_id):
        return self.graph.nodes[node_id]
    
    def set_node_attribute(self, node_id, name, value):
        self.graph.nodes[node_id][name] = value

    def set_node_attributes(self, node_id, attr):
        for name, value in attr.item():
            self.set_node_attribute(node_id, name, value)

    def set_attribute_to_all_nodes(self, name, value):
        for node_id in self.graph.nodes:
            self.set_node_attribute(node_id, name, value)
    
    def set_attributes_to_all_nodes(self, attr):
        for name, value in attr.item():
            self.set_attribute_to_all_nodes(name, value)
    
    def get_edge_attributes(self, node_id1, node_id2):
        return self.graph.edges[(node_id1, node_id2)]
    
    def set_edge_attribute(self, node_id1, node_id2, name, value):
        self.graph.edges[(node_id1, node_id2)][name] = value

    def set_edge_attributes(self, node_id1, node_id2, attr):
        for name, value in attr.item():
            self.set_edge_attribute(node_id1, node_id2, name, value)

    def set_attribute_to_all_edges(self, name, value):
        for node_id1, node_id2 in self.graph.edges:
            self.set_edge_attribute(node_id1, node_id2, name, value)
    
    def set_attributes_to_all_edges(self, attr):
        for name, value in attr.item():
            self.set_attribute_to_all_edgess(name, value)
    
    def neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
    
    def shortest_path(self, node_id1, node_id2):
        return nx.shortest_path(self.graph, source=node_id1, target=node_id2)
    
    def has_node(self, node_id):
        return self.graph.has_node(node_id)
    
    def has_edge(self, node_id1, node_id2):
        return self.graph.has_edge(node_id1, node_id2)
    
    def all_nodes(self):
        return list(self.graph.nodes)
    
    def all_edges(self):
        return list(self.graph.edges)
    
    def get_susceptible_nodes(self):
        return [node for node in self.graph.nodes if self.graph.nodes[node]["state"] == "S"]
    
    def merge_graph(self, new_graph):
        num_nodes = self.get_num_nodes()
        new_graph = nx.relabel_nodes(new_graph, lambda x: x + num_nodes)
        self.graph = nx.union(self.graph, new_graph, rename=(None,None))
        return new_graph.nodes
    
class GraphGenerator():
    def __init__(self, train_ratio, validation_ratio, test_ratio):
        assert train_ratio + validation_ratio + test_ratio == 1.
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        self.clear()

    def clear(self):
        self.graphs = [[],[],[]]

    def get_graphs(self):
        return self.graphs

    def generate_ba(self, total_number_of_graphs, n_range, m_range):
        numbers_of_graphs = [round(total_number_of_graphs * r) for r in self.ratios]
        for list_index, number_of_graph in enumerate(numbers_of_graphs):
            for _ in range(number_of_graph):
                n = random.choice(n_range)
                m = random.choice(m_range)
                self.graphs[list_index].append(nx.barabasi_albert_graph(n,m))
        return self.graphs
    
    def generate_ws(self, total_number_of_graphs, n_range, m_range, p_range):
        numbers_of_graphs = [round(total_number_of_graphs * r) for r in self.ratios]
        for list_index, number_of_graph in enumerate(numbers_of_graphs):
            for _ in range(number_of_graph):
                n = random.choice(n_range)
                m = random.choice(m_range)
                p = random.choice(p_range)
                self.graphs[list_index].append(nx.watts_strogatz_graph(n,m,p))
        return self.graphs

class SIR(Graph):

    def __init__(self, im_agent, ad_agent=None):

        super().__init__()
        self.im_agent = im_agent
        self.ad_agent = ad_agent

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
        for node_id1, node_id2 in self.graph.edges:
            self.set_edge_attribute(node_id1, node_id2, "transmission_probability", self.im_agent.transmission_probability)
        self.susceptible = list(self.graph.nodes.keys())
        self.infected = []
        self.to_infect = {}
        self.recovered = []
        self.to_recover = {}

    def get_infected_nodes(self):
        return [node for node in self.graph.nodes if self.graph.nodes[node]["state"] == "I"]
    
    def get_recovered_nodes(self):
        return [node for node in self.graph.nodes if self.graph.nodes[node]["state"] == "R"]

    def get_influence_ratio(self):
        num_nodes = self.get_num_nodes()
        return len(self.get_infected_nodes()) / num_nodes
    
    def display(self, with_labels=False, width=0.5, edge_color="gray", edgecolors="black"):

        if self.node_positions is None:
            raise ValueError("Node positions not initialized.")
        
        node_colors = [
            self.im_agent.color if self.graph.nodes[node]["state"] == "I" 
            else self.ad_agent.color if self.graph.nodes[node]["state"] == "R" 
            else "white" 
            for node in self.graph.nodes
        ]
        nx.draw(self.graph, self.node_positions, with_labels=with_labels,
                node_color=node_colors, edge_color=edge_color, edgecolors=edgecolors)
        plt.show()

    def im_step(self, reward, num_nodes=None):
        eligible_nodes = self.get_susceptible_nodes()
        im_nodes_target = self.im_agent.step(self.graph, eligible_nodes)
        for node in im_nodes_target:
            self.to_infect[node] = self.im_agent.rate
    
    def ad_step(self):
        eligible_nodes = self.get_infected_nodes()
        ad_node = self.ad_agent.step(self.graph, eligible_nodes)
        if ad_node:
            self.to_recover[ad_node] = self.ad_agent.rate

    def diffusion_step(self, is_first_step=False, verbose=False):

        for node in list(self.to_infect.keys()):
            if self.to_infect[node] == 0:
                self.set_node_attribute(node, "state", "I")
                self.susceptible.remove(node)
                self.infected.append(node)
                del self.to_infect[node]
            else:
                self.to_infect[node] -= 1

        if verbose and is_first_step:
            self.display()

        if self.ad_agent:
            for node in list(self.to_recover.keys()):
                if self.to_recover[node] == 0:
                    self.set_node_attribute(node, "state", "R")
                    self.infected.remove(node)
                    self.recovered.append(node)
                    del self.to_recover[node]
                else:
                    self.to_recover[node] -= 1
        
            if verbose and not is_first_step: self.display()

        new_infected_nodes = set()
        for node_id in self.infected:
            for neighbor_id in self.neighbors(node_id):
                if self.graph.nodes[neighbor_id]["state"] == "S":
                    transmission_probability = self.get_edge_attributes(node_id, neighbor_id)["transmission_probability"]
                    if random.random() < transmission_probability:
                        new_infected_nodes.add(neighbor_id)
        for new_infected_node in new_infected_nodes:
            self.set_node_attribute(new_infected_node, "state", "I")
            self.susceptible.remove(new_infected_node)
            self.infected.append(new_infected_node)
        
        if verbose: self.display()

    def run(self, num_diffusion_steps, verbose=False):

        self.im_step(0)
        is_first_step = True
        for _ in range(num_diffusion_steps):
            self.diffusion_step(is_first_step=is_first_step,verbose=verbose)
            if self.ad_agent: self.ad_step()
            is_first_step=False

        return self.get_influence_ratio()

class MACD(Graph): # Multi Agent Competitive Diffusion

    def __init__(self, agents, diffusion_method="deterministic"):
        super().__init__()
        self.agents = agents
        self.diffusion_method = diffusion_method
        self.infected = {}

    def set_node_attribute(self, node_id, name, value):
        super().set_node_attribute(node_id, name, value)
        if name == "state" and value != "S": self.infected[node_id] = value
    
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
                self.infected[node] = self.graph.nodes[node]["state"]

    def get_agent_influence_ratio(self, agent_id):
        infected_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["state"] == agent_id]
        num_nodes = self.get_num_nodes()
        return len(infected_nodes) / num_nodes
    
    def get_influence_ratios(self):
        influence_ratios = {}
        for agent_id in self.agents:
            influence_ratios[agent_id] = self.get_agent_influence_ratio(agent_id)
        return influence_ratios
    
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

    def extract_graph(self):
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
        x = torch.tensor(x, dtype=torch.float)
        return edge_index, x, state_to_label

    def agent_step(self, agent_id, reward, num_nodes=None):
        eligible_nodes = self.get_susceptible_nodes()
        agent = self.agents[agent_id]
        nodes_target = agent.step(self.graph, eligible_nodes)
        for node in nodes_target:
            self.set_node_attribute(node, "state", agent_id)

    def diffusion_tp_step(self, verbose=False):

        new_infected_nodes = {}
        for node_id in self.infected:
            node_state = self.infected[node_id]
            for neighbor_id in self.neighbors(node_id):
                if self.graph.nodes[neighbor_id]["state"] == "S":
                    if random.random() < self.agents[node_state].transmission_probability:
                        if neighbor_id not in new_infected_nodes: 
                            new_infected_nodes[neighbor_id] = node_state
                        elif new_infected_nodes[neighbor_id] != node_state:
                            new_infected_nodes[neighbor_id] = "N"

        for node in new_infected_nodes:
            node_state = new_infected_nodes[node]
            self.set_node_attribute(node, "state", node_state)
            if node_state != "N": self.infected[node] = node_state
        
        if verbose: self.display()
    
    def diffusion_majority_step(self, verbose=False):

        new_infected_nodes = {}
        for node_id in self.infected:
            node_state = self.infected[node_id]
            for neighbor_id in self.neighbors(node_id):
                if self.graph.nodes[neighbor_id]["state"] == "S":
                    if neighbor_id not in new_infected_nodes: 
                        new_infected_nodes[neighbor_id] = {node_state: 1}
                    else:
                        if node_state not in new_infected_nodes[neighbor_id]:
                            new_infected_nodes[neighbor_id][node_state] = 1
                        else:
                            new_infected_nodes[neighbor_id][node_state] += 1

        for node, infecting_neigbors in new_infected_nodes.items():
            majority_value = max(infecting_neigbors.values())
            majority_key = [key for key, value in infecting_neigbors.items() if value == majority_value]
            if len(majority_key) == 1: 
                final_state = majority_key[0]
                self.set_node_attribute(node, "state", final_state)
                self.infected[node] = final_state
            # else:
                # final_state = random.choice(majority_key)
                # self.set_node_attribute(node, "state", final_state)
                # self.infected[node] = final_state

        if verbose: self.display()

    def diffusion_stochastic_step(self, verbose=False):

        new_infected_nodes = {}
        for node_id in self.infected:
            node_state = self.infected[node_id]
            for neighbor_id in self.neighbors(node_id):
                if self.graph.nodes[neighbor_id]["state"] == "S":
                    if neighbor_id not in new_infected_nodes: 
                        new_infected_nodes[neighbor_id] = {node_state: 1}
                    else:
                        if node_state not in new_infected_nodes[neighbor_id]:
                            new_infected_nodes[neighbor_id][node_state] = 1
                        else:
                            new_infected_nodes[neighbor_id][node_state] += 1

        for node, infecting_neigbors in new_infected_nodes.items():
            sampled_state = random.choices(list(infecting_neigbors.keys()), 
                                         weights=list(infecting_neigbors.values()), k=1)[0]
            self.set_node_attribute(node, "state", sampled_state)
            self.infected[node] = sampled_state

        if verbose: self.display()

    def run(self, num_diffusion_steps, verbose=False):

        for agent_id in self.agents:
            self.agent_step(agent_id, 0)

        if verbose: self.display()

        match self.diffusion_method:
            case "deterministic":
                for _ in range(num_diffusion_steps):
                    self.diffusion_majority_step(verbose=verbose)
            case "stochastic":
                for _ in range(num_diffusion_steps):
                    self.diffusion_stochastic_step(verbose=verbose)
            case "transmission_probability":
                for _ in range(num_diffusion_steps):
                    self.diffusion_tp_step(verbose=verbose)

        return self.get_influence_ratios()
    


if __name__ == "__main__":
    ratio_list = np.array([])
    # num_experiments = 0
    # # node_selection_method = "degree_centrality"
    # node_selection_methods = ["random","degree_centrality","closeness_centrality","betweenness_centrality","eigenvector_centrality","LIR"]
    # im_agent_id = 0
    # budget = 3
    # ad_agent_id = 1
    # ad_agent_type = "random"
    # graph_type = "WS"
    # num_nodes = [50]
    # edges_to_new_node = [3]
    # prob = [0.2]
    # num_diffusion_steps = 5
    # verbose = True

    # graph_types = ["BA","WS"]

    # # agent_0 = IMAgent(im_agent_id, "LIR", budget, transmission_probability=0.1, color="red")
    # # agent_1 = IMAgent(ad_agent_id, "random", budget, transmission_probability=0.1, color="blue")
    # # agent_2 = IMAgent(2, "eigenvector_centrality", budget, color="green")
    # # agent_ad = RandomAgent(1)
    # # agents = {im_agent_id: agent_0, ad_agent_id: agent_1}
    # # model = SIR(agent_0, agent_ad)
    # # model.generate_watts_strogatz_graph(50, 4, 0.2)

    # agent_0 = IMAgent(0, "random", budget, transmission_probability=0.1, color="red")
    # agent_1 = IMAgent(1, "random", budget, transmission_probability=0.1, color="blue")
    # agent_2 = IMAgent(2, "random", budget, transmission_probability=0.1, color="green")
    # agent_3 = IMAgent(3, "random", budget, transmission_probability=0.1, color="yellow")
    # agents = {0: agent_0,
    #           1: agent_1, 
    #           2: agent_2, 
    #           3: agent_3}
    # model = MACD(agents, diffusion_method="stochastic")
    # # model.load_facebook_graph("gplus_101133961721621664586")
    # model.generate_barabasi_albert_graph(100, 3)
    # ratios = model.run(num_diffusion_steps,verbose=verbose)
    # # print(ratios)
    
    # for graph_type in graph_types:
    #     print(f'{graph_type}:')
    #     for selection_method in node_selection_methods:
    #         ratio_list = np.array([])
    #         for n in num_nodes:
    #             for m in edges_to_new_node:
    #                 if graph_type=="BA":
    #                     for exp in range(num_experiments):
    #                         agent_0 = IMAgent(im_agent_id, selection_method, budget, transmission_probability=0.1, color="red")
    #                         model = SIR(agent_0)
    #                         model.generate_barabasi_albert_graph(n, m, seed=exp)
    #                         ratio = model.run(num_diffusion_steps=num_diffusion_steps,verbose=False)
    #                         ratio_list = np.append(ratio_list,ratio)
    #                 else:
    #                     for p in prob:
    #                         ratio_list = np.array([])
    #                         for exp in range(num_experiments):
    #                             agent_0 = IMAgent(im_agent_id, selection_method, budget, transmission_probability=0.1, color="red")
    #                             model = SIR(agent_0)
    #                             model.generate_watts_strogatz_graph(n, m, p, seed=exp)
    #                             ratio = model.run(num_diffusion_steps=num_diffusion_steps,verbose=False)
    #                             ratio_list = np.append(ratio_list,ratio)
    #         mean = ratio_list.mean()
    #         std = ratio_list.std()
    #         standard_error = std / np.sqrt(num_experiments)
    #         margin_error = standard_error * 1.96 # 95% interval
    #         print(selection_method, round(mean,3), round(margin_error,3))
        
    
