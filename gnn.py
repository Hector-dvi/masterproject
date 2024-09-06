import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
import random
import numpy as np
import networkx as nx
import copy

from graph import IMAgent, MACD

class GCN1(torch.nn.Module): # good for mini graph
    def __init__(self, num_node_features, num_classes):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GCN2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, data, criterion, verbose=False):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].argmax(dim=1))
    loss.backward()
    optimizer.step()

    _, preds = model(data).max(dim=1)
    correct = preds[data.train_mask].eq(data.y[data.train_mask].argmax(dim=1))
    accuracy = int(correct.sum().item()) / int(data.train_mask.sum())

    if verbose:
        for node in data.train_mask.nonzero(as_tuple=True)[0].tolist():
            prev = data.x[node].tolist().index(1.0)
            label = data.y[node].argmax().item()
            pred = preds[node].item()
            if pred != label:
                print(node, prev, label, pred, out[node].tolist())
    return accuracy, loss.item()

def test(model, data, verbose=False):
    model.eval()
    probs = model(data)
    _, preds = probs.max(dim=1)
    correct = preds[data.test_mask].eq(data.y[data.test_mask].argmax(dim=1))
    accuracy = int(correct.sum().item()) / int(data.test_mask.sum())
    if verbose:
        for node in data.test_mask.nonzero(as_tuple=True)[0].tolist():
            prev = data.x[node].tolist().index(1.0)
            label = data.y[node].argmax().item()
            pred = preds[node].item()
            print(node, prev, label, pred, probs[node].tolist())
    return accuracy

def experiment(model, data, num_epochs, subset=None, verbose=False):

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if subset is not None:
        for node in subset:
            train_mask[node] = True
    else:
        indices = torch.randperm(num_nodes)
        train_mask[indices[:int(num_nodes * 0.8)]] = True  # 80% for training
    test_mask = ~train_mask

    data.train_mask = train_mask
    data.test_mask = test_mask

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_accuracy = 0
    x = int(num_epochs/10)
    train_verbose=False
    for epoch in range(num_epochs):
        if epoch == num_epochs-1:
            train_verbose = True
        train_accuracy, loss = train(model, optimizer, data, criterion, verbose=False)
        test_accuracy = test(model, data)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        if epoch % x == 0 and verbose:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    if verbose: 
        print(num_nodes)
        print(f"Best test accuracy: {best_test_accuracy:.4f}")
    # macd.display(with_labels=True)
    return best_test_accuracy
    
def generate_mini_graph(macd, n, states):
    num_nodes = macd.get_num_nodes()
    macd.add_node(num_nodes)
    state = random.choice(states)
    macd.set_node_attribute(num_nodes, "state", state)
    for i in range(1,n):
        state = random.choice(states)
        macd.add_node(num_nodes+i)
        macd.set_node_attribute(num_nodes+i, "state", state)
        macd.add_edge(num_nodes,num_nodes+i)
    return num_nodes
    
def mini_graphs_experiment(step_mode, agents, verbose):
    macd = MACD(agents=agents)

    states = []
    for agent in agents: 
        states.append(agent)
        states.append("S")

    subset_nodes = []
    
    for n in range(2,6):
        for _ in range(n*n*3*int(len(states)/2)):
            central_node = generate_mini_graph(macd, n, states)
            subset_nodes.append(central_node)
    macd.node_positions = nx.spring_layout(macd.graph)
    # macd.display()
    
    edge_index, x, state_to_label = macd.extract_graph()
    if step_mode == "deterministic":
        macd.diffusion_majority_step()
    else:
        macd.diffusion_stochastic_step()
    _, y, _ = macd.extract_graph()

    subset_nodes = torch.tensor(subset_nodes)
    data = Data(x=x,edge_index=edge_index, y=y)
    model = GCN2(num_node_features=len(state_to_label), num_classes=len(state_to_label))
    best_acc =  experiment(model, data, 500, subset=None, verbose=verbose)
    return best_acc

def get_agents_copy(agents):
    agent_copy = {}
    for agent in agents:
        agent_copy[agent] = copy.deepcopy(agents[agent])
    return agent_copy

def multi_ba_expirement(n, m, step_mode, agents, verbose):

    main_macd = MACD(agents=agents, diffusion_method=step_mode)
    main_macd.generate_barabasi_albert_graph(n=n, m=m)
    agents_copy = get_agents_copy(agents)
    main_macd.run(0)
    for _ in range(19):
        new_macd = MACD(agents=agents_copy, diffusion_method=step_mode)
        new_macd.generate_barabasi_albert_graph(n=n, m=m)
        num_steps = random.randint(0, 1)
        agents_copy = get_agents_copy(agents_copy)
        new_macd.run(num_steps)
        main_macd.merge_graph(new_macd.graph)
    main_macd.node_positions = nx.spring_layout(main_macd.graph)
    # main_macd.display()
    edge_index, x, state_to_label = main_macd.extract_graph()

    if step_mode == "deterministic":
        main_macd.diffusion_majority_step()
    elif step_mode == "stochastic":
        main_macd.diffusion_stochastic_step()

    # main_macd.display()
    _, y, _ = main_macd.extract_graph()

    data = Data(x=x,edge_index=edge_index, y=y)
    model = GCN2(num_node_features=len(state_to_label), num_classes=len(state_to_label))
    best_acc = experiment(model, data, 500, verbose=verbose)
    return best_acc
        
