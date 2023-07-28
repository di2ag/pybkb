import os
import re
import json
import copy
import networkx as nx
import numpy as np
from collections import defaultdict
from ipycytoscape import CytoscapeWidget

DEFAULT_STYLE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../',
        'utils/assets/bkb-style.json',
        )

class BKBNetworkXMixin:
    def construct_nx_data(
            self,
            include_sources=False,
            include_source_weight=True,
            node_padding:int=7,
            no_snodes:bool=False,
            collapse_edges:bool=True,
            ):
        # Initialize
        node_info_template = {"label_width": None, "type": None, "tooltip":None}
        nodes = defaultdict(lambda: copy.deepcopy(node_info_template))
        edges = []
        # Construct S-nodes by head
        snodes_by_head = self.snodes_by_head
        # Construct an S-node
        for snode_idx, snode_prob in enumerate(self.snode_probs):
            head_feature, head_state = self.get_snode_head(snode_idx)
            # Cast name to string to handle names that are ints
            _head_feature = str(head_feature)
            if '__Source__' in _head_feature and not include_sources:
                continue
            head_name = f'{head_feature} = {head_state}'
            # Create head I-node
            nodes[head_name]["label_width"] = len(head_name) * node_padding
            nodes[head_name]["type"] = 'i'
            # Create S-node
            if not no_snodes:
                snode_name = str(snode_idx)#r'$s_{%s} = %.3f$' % (str(snode_idx), snode_prob)
                nodes[snode_name]["type"] = 's'
                nodes[snode_name]["tooltip"] = str(snode_prob)
                # Add edges from S-node to head inode
                edges.append((snode_name, head_name))
            tail_filter = []
            for tail_feature, tail_state in self.get_snode_tail(snode_idx):
                tail_name = f'{tail_feature} = {tail_state}'
                # Cast name to string to handle names that are ints
                _tail_feature = str(tail_feature)
                if '__Source__' in _tail_feature:
                    if not include_sources:
                        continue
                if tail_name not in nodes:
                    nodes[tail_name]["label_width"] = len(tail_name) * node_padding
                    nodes[tail_name]["type"] = 'i'
                tail_filter.append(tail_name)
            # Add edges
            for tail_name in tail_filter:
                if no_snodes:
                    edges.append((tail_name, head_name))
                else:
                    edges.append((tail_name, snode_name))
        return nodes, self.process_edges(edges)

    def construct_nx_bkb(
            self,
            include_sources=False,
            node_padding=7,
            no_snodes:bool=False,
            collapse_edges:bool=True,
            show_num_edges:bool=False,
            edge_weight_multiplier:float=1.0,
            count_min=False,
            ):
        nodes, edges = self.construct_nx_data(
                include_sources=include_sources,
                node_padding=node_padding,
                no_snodes=no_snodes,
                collapse_edges=collapse_edges,
                )
        G = nx.MultiDiGraph()
        # Add nodes
        for node_id, node_data in nodes.items():
            G.add_node(
                    node_id,
                    label_width = node_data["label_width"],
                    type=node_data["type"],
                    tooltip=node_data["tooltip"],
                    )
        return self.add_edges_to_graph(edges, G, collapse_edges, show_num_edges, edge_weight_multiplier, count_min)

    @staticmethod
    def process_edges(edges):
        # Initialize
        edge_info_template = {"count": 0}
        edges_dict = defaultdict(lambda: copy.deepcopy(edge_info_template))
        # Collect and Preprocess
        for e in edges:
            edge = edges_dict[e]
            edge["count"] += 1
        return edges_dict

    @staticmethod
    def add_edges_to_graph(edges, G, collapse_edges, show_num_edges, edge_weight_multiplier, count_min):
        for e, edata in edges.items():
            u, v = e
            if count_min:
                if edata["count"] < count_min:
                    continue
            if not collapse_edges:
                for _ in range(edata["count"]):
                    if show_num_edges:
                        G.add_edge(u, v, label='1', count=1, weight=edge_weight_multiplier)
                    else:
                        G.add_edge(u, v, count=1, weight=edge_weight_multiplier)
            elif show_num_edges:
                G.add_edge(u, v, label=str(edata["count"]), count=edata["count"], weight=edata["count"]*edge_weight_multiplier)
            else:
                G.add_edge(u, v, count=edata["count"], weight=edata["count"]*edge_weight_multiplier)
        return G

    def construct_nx_rv_data(
            self,
            node_padding:int=7,
            collapse_edges:bool=True,
            show_num_edges:bool=False,
            edge_weight_multiplier:float=1.0,
            count_min=False,
            ):
        # Initialize
        node_info_template = {"label_width": None, "type": None, "tooltip":None}
        nodes = defaultdict(lambda: copy.deepcopy(node_info_template))
        edges = []
        # Construct S-nodes by head
        snodes_by_head = self.snodes_by_head
        # Construct an S-node
        for snode_idx, snode_prob in enumerate(self.snode_probs):
            head_name, _ = self.get_snode_head(snode_idx)
            # Cast name to string to handle names that are ints
            _head_name = str(head_name)
            if '__Source__' in _head_name:
                continue
            # Create head I-node
            nodes[head_name]["label_width"] = len(head_name) * node_padding
            nodes[head_name]["type"] = 'i'
            tail_filter = []
            for tail_name, _ in self.get_snode_tail(snode_idx):
                # Cast name to string to handle names that are ints
                _tail_name = str(tail_name)
                if '__Source__' in _tail_name:
                    continue
                if tail_name not in nodes:
                    nodes[tail_name]["label_width"] = len(tail_name) * node_padding
                    nodes[tail_name]["type"] = 'i' 
                # Add edge
                edges.append((tail_name, head_name))
        return nodes, self.process_edges(edges)

    def construct_nx_bkb_rv(
            self, 
            node_padding:int=7,
            collapse_edges:bool=True,
            show_num_edges:bool=False,
            edge_weight_multiplier:float=1.0,
            count_min=False,
            ):
        nodes, edges = self.construct_nx_rv_data(
                node_padding,
                collapse_edges,
                show_num_edges,
                count_min,
                )
        G = nx.MultiDiGraph()
        for node_id, node_data in nodes.items():
            G.add_node(
                    node_id,
                    label_width=node_data["label_width"],
                    type=node_data["type"],
                    tooltip=node_data["tooltip"],
                    )
        return self.add_edges_to_graph(edges, G, collapse_edges, show_num_edges, edge_weight_multiplier, count_min)

    def construct_nx_graph(
            self,
            node_padding:int=7,
            include_sources:bool=False,
            subgraph_on:list=None,
            subgraph_on_partial:list=None,
            only_rvs:bool=False,
            remove_simple_cycles:bool=False,
            no_snodes:bool=True,
            collapse_edges:bool=True,
            show_num_edges:bool=False,
            edge_weight_multiplier:float=1.0,
            count_min=False,
            ):
        # Load networkx
        if only_rvs:
            G = self.construct_nx_bkb_rv(
                    node_padding,
                    collapse_edges,
                    show_num_edges,
                    edge_weight_multiplier,
                    count_min,
                    )
        else:
            G = self.construct_nx_bkb(
                    include_sources,
                    node_padding,
                    no_snodes,
                    collapse_edges,
                    show_num_edges,
                    edge_weight_multiplier,
                    count_min,
                    )
        if subgraph_on is not None:
            G = nx.subgraph(G, subgraph_on)
        if subgraph_on_partial is not None:
            # Gather nodes that match partial string
            nodes = set()
            for partial in subgraph_on_partial:
                matches = [n for n in G.nodes if partial in n]
                nodes.update(set(matches))
            # Subgraph
            G = nx.subgraph(G, list(nodes))
        return G

    def construct_nx_adj(
            self,
            include_sources:bool=False,
            subgraph_on:list=None,
            subgraph_on_partial:list=None,
            only_rvs:bool=False,
            remove_simple_cycles:bool=False,
            no_snodes:bool=False,
            collapse_edges:bool=True,
            ):
        # Construct graph
        G = self.construct_nx_graph(
                include_sources=include_sources,
                subgraph_on=subgraph_on,
                subgraph_on_partial=subgraph_on_partial,
                only_rvs=only_rvs,
                remove_simple_cycles=remove_simple_cycles,
                no_snodes=no_snodes,
                collapse_edges=collapse_edges,
                )
        nodes = sorted(list(G.nodes()))
        nodes_map = {n: idx for idx, n in enumerate(nodes)}
        if collapse_edges:
            return nx.to_numpy_matrix(G, nodelist=nodes), nodes
        # Process edge numbers as weight adj matrix
        adj = np.zeros((len(nodes), len(nodes)))
        for u, v, data in G.edges.data():
            u_idx = nodes_map[u]
            v_idx = nodes_map[v]
            curr_count = adj[u_idx, v_idx]
            adj[u_idx, v_idx] = data["count"] + curr_count
        return adj, nodes

    def render_nx(
            self,
            style_file:str=None,
            layout_name:str='dagre',
            node_spacing:int=10,
            edge_length_val:int=10,
            node_padding:int=7,
            include_sources:bool=False,
            subgraph_on:list=None,
            subgraph_on_partial:list=None,
            only_rvs:bool=False,
            remove_simple_cycles:bool=False,
            no_snodes:bool=False,
            collapse_edges:bool=True,
            show_num_edges:bool=False,
            edge_weight_multiplier:float=1.0,
            count_min=False,
            ):
        # Load style file
        if style_file is None:
            style_file = DEFAULT_STYLE_FILE
        with open(style_file, 'r') as json_file:
            style_obj = json.load(json_file)
        # Create widget
        cyto_obj = CytoscapeWidget()
        # Set layout
        cyto_obj.set_layout(name=layout_name, nodeSpacing=10, edgeLengthVal=10)
        # Set style
        cyto_obj.set_style(style_obj[0]['style'])
        # Construct graph
        G = self.construct_nx_graph(
            node_padding,
            include_sources,
            subgraph_on,
            subgraph_on_partial,
            only_rvs,
            remove_simple_cycles,
            no_snodes,
            collapse_edges,
            show_num_edges,
            edge_weight_multiplier,
            count_min,
            )
        # Add graph
        cyto_obj.graph.add_graph_from_networkx(G, directed=True)
        # Render
        return cyto_obj
