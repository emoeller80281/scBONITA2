import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import copy
from scipy.stats.stats import spearmanr
import logging
from sklearn import preprocessing
from alive_progress import alive_bar

from cell_class import Cell
from node_class import Node
from kegg_parser import *
from rule_determination import RuleDetermination

class RuleInference:

    """Class for single-cell experiments"""

    def __init__(
        self,
        data_file,
        graph,
        dataset_name,
        network_name,
        sep,
        node_indices,
        binarize_threshold=0.001,
        sample_cells=True,
    ):

        self.node_indices = node_indices
        self.dataset_name = dataset_name
        self.network_name = network_name
        self.binarize_threshold = binarize_threshold
        self.max_samples = 15000
        self.cells = []

        # Initialize lists to store information about nodes and connections
        self.predecessors_final = []
        self.rvalues = []
        self.predecessors = []
        self.num_successors = []
        self.graph = graph

        # Initialize node attributes
        self.node_list = list(graph.nodes)  # List of nodes in the graph
        self.node_dict = {self.node_list[i]: i for i in range(len(self.node_list))}  # Dictionary for node lookup

        # Initializes an empty directed graph
        self.rule_graph = nx.empty_graph(0, create_using=nx.DiGraph)

        logging.info(f'\n-----EXTRACTING AND FORMATTING DATA-----')

        # Extract the data from the data file based on the separator, sample the cells if over 15,000 cells
        logging.info(f'Extracting cell expression data from "{data_file}"')
        self.cell_names, self.gene_names, self.data = self._extract_data(data_file, sep, sample_cells, node_indices)

        # Creating a csr sparse matrix from the dataset
        self.sparse_matrix = sparse.csr_matrix(self.data)
        logging.info(f'\tCreated sparse matrix')

        # self.gene_names = list(self.gene_names)
        # self.cell_names = list(self.cell_names)
        self.sparse_matrix.eliminate_zeros()

        # Check if there are at least 1 sample selected
        if self.data.shape[0] > 0:
            # Binarize the values in the sparse matrix
            logging.info(f'\tBinarized sparse matrix')
            
            self.binarized_matrix = preprocessing.binarize(self.sparse_matrix, threshold=binarize_threshold, copy=True)
            logging.debug(f'{self.binarized_matrix[:5,:5]}')
            
        else:
            # Handle the case where no samples were selected
            raise ValueError("No samples selected for binarization")

        self.filterData(self.binarize_threshold)

        # Calculate the node information
        self.calculate_node_information()

        # Create Node objects containing the calculated information for each node in the network
        self.nodes, self.deap_individual_length = self.create_nodes()

        # Create Cell objects
        self.create_cells()

        # Runs the rule refinement
        rule_determination = RuleDetermination(
            self.graph,
            self.network_name,
            self.dataset_name,
            self.binarized_matrix,
            self.nodes,
            self.node_dict
        )

        self.ruleset = rule_determination.infer_ruleset()

    def _extract_data(self, data_file, sep, sample_cells, node_indices):
        """
        Extract the data from the data file
        Parameters
        ----------
        data_file
        sep
        sample_cells

        Returns
        -------
        cell_names, data
        """
        with open(data_file, 'r') as file:
            reader = csv.reader(file, delimiter=sep)

            # Extract the header (cell_names)
            cell_names = next(reader)[1:]

            cell_count = len(cell_names)

            # Randomly sample the cells in the dataset
            if cell_count >= self.max_samples or sample_cells:
                logging.info(f'\tRandomly sampling {self.max_samples} cells...')
                sampled_cell_indices = np.random.choice(
                    range(cell_count),
                    replace=False,
                    size=min(self.max_samples, cell_count),
                )
                logging.info(f'\t\tNumber of cells: {len(sampled_cell_indices)}')

            else:
                sampled_cell_indices = range(cell_count)
                logging.info(f'\tLoading all {len(sampled_cell_indices)} cells...')

            # Data extraction
            data_shape = (len(node_indices), len(sampled_cell_indices))
            data = np.empty(data_shape, dtype="float")
            gene_names = []
            data_row_index = 0  # Separate index for data array
            for i, row in enumerate(reader):
                if i in node_indices:  # Only keeps the nodes involved, skips the cell name row
                    gene_names.append(row[0])

                    # Offset cell indices by 1 to skip the gene name column
                    # Treat missing/blank strings as 0.0 to avoid float conversion errors
                    selected_data = []
                    for cell_index in sampled_cell_indices:
                        value_str = row[cell_index + 1].strip()
                        if value_str in ("", "NA", "NaN", "nan"):
                            value = 0.0
                        else:
                            value = float(value_str)
                        selected_data.append(value)

                    data[data_row_index, :] = selected_data
                    data_row_index += 1

            # Convert the filtered data to a NumPy array
            logging.info("\tConverting filtered data to numpy array...")

            logging.info(f'\tFirst 2 genes: {gene_names[:2]}')
            logging.info(f'\tFirst 2 cells: {cell_names[:2]}')

            logging.info(f'\tNumber of genes: {len(gene_names)}')
            logging.info(f'\tNumber of cells: {len(cell_names)}')

            return cell_names, gene_names, data

    def filterData(self, threshold):
        """
        Filters the data to include genes with high variability
        (genes with a std dev / mean ratio above the cv_cutoff threshold)
        """
        self.cv_genes = []
        if threshold is not None:
            for i in range(0, self.sparse_matrix.get_shape()[0]):
                rowData = list(self.sparse_matrix.getrow(i).todense())
                if np.std(rowData) / np.mean(rowData) >= threshold:
                    self.cv_genes.append(self.gene_names[i])
        else:
            self.cv_genes = copy.deepcopy(self.gene_names)

    def plot_graph_from_graphml(self, network):
        G = network

        # Extract values and ensure they are within [0, 1]
        values = [node.importance_score for node in self.nodes]
        logging.debug(f'\nNormalized Values: {values}')

        # Choose a colormap
        cmap = plt.cm.Greys

        def scale_numbers(numbers, new_min, new_max):
            # Calculate the min and max of the input numbers
            old_min = min(numbers)
            old_max = max(numbers)
            
            # Scale the numbers to the new range
            scaled_numbers = []
            for num in numbers:
                scaled_num = ((num - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                scaled_numbers.append(scaled_num)
            
            return scaled_numbers

        new_min = 0.1
        new_max = 0.7
        scaled_numbers = scale_numbers(values, new_min, new_max)

        node_colors = [cmap(value) for value in scaled_numbers]

        # Map 'values' to colors
        logging.debug(f'\nNode Colors: {node_colors}')

        pos = nx.spring_layout(G, k=1)  # Layout for visualizing the graph

        # Draw the graph
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="black", font_size=10, ax=ax)

        ax.set_title("Importance Score for Each Node in the Network")
        ax.set_axis_off()  # Hide the axes
        # plt.show()

        return fig

    def calculate_node_information(self):
        """
        Calculates the information for each node in the network and stores the information as object of class Node
        from node_class.py
        """
        # Iterate over all nodes to find predecessors and calculate possible connections
        for node_num, _ in enumerate(self.node_list):
            predecessors_final = self.find_predecessors(self.rule_graph, self.node_list, self.graph, node_num)
            node_predecessors = [self.node_list.index(corr_tuple[0]) for corr_tuple in predecessors_final]
            self.predecessors.append(node_predecessors)

    def calculate_spearman_correlation(self, node, predecessors_temp):
        """
        Calculate the Spearman correlation between incoming nodes to find the top three with the
        highest correlation, used to reduce the dimensionality of the calculations.
        """
        # Find correlation between the predecessors and the node
        node_positions = [self.gene_names.index(node) for node in self.gene_names]

        # find binarized expression data for node "i"
        node_expression_data = self.binarized_matrix[node_positions[node], :].todense().tolist()[0]

        # temporarily store correlations between node "i" and all its predecessors
        predecessor_correlations = []

        for predecessor_gene in predecessors_temp:
            # find index of predecessor in the node_list from the data
            predIndex = self.node_list.index(predecessor_gene)

            # find binarized expression data for predecessor
            predData = (self.binarized_matrix[predIndex, :].todense().tolist()[0])
            mi, pvalue = spearmanr(node_expression_data, predData)

            if np.isnan(mi):
                predecessor_correlations.append(0)
            else:
                predecessor_correlations.append(mi)  # store the calculated correlation
        return predecessor_correlations

    # Finds the predecessors of each node and stores the top 3
    def find_predecessors(self, rule_graph, node_list, graph, node_index):
        """
        Find the incoming nodes for each node in the graph, store the top 3 connections as calculated by a spearman
        correlation
        Parameters
        ----------
        rule_graph
        node_list
        graph
        node_dict
        node

        Returns
        -------

        """
        # --- Find the predecessors of each node ---
        # Get NAMES of incoming nodes targeting the current node
        predecessors_temp = list(graph.predecessors(node_list[node_index]))

        # Calculate the Spearman correlation for the incoming nodes
        predecessor_correlations = self.calculate_spearman_correlation(node_index, predecessors_temp)

        # Select the top 3 predecessors of the node according to the Spearman correlation
        predecessors_final = sorted(
            zip(predecessors_temp, predecessor_correlations),
            reverse=True,
            key=lambda corrs: corrs[1], )[:3]

        # Get NAMES of successors of the node
        successors_temp = list(graph.successors(node_list[node_index]))
        self.num_successors.append(len(successors_temp))

        # Store the correlations between incoming nodes in "rvalues"
        top_three_incoming_node_correlations = sorted(predecessor_correlations, reverse=True)[:3]
        self.rvalues.append(top_three_incoming_node_correlations)

        # Append the permanent list with the top 3 predecessors for this node
        self.predecessors_final.append([pred[0] for pred in predecessors_final])

        # Add the incoming nodes and their properties to the newly created rule_graph
        for parent in predecessors_final:
            if "interaction" in list(graph[parent[0]][node_list[node_index]].keys()):
                rule_graph.add_edge(
                    parent[0],
                    node_list[node_index],
                    weight=parent[1],
                    activity=graph[parent[0]][node_list[node_index]]["interaction"],
                )
            if "signal" in list(graph[parent[0]][node_list[node_index]].keys()):
                rule_graph.add_edge(
                    parent[0],
                    node_list[node_index],
                    weight=parent[1],
                    activity=graph[parent[0]][node_list[node_index]]["signal"],
                )

        return predecessors_final

    # Calculates the inversion rules for each rule based on if the incoming nodes are inhibiting or activating
    def calculate_inversion_rules(self, node_predecessors, node_index):
        """
        Calculates the inversion rules for a node based on the graph interactions or signal for each incoming node
        Parameters
        ----------
        node

        Returns
        -------
        inversion_rules
        """

        inversion_rules = {}
        for incoming_node in list(node_predecessors):
            edge_attribute = list(self.graph[self.node_list[incoming_node]][self.node_list[node_index]].keys())

            # check the 'interaction' edge attribute
            if "interaction" in edge_attribute:
                if self.graph[self.node_list[incoming_node]][self.node_list[node_index]]["interaction"] == "i":
                    inversion_rules[incoming_node] = True
                else:
                    inversion_rules[incoming_node] = False

            # check the 'signal' edge attribute
            elif "signal" in edge_attribute:
                if self.graph[self.node_list[incoming_node]][self.node_list[node_index]]["signal"] == "i":
                    inversion_rules[incoming_node] = True
                else:
                    inversion_rules[incoming_node] = False

            # for some reason, when I used a modified processed graphml file as a custom graphml file I needed to use this method
            else:
                for _, value in self.graph[self.node_list[incoming_node]][self.node_list[node_index]].items():
                    for attribute, value in value.items():
                        if attribute == "signal" or "interaction":
                            if value == "i":
                                inversion_rules[incoming_node] = True
                            else:
                                inversion_rules[incoming_node] = False

        return inversion_rules

    # 1.6 Creates nodes containing the information calculated from the graph
    def create_nodes(self):
        """
        Creates Node class objects using the information calculated in the rest of the calculate_node_information
        function
        """
        gene_name_to_index = {gene_name: gene_index for gene_index, gene_name in enumerate(self.gene_names)}

        nodes = []
        rule_index = 0
        with alive_bar(len(self.node_list)) as bar:
            for node_index, node_name in enumerate(self.node_list):
                name = node_name
                # Safely retrieve predecessors and put them into a dictionary where key = node index, value = node name
                predecessor_indices = self.predecessors[node_index] if node_index < len(self.predecessors) else []
                predecessors = {}
                for index in predecessor_indices:
                    inverted_node_dict = {v: k for k, v in self.node_dict.items()}
                    predecessors[index] = inverted_node_dict[index]

                node_inversions = self.calculate_inversion_rules(predecessors, node_index)

                # Create a new Node object
                node = Node(name, node_index, predecessors, node_inversions)

                # Find the dataset row index of the gene
                node.dataset_index = gene_name_to_index.get(node_name)

                nodes.append(node)
                bar()

        return nodes, rule_index

    def create_cells(self):
        """
        Creates Cell objects containing the cells gene expression value for each gene
        """
        full_matrix = self.binarized_matrix.todense()

        # Create cell objects
        cells = []
        for cell_index, cell_name in enumerate(self.cell_names):
            cell = Cell(cell_index)
            cell.name = cell_name
            for row_num, row in enumerate(full_matrix):
                row_array = np.array(row).flatten()
                try:
                    cell.expression[self.gene_names[row_num]] = row_array[cell_index]
                except IndexError as e:
                    logging.debug(
                        f'Encountered error {e} at row {row_num}, col {cell_index}. If at the last gene position, ignore')
            cells.append(cell)

        return cells