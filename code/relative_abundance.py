import argparse
import logging
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
from network_class import Network
from sklearn import preprocessing
import scipy.sparse as sparse
import csv
import seaborn as sns
from matplotlib.cm import ScalarMappable

from metadata_parser import metadata_parser
from user_input_prompts import *
from file_paths import file_paths

def extract_data(data_file, sep, sample_cells, node_indices, max_samples):
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
        if cell_count >= max_samples or sample_cells:
            logging.debug(f'\tRandomly sampling {max_samples} cells...')
            sampled_cell_indices = np.random.choice(
                range(cell_count),
                replace=False,
                size=min(max_samples, cell_count),
            )
            logging.debug(f'\t\tNumber of cells: {len(sampled_cell_indices)}')

        else:
            sampled_cell_indices = range(cell_count)
            logging.debug(f'\tLoading all {len(sampled_cell_indices)} cells...')

        # Data extraction
        data_shape = (len(node_indices), len(sampled_cell_indices))
        data = np.empty(data_shape, dtype="float")
        gene_names = []
        data_row_index = 0  # Separate index for data array

        for i, row in enumerate(reader):
            if (i + 1) in node_indices:  # Adjust index for skipped header
                gene_names.append(row[0])
                # Offset cell indices by 1 to skip the gene name column
                selected_data = [float(row[cell_index + 1]) for cell_index in sampled_cell_indices]
                data[data_row_index, :] = selected_data
                data_row_index += 1

        return cell_names, gene_names, data

def figure_graph(experimental_network, dataset_name, relative_abundances):
    G = experimental_network.network       

    # Extracting importance scores and applying log transformation to relative abundances
    importance_scores = []
    node_colors = []

    for node in experimental_network.nodes:
        node_colors.append(relative_abundances[node.name])
        importance_scores.append(node.importance_score)

    # Apply log transformation to relative abundances to get log fold change
    log_fold_changes = np.log2(np.array(node_colors))  # Adding a small value to avoid log(0)

    # Define minimum and maximum node sizes
    min_node_size = 100  # Minimum node size
    max_node_size = 1000  # Maximum node size

    # Scaling importance scores for node sizes within the specified range
    min_importance_score = min(importance_scores)
    max_importance_score = max(importance_scores)
    scaled_importance_scores = [
        ((iscore - min_importance_score) / (max_importance_score - min_importance_score) * (max_node_size - min_node_size) + min_node_size)
        for iscore in importance_scores
    ]

    cmap = plt.cm.coolwarm
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    # Clamp the values between -3 and 3 to not drown out smaller changes with huge outliers
    log_fold_changes = np.clip(log_fold_changes, -3, 3)

    colors = [cmap(norm(value)) for value in log_fold_changes]

    # Drawing the graph
    # Create custom legend
    blue_patch = mpatches.Patch(color='blue', label=f'Decreased expression in {experimental_group}')
    red_patch = mpatches.Patch(color='red', label=f'Increased expression in {experimental_group}')
    
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap='coolwarm', node_size=scaled_importance_scores, font_size=10, ax=ax)

    # Create a ScalarMappable and initialize a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Fold Change')

    ax.set_title(f"Importance Score and Log2 Fold Change for network {experimental_network.name.split('_')[0]} for dataset {dataset_name}")
    plt.legend(handles=[blue_patch, red_patch], title=f'Log2 Fold Change of {experimental_group} compared to {control_group}')

    return fig

def plot_bootstrap_histogram(bootstrap_scores, pathway_modulation):
    # Plotting
    figure = plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_scores, kde=True, color="skyblue", bins=30)
    plt.axvline(x=pathway_modulation, color='red', linestyle='--', label=f'Pathway Modulation Score: {round(pathway_modulation, 2)}\np = {round(p_value, 4)}')
    plt.title(f'{dataset_name.capitalize()} {experimental_group.capitalize()} vs {control_group.capitalize()} Distribution of Bootstrap Pathway Modulation Scores for {network.name}')
    plt.xlabel('Pathway Modulation Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    return figure

def relative_abundance_arguments(control_group, experimental_group):
    control_group = check_control_group(control_group)
    experimental_group = check_experimental_group(experimental_group)
    
    logging.info(f'\t\tControl Group: {control_group}')
    logging.info(f'\t\tExperimental Group: {experimental_group}')

    return control_group, experimental_group

def plot_abundance_heatmap(gene_names, mean_expression_control, mean_expression_experimental, control_group_name, experimental_group_name, network_name):
    data = {
        'Gene': gene_names,
        f'{control_group_name}': mean_expression_control,
        f'{experimental_group_name}': mean_expression_experimental
    }
    df = pd.DataFrame(data).set_index('Gene').transpose()

    # Create the figure with modified figsize to affect the heatmap height
    fig = plt.figure(figsize=(18, 6))  # Reduced height from 8 to 6
    gs = plt.GridSpec(1, 2, width_ratios=[30, 1])  # Adjust grid specification for a narrower colorbar section
    ax = fig.add_subplot(gs[0])
    
    # Create the heatmap without automatically adding a colorbar
    heatmap = sns.heatmap(df, annot=False, cmap='coolwarm', cbar=False, ax=ax, vmin=0, vmax=1)

    # Customizing the heatmap
    plt.title(f'{network_name} Mean Expression Values Between {control_group_name} and {experimental_group_name}', fontsize=20)
    ax.set_ylabel('Groups', fontsize=18)
    ax.set_xlabel('Genes', fontsize=18)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    # Manually adding a colorbar
    cax = fig.add_subplot(gs[1])
    plt.colorbar(heatmap.collections[0], cax=cax, orientation='vertical')
    cax.set_ylabel('Mean Expression Value', fontsize=16)

    plt.tight_layout()

    return fig

def bubble_plot(network_names, p_values):
    # Assuming p_values is a list containing all your p-values before correction
    num_tests = len(p_values)  # The number of tests
    alpha = 0.05  # The typical significance level

    # Perform Bonferroni correction
    bonferroni_corrected_p_values = [min(p * num_tests, 1.0) for p in p_values]

    # Calculate -log10 of Bonferroni-corrected p-values
    # Added a small constant inside the log10 function to avoid log10(0)
    neg_log10_bonferroni_corrected_p_values = [-np.log10(p + 1e-5) for p in bonferroni_corrected_p_values]

    # Sample sizes for the bubbles (you might want to scale these based on another metric)
    bubble_sizes = [300 for _ in range(len(network_names))]  # Uniform size for now

    # Create the bubble plot
    figure, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        neg_log10_bonferroni_corrected_p_values,
        range(len(network_names)),
        s=bubble_sizes,
        cmap="viridis"
    )

    # Set the y-axis to show the network names
    ax.set_yticks(range(len(network_names)))
    ax.set_yticklabels(network_names)

    # Set the x-axis label
    ax.set_xlabel('-log10(adjusted p-value)')

    # Invert y-axis to have the first entry at the top
    ax.invert_yaxis()

    # Show grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()

    return figure

def parse_data_and_scale(network, gene_list, scaler=None, fit_scaler=False):
    gene_data = {}
    dataset = network.dataset.todense()
    for node in network.nodes:
        if node.name in gene_list:
            gene_data[node.name] = [value for value in dataset[node.index]]

    # Ensure gene data is ordered according to gene_list
    ordered_gene_data = [gene_data[gene] for gene in gene_list if gene in gene_data]
    ordered_gene_data = np.array(ordered_gene_data, dtype=float).T.squeeze()  # Transpose to match the original data structure

    # Scale the data
    if fit_scaler:
        scaler.fit(ordered_gene_data)

    scaled_data = scaler.transform(ordered_gene_data)

    return scaled_data

if __name__ == '__main__':
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Set the logging level for output
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()

    add_dataset_name_arg(parser)
    add_metadata_file_arg(parser)
    add_metadata_sep_arg(parser)
    add_dataset_file_arg(parser)
    add_dataset_sep_arg(parser)
    add_control_group_arg(parser)
    add_experimental_group_arg(parser)
    add_cell_name_index(parser)
    add_group_indices(parser)
    add_header(parser)
    add_overwrite(parser)
    add_list_of_kegg_pathways(parser)
    add_organism_code(parser)

    args = parser.parse_args()

    dataset_name = check_dataset_name(args.dataset_name)
    dataset_file = check_dataset_file(args.dataset_file)
    dataset_sep = check_separator(args.dataset_sep)

    metadata_file = check_metadata_file(args.metadata_file)
    metadata_sep = check_separator(args.metadata_sep)

    control_group = args.control_group
    experimental_group = args.experimental_group

    cell_name_index = check_cell_name_index(args.cell_name_index)
    group_indices = check_group_indices(args.group_indices)
    header = check_header(args.header)
    overwrite = args.overwrite

    organism = args.organism
    network_names = args.list_of_kegg_pathways

    # If no network is specified, get all of the rulesets for the dataset
    if network_names[0] == "":
        network_names_list = []
        for filename in os.listdir(f'{file_paths["rules_output"]}/{dataset_name}_rules/'):
            network = filename.split('_')[0]
            network_names_list.append(network)
        network_name_set = set(network_names_list)
        network_names = list(network_name_set)

    txt = f'Relative Abundance for {dataset_name} {control_group} vs {experimental_group}'
    logging.info(f' -----{"-" * len(txt)}----- '.center(20))
    logging.info(f'|     {txt.upper()}     |'.center(20))
    logging.info(f' -----{"-" * len(txt)}----- '.center(20))

    # Split the dataset into groups
    split_datasets, groups, cell_indices_per_group = metadata_parser(
        metadata_file,
        metadata_sep, 
        dataset_file, 
        dataset_sep,
        cell_name_index,
        group_indices,
        header,
        overwrite
    )
        
    if len(network_names) > 0:

        for network_name in network_names:
            if organism not in network_name:
                network_name = organism + network_name
            # Path to the ruleset pickle file
            ruleset_pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/ruleset_pickle_files/{dataset_name}_{network_name}.ruleset.pickle'
            network_pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/{dataset_name}_{network_name}.network.pickle'

            # Check to make sure the ruleset pickle file exists
            if os.path.exists(ruleset_pickle_file_path):
                logging.debug('ruleset pickle file exists')
            else:
                logging.error(f'\n\nERROR: ruleset pickle file not found: {ruleset_pickle_file_path}')
                assert FileNotFoundError(ruleset_pickle_file_path)

            # Check to make sure the network pickle file exists
            if os.path.exists(network_pickle_file_path):
                # Load the ruleset object for the network
                ruleset = pickle.load(open(ruleset_pickle_file_path, "rb"))
                network = pickle.load(open(network_pickle_file_path, "rb"))

                # Find the importance score for each of the split datasets
                for group_num, dataset_path in enumerate(split_datasets):

                    # Join the groups
                    group = groups[group_num]
                    # Specify the path to the group network pickle file
                    network_folder = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/{dataset_name}_{group}_pickle_files'
                    os.makedirs(network_folder, exist_ok=True)

                    network_file_path = f'{network_folder}/{dataset_name}_{network_name}_{group}.network.pickle'
                    
                    # Check if the group file exists for this dataset and if the user passed in the overwrite argument
                    if overwrite_check(overwrite, network_file_path) == True or os.path.exists(network_file_path) == False:
                        # Store the network information in a pickle file for each group
                        new_network = Network(name=f'{network_name}_{group}')
                        new_network.nodes = network.nodes
                        new_network.rulesets = ruleset.ruleset
                        new_network.network = network.network
                        
                        # Extract the data from each dataset
                        sample_cells = True
                        cell_names, gene_names, split_dataset = extract_data(dataset_path, dataset_sep, sample_cells, ruleset.node_indices, ruleset.max_samples)
                        new_network.cells = cell_indices_per_group[group]

                        # Append the row index for each gene in the network to a list
                        gene_indices = []
                        for row_index, row in enumerate(split_dataset):
                            if row[0] in gene_names:
                                gene_indices.append(row_index)

                        # Create a sparse matrix
                        split_sparse_matrix = sparse.csr_matrix(split_dataset)

                        split_sparse_matrix.eliminate_zeros()

                        # Replace NaN with 0 and infinity with a large finite number
                        split_sparse_matrix.data = np.nan_to_num(split_sparse_matrix.data, nan=0.0, posinf=1e10,
                                                                 neginf=-1e10)

                        if split_dataset.shape[0] > 0:
                            split_binarized_matrix = preprocessing.binarize(split_sparse_matrix, threshold=ruleset.binarize_threshold, copy=True)
                            new_network.dataset = split_binarized_matrix
                        else:
                            raise ValueError("No samples selected for binarization")
                        
                        network_folder = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/{dataset_name}_{group}_pickle_files'
                        
                        os.makedirs(network_folder, exist_ok=True)
                        logging.info(f'\tSaving network pickle file {group} dataset for {network_name}')

                        # Save the new group network pickle file
                        pickle.dump(new_network, open(network_file_path, 'wb'))
                    else:
                        logging.info(f'\t\tUsing existing group network {network.name} file for {group} ')
                # Save the full network pickle file with the cell indices per group
                pickle.dump(network, open(network_pickle_file_path, 'wb'))
            else:
                logging.error(f'\n\nERROR: network pickle file not found: {network_pickle_file_path}')
                
    else:
        logging.error(f'\n\tERROR: No networks loaded')
        exit(1)

    control_group, experimental_group = relative_abundance_arguments(control_group, experimental_group)

    logging.info(f'\n----- Loading {dataset_name} networks -----')

    # Specify the path to the network pickle files for this dataset
    while True:
        pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files'
        if os.path.exists(pickle_file_path):
            logging.debug(f'\tPickle files found')
            break
        else:
            dataset_name = input(f'{dataset_name} network pickle files do not exist, check spelling and try again: ')

    # Find and load the group 1 pickle files
    control_group_networks = []
    while True:
        group_path = f'{dataset_name}_{control_group}_pickle_files'
        control_group_path = f'{pickle_file_path}/{group_path}/{dataset_name}_*_{control_group}.network.pickle'

        logging.info(f'\tLoading CONTROL group networks')

        if len(glob.glob(control_group_path)) > 0:
            for pickle_file in glob.glob(control_group_path):
                try:
                    network = pickle.load(open(pickle_file, "rb"))
                    dense_dataset = network.dataset.todense()
                    
                    # Check if the network's dataset has more than one column
                    if dense_dataset.shape[1] > 1:
                        logging.info(f'\t\tLoaded network {network.name}')
                        control_group_networks.append(network)
                    else:
                        logging.info(f'Skipped network {network.name} due to it having only one column')
                except:
                    assert FileNotFoundError("Network pickle file not found")
            break
        else:
            control_group = input(f'{control_group} network pickle files do not exist, check spelling and try again If you continue to have issues, re-generate the data files for the groups: ')
    
    # Find and load the group 2 pickle files
    experimental_group_networks = []
    while True:
        group_path = f'{dataset_name}_{experimental_group}_pickle_files'
        experimental_group_path = f'{pickle_file_path}/{group_path}/{dataset_name}_*_{experimental_group}.network.pickle'

        logging.info(f'\n\tLoading EXPERIMENTAL group networks')

        if len(glob.glob(experimental_group_path)) > 0:
            for pickle_file in glob.glob(experimental_group_path):
                if len(pickle_file) > 0:
                    network = pickle.load(open(pickle_file, "rb"))
                    dense_dataset_exp = network.dataset.todense()

                    if dense_dataset.shape[1] > 1:
                        logging.info(f'\t\tLoaded network {network.name}')
                        experimental_group_networks.append(network)
                    else:
                        logging.info(f'Skipped network {network.name} due to it having only one column')                    
                else:
                    assert FileNotFoundError("Network pickle file not found")
            break
        else:
            experimental_group = input(f'{experimental_group} network pickle files do not exist, check spelling and try again: ')
    
    max_abs_scaler = MaxAbsScaler()
    p_values = []
    network_names = [control_group_network.name.split('_')[0] for control_group_network in control_group_networks]

    logging.info(f'\n----- Calculating Relative Abundance between groups {experimental_group} and {control_group} -----')
    for control_group_network in control_group_networks:
        for experimental_group_network in experimental_group_networks:
            if control_group_network.name.split('_')[0] == experimental_group_network.name.split('_')[0]:
                network_name = control_group_network.name.split("_")[0]
                logging.info(f'\tNetwork: {network_name}')
                
                # Find the common genes genes present in the control and experimental groups for the network
                control_group_nodes = [node.name for node in control_group_network.nodes]
                experimental_group_nodes = [node.name for node in experimental_group_network.nodes]

                control_group_genes_set = set(control_group_nodes)
                experimental_group_genes_set = set(experimental_group_nodes)
                
                common_genes_set = control_group_genes_set.intersection(experimental_group_genes_set)
                gene_list = list(common_genes_set)

                # Scale the data for the control and experimental group based on the control group
                control_group_data = parse_data_and_scale(control_group_network, gene_list, scaler=max_abs_scaler, fit_scaler=True)
                experimental_group_data = parse_data_and_scale(experimental_group_network, gene_list, scaler=max_abs_scaler, fit_scaler=False)

                min_num_cells = min(control_group_data.shape[0], experimental_group_data.shape[0])

                # Calculate the relative abundance as the mean expression of the experimental / mean expression of the control
                mean_expression_control = np.mean(control_group_data, axis=0)
                mean_expression_experimental = np.mean(experimental_group_data, axis=0)

                stdev_expression_control = np.std(control_group_data, axis=0)
                stdev_expression_experimental = np.std(experimental_group_data, axis=0)

                relative_abundances = (np.round(mean_expression_experimental + 1e-3,3)) / (np.round(mean_expression_control+ 1e-3,3)) # pseudocount added to avoid large changes based on small numbers

                file_path = f'{file_paths["relative_abundance_output"]}/{dataset_name}/{experimental_group}_vs_{control_group}'
                filename = f'{control_group_network.name.split("_")[0]}_{dataset_name}_{experimental_group}_vs_{control_group}_relative_abundance'
                
                figure = plot_abundance_heatmap(gene_list, mean_expression_control, mean_expression_experimental, control_group, experimental_group, network_name)
                png_file_path = f'{file_path}/png_files'
                os.makedirs(png_file_path, exist_ok=True)
                figure.savefig(f'{png_file_path}/{control_group_network.name}_{dataset_name}_heatmap_expression.png', format="png")

                plt.close(figure)
                
                
                # Calculate the pathway modulation score
                pathway_modulation = 0

                for node_number, shared_node in enumerate(gene_list):
                    
                    # Set the relative abundance for group 1 vs group 2
                    for node in experimental_group_network.nodes:
                        if shared_node == node.name:
                            node_score = relative_abundances[node_number] * stdev_expression_experimental[node_number] * node.importance_score
                            pathway_modulation += node_score
                
                logging.info(f'\t\tPathway Modulation Score: {pathway_modulation}')

                # Bootstrapping to find the P value
                n_iterations = 5000
                bootstrap_scores = []

                # Bootstrap process
                # Creates a custom distribution of pathway modulation scores by randomly choosing relative abundance scores
                # for each node and calculating a new pathway modulation score using the original stdev and importance scores
                # for each node. This creates a distribution of pathway modulation scores that could occur by chance given the
                # observed data. 
                logging.info(f'\t\tCalculating p-value with bootstrapping:')
                for i in range(n_iterations):
                    bootstrap_modulation = 0
                    # Resample RA values with replacement
                    resampled_RAs = np.random.choice(relative_abundances, size=len(relative_abundances), replace=True)

                    # Calculate bootstrap modulation score
                    bootstrap_modulation = np.sum(resampled_RAs *
                                                    stdev_expression_experimental * 
                                                    [node.importance_score for node in experimental_group_network.nodes if node.name in gene_list])
                    
                    bootstrap_scores.append(bootstrap_modulation)
                
                # Step 1: Calculate the absolute difference of the original score from the bootstrap mean
                original_abs_difference = abs(pathway_modulation - np.mean(bootstrap_scores))

                # Step 2: Calculate absolute differences for bootstrap scores and
                # then calculate the proportion of bootstrap scores that are as extreme as the original score
                extreme_count = np.sum(np.abs(bootstrap_scores - np.mean(bootstrap_scores)) >= original_abs_difference)
                proportion_extreme = extreme_count / len(bootstrap_scores)

                # Step 3: Calculate the p-value (two-tailed)
                p_value_two_tailed = 2 * proportion_extreme
                
                # Ensure p-value does not exceed 1
                p_value = min(p_value_two_tailed, 1.0)  
                
                # Adjust for when the p-value is too small for the computer to represent
                if p_value <= 0:
                    log_p_value = 0 
                else:
                    log_p_value = -np.log10(p_value + 1e-10)

                logging.info(f'\t\t\tP-value: {p_value}')
                logging.info(f'\t\t\t-log10(P-value): {log_p_value}')

                p_values.append(p_value)

                # Create the paths to the relative abundance results
                text_file_path = f'{file_path}/text_files'
                
                svg_file_path = f'{file_path}/svg_files'

                os.makedirs(text_file_path, exist_ok=True)
                
                os.makedirs(svg_file_path, exist_ok=True)

                node_abundances = {}

                # Write the relative abundance results as a text file
                text_file_name = f'{filename}.txt'
                with open(text_file_path + '/' + text_file_name, 'w') as abundance_file:
                    abundance_file.write(f'node,importance_score,experimental_mean,control_mean,{experimental_group}_vs_{control_group},node_percent_pm_score\n')
                    for node_number, shared_node in enumerate(gene_list):
                        
                        # Set the relative abundance for group 1 vs group 2
                        for node in experimental_group_network.nodes:
                            if shared_node == node.name:
                                # relative abundance negative to show how increased or decreased experimental is compared to control
                                rounded_mean_expr = round(mean_expression_experimental[node_number] + 0.001,3)
                                rounded_mean_ctrl = round(mean_expression_control[node_number] + 0.001,3)
                                node_score = relative_abundances[node_number] * stdev_expression_experimental[node_number] * node.importance_score
                                node_impact = round(node_score / pathway_modulation,3)

                                abundance_file.write(f'{node.name},{round(node.importance_score,2)},{rounded_mean_expr},{rounded_mean_ctrl},{round(relative_abundances[node_number],3)},{node_impact}\n')
                                node_abundances[node.name] = relative_abundances[node_number]
                
                # Create the relative abundance figures in png and svg format
                relative_abundance_graph = figure_graph(experimental_group_network, dataset_name, node_abundances)
                bootstrap_fig = plot_bootstrap_histogram(bootstrap_scores, pathway_modulation)

                figure_path_png = f'{png_file_path}/{control_group_network.name.split("_")[0]}_{dataset_name}_{experimental_group}_vs_{control_group}.png'
                figure_path_svg = f'{svg_file_path}/{control_group_network.name.split("_")[0]}_{dataset_name}_{experimental_group}_vs_{control_group}.svg'

                relative_abundance_graph.savefig(figure_path_png, format='png')
                relative_abundance_graph.savefig(figure_path_svg, format='svg')

                bootstrap_fig.savefig(f'{png_file_path}/{experimental_group_network.name.split("_")[0]}_bootstrap_histogram.png', format='png')
                
                plt.close(bootstrap_fig)
                plt.close(relative_abundance_graph)

    bubble_plot_fig = bubble_plot(network_names, p_values)
    bubble_plot_fig.show()
    bubble_plot_fig.savefig(f'{png_file_path}/bubbleplot_histogram.png', format='png')
    plt.close(bubble_plot_fig)
    logging.info(f'\nResults saved to: "relative_abundance_output/{dataset_name}/{experimental_group}_vs_{control_group}"\n')




