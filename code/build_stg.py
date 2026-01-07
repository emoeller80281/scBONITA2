import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import logging
import json
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from file_paths import file_paths

logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset_name = "george_hiv"
organism = "hsa"
pathway = "05417"

# Output directory
stg_output_dir = f'{file_paths["trajectories"]}/{dataset_name}_{organism}{pathway}/stg_output'
os.makedirs(stg_output_dir, exist_ok=True)

# ============================================================================
# STEP 1: Load all trajectory files and build state library
# ============================================================================
logging.info("STEP 1: Loading trajectory files and building state library...")

cell_trajectory_dir = f'{file_paths["trajectories"]}/{dataset_name}_{organism}{pathway}/text_files/cell_trajectories'
trajectory_files = sorted([f for f in os.listdir(cell_trajectory_dir) if f.endswith('.csv')])

all_trajectories = []
all_genes = []
state_library = {}  # Maps tuple of binary state -> unique_state_id
state_id_counter = 0
cell_trajectories_mapped = {}  # Maps cell_number -> list of state transitions

logging.info(f"Found {len(trajectory_files)} trajectory files")

# First pass: collect all unique states
for file in trajectory_files:
    df = pd.read_csv(f'{cell_trajectory_dir}/{file}', header=None)
    df.columns = ['Gene'] + [f'Time{i}' for i in range(1, df.shape[1])]
    df.set_index('Gene', inplace=True)
    
    genes = df.index.tolist()
    if not all_genes:
        all_genes = genes
    
    # Extract cell number from filename (pattern cell_<number>_trajectory.csv)
    cell_number = int(file.split('_')[1])
    
    # Convert each column (time step) to a binary state tuple
    time_steps = df.columns
    states_for_cell = []
    
    for time_step in time_steps:
        # Get the binary state at this time step
        state_values = df[time_step].values
        state_tuple = tuple(state_values.astype(int))  # Convert to binary integers
        states_for_cell.append(state_tuple)
        
        # Add to state library if not seen before
        if state_tuple not in state_library:
            state_library[state_tuple] = state_id_counter
            state_id_counter += 1
    
    cell_trajectories_mapped[cell_number] = states_for_cell
    all_trajectories.append({
        'cell_number': cell_number,
        'genes': genes,
        'dataframe': df,
        'states': states_for_cell
    })

logging.info(f"Built state library with {len(state_library)} unique states")

# ============================================================================
# STEP 2: Identify attractors and state transitions
# ============================================================================
logging.info("\nSTEP 2: Identifying attractors and state transitions...")

state_transitions = defaultdict(list)  # Maps (state_from, state_to) -> count
attractor_identification = defaultdict(int)  # Maps state -> count of cells reaching it
cell_attractor_mapping = {}  # Maps cell -> attractor it reaches
states_in_compressed_trajectories = set()  # Track which states actually appear in compressed trajectories

for cell_number, states in cell_trajectories_mapped.items():
    # Collapse consecutive duplicate states so STG updates only reflect unique consecutive states
    if not states:
        continue

    unique_states = []
    prev_state = None
    for s in states:
        if prev_state is None or s != prev_state:
            unique_states.append(s)
            states_in_compressed_trajectories.add(s)
        prev_state = s

    # The attractor is the last (stable) state in the compressed trajectory
    attractor = unique_states[-1]
    attractor_identification[attractor] += 1
    cell_attractor_mapping[cell_number] = attractor

    # Record state transitions along the compressed trajectory (no self-loops)
    for i in range(len(unique_states) - 1):
        state_from = unique_states[i]
        state_to = unique_states[i + 1]
        state_transitions[(state_from, state_to)].append(cell_number)

# Find the largest attractors
attractor_counts = Counter(attractor_identification)
largest_attractors = attractor_counts.most_common(10)

logging.info(f"Identified {len(attractor_counts)} unique attractors")
logging.info("Top 10 largest attractors:")
for attractor, count in largest_attractors:
    attractor_id = state_library[attractor]
    logging.info(f"  Attractor {attractor_id}: {count} cells (state: {attractor})")

# ============================================================================
# STEP 2B: Load cell group information (HIV vs Healthy)
# ============================================================================
logging.info("\nSTEP 2B: Loading cell group information...")

# Load cell group file to map cells to condition (HIV or Healthy)
# Format: index cell_barcode group (space-separated, no header)
cell_group_file = f'{file_paths["metadata"]}/george_HIV_metadata.txt'
cell_to_group = {}

if os.path.exists(cell_group_file):
    try:
        # Read metadata file with space separator, no header
        # Columns: 0=index, 1=cell_barcode, 2=group
        cell_group_df = pd.read_csv(cell_group_file, sep=' ', header=None, dtype=str)
        
        # Map cell_number (index) to group
        for _, row in cell_group_df.iterrows():
            try:
                cell_index = int(row[0].strip('"'))  # Remove quotes from index
                group = row[2].strip('"')  # Remove quotes from group
                cell_to_group[cell_index] = group
            except (ValueError, IndexError) as e:
                logging.debug(f"Could not parse metadata row: {e}")
                continue
        
        logging.info(f"Loaded group information for {len(cell_to_group)} cells")
        logging.info(f"Groups found: {set(cell_to_group.values())}")
    except Exception as e:
        logging.warning(f"Could not load cell group file: {e}")
else:
    logging.warning(f"Cell group file not found: {cell_group_file}")

# Track which cells (and their groups) pass through each state
state_to_cells = defaultdict(set)  # Maps state -> set of cell_numbers
state_to_groups = defaultdict(lambda: defaultdict(int))  # Maps state -> {group: count}

for cell_number, states in cell_trajectories_mapped.items():
    # Get the group for this cell
    cell_group = cell_to_group.get(cell_number, 'Unknown')
    
    # Track all states in this cell's trajectory
    for state in set(states):  # Use set to count each state only once per cell
        state_to_cells[state].add(cell_number)
        state_to_groups[state][cell_group] += 1

logging.info(f"Mapped {len(state_to_cells)} states to cell group composition")

# ============================================================================
# STEP 3: Build the state transition graph
# ============================================================================
logging.info("\nSTEP 3: Building state transition graph...")

G = nx.DiGraph()

# Add nodes only for states that appear in compressed trajectories
for state in states_in_compressed_trajectories:
    state_id = state_library[state]
    is_attractor = state in attractor_identification
    attractor_size = attractor_identification.get(state, 0)
    
    # Determine dominant group(s) for this state
    group_counts = state_to_groups.get(state, {})
    dominant_groups = []
    max_count = max(group_counts.values()) if group_counts else 0
    if max_count > 0:
        dominant_groups = [g for g, count in group_counts.items() if count == max_count]
    dominant_group = ','.join(sorted(dominant_groups)) if dominant_groups else 'Unknown'
    
    # Count cells by group for this state
    hiv_count = group_counts.get('HIV', 0)
    healthy_count = group_counts.get('Healthy', 0)
    unknown_count = group_counts.get('Unknown', 0)
    
    G.add_node(state_id, 
               state=str(state),  # Convert tuple to string
               is_attractor=is_attractor,
               attractor_size=attractor_size,
               dominant_group=dominant_group,
               hiv_cells=hiv_count,
               healthy_cells=healthy_count,
               unknown_cells=unknown_count,
               total_cells=hiv_count + healthy_count + unknown_count)

# Add edges for state transitions
for (state_from, state_to), cells in state_transitions.items():
    state_id_from = state_library[state_from]
    state_id_to = state_library[state_to]
    transition_count = len(cells)
    # Store cell list as string to avoid GraphML type issues
    G.add_edge(state_id_from, state_id_to, 
               weight=transition_count,
               cell_count=transition_count)

logging.info(f"STG has {G.number_of_nodes()} nodes (from {len(state_library)} total unique states) and {G.number_of_edges()} edges")
logging.info(f"Filtered out {len(state_library) - len(states_in_compressed_trajectories)} states eliminated during consecutive-duplicate compression")

# Validation: Check for nodes that should have paths to attractors
orphaned_nodes = []
for node in G.nodes():
    if not G.nodes[node]['is_attractor']:
        # Non-attractor node should have at least one outgoing edge
        if G.out_degree(node) == 0:
            orphaned_nodes.append(node)
            logging.warning(f"Node {node} is not an attractor but has no outgoing edges!")

if orphaned_nodes:
    logging.warning(f"Found {len(orphaned_nodes)} orphaned non-attractor nodes")
else:
    logging.info("All non-attractor nodes have outgoing edges")

# ============================================================================
# STEP 4: Save state library and mappings
# ============================================================================
logging.info("\nSTEP 4: Saving state library and mappings...")

# Save state library
state_library_df = pd.DataFrame([
    {
        'state_id': state_id,
        'state': str(state),
        'is_attractor': state in attractor_identification,
        'cells_reaching': attractor_identification.get(state, 0)
    }
    for state, state_id in sorted(state_library.items(), key=lambda x: x[1])
])
state_library_df.to_csv(f'{stg_output_dir}/state_library.csv', index=False)
logging.info(f"Saved state library to {stg_output_dir}/state_library.csv")

# Save cell-to-attractor mapping
cell_attractor_df = pd.DataFrame([
    {
        'cell_number': cell,
        'attractor_state_id': state_library[attractor],
        'attractor_state': np.array(attractor)
    }
    for cell, attractor in cell_attractor_mapping.items()
])
cell_attractor_df.to_csv(f'{stg_output_dir}/cell_attractor_mapping.csv', index=False)
logging.info(f"Saved cell-to-attractor mapping to {stg_output_dir}/cell_attractor_mapping.csv")

# Save state transition graph
graph_data = {
    'nodes': [
        {
            'id': node,
            'state': str(G.nodes[node]['state']),
            'is_attractor': G.nodes[node]['is_attractor'],
            'attractor_size': G.nodes[node]['attractor_size']
        }
        for node in G.nodes()
    ],
    'edges': [
        {
            'source': u,
            'target': v,
            'weight': G[u][v]['weight'],
            'cell_count': G[u][v]['cell_count']
        }
        for u, v in G.edges()
    ]
}
with open(f'{stg_output_dir}/stg_graph.json', 'w') as f:
    json.dump(graph_data, f, indent=2)
logging.info(f"Saved STG graph to {stg_output_dir}/stg_graph.json")


# ============================================================================
# STEP 5: Calculate pseudotime for all states
# ============================================================================
logging.info("\nSTEP 5: Calculating pseudotime for all states...")

# Dictionary to store pseudotime for each state relative to its attractor
state_pseudotime = {}

# Process each attractor separately
for attractor_state in attractor_identification.keys():
    attractor_id = state_library[attractor_state]
    
    # Find all states that can reach this attractor (using ancestors)
    states_leading_to_attractor = nx.ancestors(G, attractor_id)
    states_leading_to_attractor.add(attractor_id)  # Include the attractor itself
    
    # Calculate shortest path distance from each state to the attractor
    state_distances = {}
    for state_id in states_leading_to_attractor:
        try:
            # Get shortest path length (number of steps to attractor)
            distance = nx.shortest_path_length(G, state_id, attractor_id)
            state_distances[state_id] = distance
        except nx.NetworkXNoPath:
            # If no path exists (shouldn't happen), skip this state
            continue
    
    # Find min and max distances for this attractor
    if state_distances:
        s_min = min(state_distances.values())
        s_max = max(state_distances.values())
        
        # Calculate pseudotime for each state using the formula:
        # pState_i = 1 - (s - s_min) / (s_max - s_min)
        for state_id, s in state_distances.items():
            if s_max == s_min:
                # All states are at the same distance (edge case)
                pseudotime = 1.0
            else:
                pseudotime = 1.0 - (s - s_min) / (s_max - s_min)
            
            # Store pseudotime with attractor reference
            state_pseudotime[state_id] = {
                'pseudotime': pseudotime,
                'steps_to_attractor': s,
                'attractor_id': attractor_id,
                'attractor_state': str(attractor_state)
            }

# Add pseudotime as node attribute
for node in G.nodes():
    if node in state_pseudotime:
        G.nodes[node]['pseudotime'] = state_pseudotime[node]['pseudotime']
        G.nodes[node]['steps_to_attractor'] = state_pseudotime[node]['steps_to_attractor']
    else:
        # States not leading to any attractor (shouldn't happen in our case)
        G.nodes[node]['pseudotime'] = -1.0
        G.nodes[node]['steps_to_attractor'] = -1
        if not G.nodes[node]['is_attractor']:
            logging.warning(f"Non-attractor node {node} has no pseudotime! Out-degree: {G.out_degree(node)}, In-degree: {G.in_degree(node)}")

# Save pseudotime data
pseudotime_df = pd.DataFrame([
    {
        'state_id': state_id,
        'state': str(G.nodes[state_id]['state']),
        'pseudotime': info['pseudotime'],
        'steps_to_attractor': info['steps_to_attractor'],
        'attractor_id': info['attractor_id'],
        'attractor_state': info['attractor_state']
    }
    for state_id, info in sorted(state_pseudotime.items())
])
pseudotime_df.to_csv(f'{stg_output_dir}/state_pseudotime.csv', index=False)
logging.info(f"Saved pseudotime data to {stg_output_dir}/state_pseudotime.csv")
logging.info(f"Calculated pseudotime for {len(state_pseudotime)} states")

# Now that pseudotime is attached to nodes, save GraphML
nx.write_graphml(G, f'{stg_output_dir}/stg_graph.graphml')
logging.info(f"Saved STG as GraphML (with pseudotime) to {stg_output_dir}/stg_graph.graphml")

# ============================================================================
# STEP 5B: Calculate condition bias and disease-state regions
# ============================================================================
logging.info("\nSTEP 5B: Calculating condition bias and disease-state clustering...")

# Calculate condition bias for each state
# condition_bias = (HIV_cells - Healthy_cells) / (HIV_cells + Healthy_cells)
# Range: -1 (all Healthy) to +1 (all HIV)
condition_bias = {}
all_biases = []

for node in G.nodes():
    hiv = G.nodes[node]['hiv_cells']
    healthy = G.nodes[node]['healthy_cells']
    total = hiv + healthy
    
    if total > 0:
        bias = (hiv - healthy) / total
        condition_bias[node] = bias
        all_biases.append(bias)
    else:
        condition_bias[node] = 0.0  # Default for states with no HIV/Healthy cells

# Calculate standard deviation of condition bias
if all_biases:
    sigma = np.std(all_biases)
else:
    sigma = 0.1  # Default if no biases computed

logging.info(f"Condition bias calculated for {len(condition_bias)} states")
logging.info(f"Condition bias range: [{min(all_biases):.3f}, {max(all_biases):.3f}]")
logging.info(f"Standard deviation (σ): {sigma:.3f}")

# Classify states into disease-state regions
# Healthy: condition_bias < -σ (predominantly Healthy cells)
# Mixed: |condition_bias| ≤ σ (balanced HIV and Healthy)
# HIV: condition_bias > σ (predominantly HIV cells)
for node in G.nodes():
    bias = condition_bias[node]
    
    if bias < -sigma:
        region = 'Healthy'
    elif bias > sigma:
        region = 'HIV'
    else:
        region = 'Mixed'
    
    G.nodes[node]['condition_bias'] = float(bias)
    G.nodes[node]['disease_region'] = region

# Count states by region
region_counts = Counter([G.nodes[n]['disease_region'] for n in G.nodes()])
logging.info(f"Disease-state region distribution:")
for region, count in sorted(region_counts.items()):
    logging.info(f"  {region}: {count} states")

# Save condition bias data
condition_bias_df = pd.DataFrame([
    {
        'state_id': node,
        'state': G.nodes[node]['state'],
        'hiv_cells': G.nodes[node]['hiv_cells'],
        'healthy_cells': G.nodes[node]['healthy_cells'],
        'total_cells': G.nodes[node]['total_cells'],
        'condition_bias': G.nodes[node]['condition_bias'],
        'disease_region': G.nodes[node]['disease_region']
    }
    for node in G.nodes()
])
condition_bias_df.to_csv(f'{stg_output_dir}/condition_bias.csv', index=False)
logging.info(f"Saved condition bias data to {stg_output_dir}/condition_bias.csv")

# ============================================================================
# STEP 5B: Link attractors based on similarity
# ============================================================================
logging.info("\nSTEP 5B: Computing attractor-to-attractor similarity...")

# Define similarity thresholds (as fraction of genes that differ)
STRONG_SIMILARITY_THRESHOLD = 0.02  # <2% different = strong link
WEAK_SIMILARITY_THRESHOLD = 0.05    # <5% different = weak link
# Beyond 5% = no link

# Only compute similarity for top N attractors to keep graph manageable
TOP_N_ATTRACTORS_FOR_SIMILARITY = 100
attractor_list = [state for state, _ in attractor_counts.most_common(TOP_N_ATTRACTORS_FOR_SIMILARITY)]
num_genes = len(attractor_list[0]) if attractor_list else 0

logging.info(f"Computing similarity for top {len(attractor_list)} attractors (by cell count)")

attractor_similarity_edges = []

for i, attractor_a in enumerate(attractor_list):
    for j, attractor_b in enumerate(attractor_list):
        if i >= j:  # Only compute upper triangle (avoid duplicates and self-loops)
            continue
        
        # Calculate Hamming distance (number of differing positions)
        hamming_dist = sum(a != b for a, b in zip(attractor_a, attractor_b))
        hamming_fraction = hamming_dist / num_genes if num_genes > 0 else 1.0
        
        # Determine if we should link them
        if hamming_fraction < STRONG_SIMILARITY_THRESHOLD:
            link_type = 'strong'
            similarity_weight = 1.0 - hamming_fraction  # Close to 1.0
        elif hamming_fraction < WEAK_SIMILARITY_THRESHOLD:
            link_type = 'weak'
            similarity_weight = 1.0 - hamming_fraction  # Between 0.95-0.98
        else:
            continue  # No link for dissimilar attractors
        
        attractor_id_a = state_library[attractor_a]
        attractor_id_b = state_library[attractor_b]
        
        # Add bidirectional edges between attractors (undirected similarity)
        G.add_edge(attractor_id_a, attractor_id_b,
                   edge_type='attractor_similarity',
                   similarity=float(similarity_weight),
                   hamming_distance=int(hamming_dist),
                   link_strength=link_type)
        G.add_edge(attractor_id_b, attractor_id_a,
                   edge_type='attractor_similarity',
                   similarity=float(similarity_weight),
                   hamming_distance=int(hamming_dist),
                   link_strength=link_type)
        
        attractor_similarity_edges.append({
            'attractor_a': attractor_id_a,
            'attractor_b': attractor_id_b,
            'hamming_distance': hamming_dist,
            'similarity': similarity_weight,
            'link_strength': link_type
        })

logging.info(f"Added {len(attractor_similarity_edges)} attractor similarity edges ({len(attractor_similarity_edges)*2} directed edges)")

logging.info(f"Added {len(attractor_similarity_edges)} attractor similarity edges ({len(attractor_similarity_edges)*2} directed edges)")

# Save attractor similarity matrix
if attractor_similarity_edges:
    similarity_df = pd.DataFrame(attractor_similarity_edges)
    similarity_df.to_csv(f'{stg_output_dir}/attractor_similarity.csv', index=False)
    logging.info(f"Saved attractor similarity to {stg_output_dir}/attractor_similarity.csv")

# Re-save GraphML with attractor similarity edges
nx.write_graphml(G, f'{stg_output_dir}/stg_graph_with_similarity.graphml')
logging.info(f"Saved enhanced STG with attractor similarity to {stg_output_dir}/stg_graph_with_similarity.graphml")

# ============================================================================
# STEP 6: Visualize the state transition graph
# ============================================================================
logging.info("\nSTEP 6: Visualizing state transition graph...")

# Filter graph to only include trajectories leading to top 5 attractors
top_5_attractors = [state for state, _ in largest_attractors[:5]]
top_5_attractor_ids = [state_library[state] for state in top_5_attractors]

# Find all nodes that lead to top 5 attractors (backward reachability)
nodes_to_keep = set(top_5_attractor_ids)
for attractor_id in top_5_attractor_ids:
    # Get all predecessors (nodes that can reach this attractor)
    predecessors = nx.ancestors(G, attractor_id)
    nodes_to_keep.update(predecessors)
    nodes_to_keep.add(attractor_id)

# Create subgraph with only relevant nodes
G_filtered = G.subgraph(nodes_to_keep).copy()
logging.info(f"Filtered STG to top 5 attractors: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Filtered STG with layout
ax = axes[0]
pos = nx.spring_layout(G_filtered, k=2, iterations=50, seed=42)

# Draw nodes
node_sizes = [G_filtered.nodes[node]['attractor_size'] * 50 + 100 for node in G_filtered.nodes()]
node_colors = ['#ff7f0e' if G_filtered.nodes[node]['is_attractor'] else '#1f77b4' for node in G_filtered.nodes()]

nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color=node_colors, 
                       alpha=0.7, ax=ax)

# Draw edges with varying widths based on transition count
# Only draw trajectory edges (those with 'weight'), not similarity edges
trajectory_edges = [(u, v) for u, v in G_filtered.edges() if 'weight' in G_filtered[u][v]]
if trajectory_edges:
    max_weight = max([G_filtered[u][v]['weight'] for u, v in trajectory_edges])
    edge_widths = [G_filtered[u][v]['weight'] / max_weight * 3 for u, v in trajectory_edges]
    nx.draw_networkx_edges(G_filtered, pos, edgelist=trajectory_edges, width=edge_widths, alpha=0.5, 
                           edge_color='gray', ax=ax, arrowsize=15)

# Draw labels for attractor nodes
attractor_labels = {node: str(node) for node in G_filtered.nodes() 
                    if G_filtered.nodes[node]['is_attractor']}
nx.draw_networkx_labels(G_filtered, pos, labels=attractor_labels, font_size=8, ax=ax)

ax.set_title('State Transition Graph - Top 5 Attractors (Orange = Attractors, Blue = Transient States)', fontsize=12)
ax.axis('off')

# Subplot 2: Attractor distribution (top 5 only)
ax = axes[1]
attractor_names = [f"Attr {state_library[state]}" for state, _ in largest_attractors[:7]]
attractor_values = [count for _, count in largest_attractors[:7]]
colors_bar = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']

ax.bar(range(len(attractor_names)), attractor_values, color=colors_bar, alpha=0.7)
ax.set_xticks(range(len(attractor_names)))
ax.set_xticklabels(attractor_names, rotation=45, ha='right')
ax.set_ylabel('Number of Cells')
ax.set_xlabel('Attractors')
ax.set_title('Cell Distribution Across Top 5 Attractors', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{stg_output_dir}/stg_visualization.png', dpi=300, bbox_inches='tight')
logging.info(f"Saved STG visualization to {stg_output_dir}/stg_visualization.png")
plt.close()

# ============================================================================
# STEP 6B: Visualize disease-state regions
# ============================================================================
logging.info("\nSTEP 6B: Visualizing disease-state regions...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Condition bias distribution
ax = axes[0]
all_biases_list = list(condition_bias.values())
ax.hist(all_biases_list, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(x=-sigma, color='green', linestyle='--', linewidth=2, label=f'Healthy threshold (-σ = {-sigma:.3f})')
ax.axvline(x=sigma, color='red', linestyle='--', linewidth=2, label=f'HIV threshold (+σ = {sigma:.3f})')
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='Mixed center (0)')
ax.set_xlabel('Condition Bias')
ax.set_ylabel('Number of States')
ax.set_title('Distribution of Condition Bias Across All States')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Subplot 2: States by disease region
ax = axes[1]
region_colors = {'Healthy': '#1f77b4', 'Mixed': '#ff7f0e', 'HIV': '#d62728'}
regions = list(region_counts.keys())
region_vals = [region_counts[r] for r in regions]
region_cols = [region_colors.get(r, 'gray') for r in regions]
ax.bar(regions, region_vals, color=region_cols, alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of States')
ax.set_xlabel('Disease Region')
ax.set_title('State Distribution by Disease-State Region')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{stg_output_dir}/disease_state_regions.png', dpi=300, bbox_inches='tight')
logging.info(f"Saved disease-state region visualization to {stg_output_dir}/disease_state_regions.png")
plt.close()

# ============================================================================
# STEP 7: Generate STG summary statistics
# ============================================================================
logging.info("\nSTEP 7: Generating summary statistics...")

summary_stats = {
    'total_cells': len(cell_trajectories_mapped),
    'total_trajectory_steps': sum(len(states) for states in cell_trajectories_mapped.values()),
    'unique_states': len(state_library),
    'total_attractors': len(attractor_counts),
    'graph_nodes': G.number_of_nodes(),
    'graph_edges': G.number_of_edges(),
    'largest_attractors': str([(state_library[state], count) for state, count in largest_attractors[:5]])
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(f'{stg_output_dir}/stg_summary.csv', index=False)

logging.info("\nState Transition Graph Summary:")
for key, value in summary_stats.items():
    logging.info(f"  {key}: {value}")

# ============================================================================
# STEP 8: Attractor-based clustering with pseudotime-scaled trajectories
# ============================================================================
logging.info("\nSTEP 8: Creating attractor-based clustering visualization...")

# Use ALL cells and ALL attractors (not filtered)
logging.info(f"Using all {len(cell_attractor_mapping)} cells reaching all {len(attractor_counts)} attractors")

# Collect all unique states from all trajectories
all_trajectory_states = set()
cell_trajectories_to_plot = {}

for cell_num, trajectory_states in cell_trajectories_mapped.items():
    cell_trajectories_to_plot[cell_num] = trajectory_states
    
    # Add all states in this trajectory
    for state in trajectory_states:
        all_trajectory_states.add(state)

# Convert to list for indexing
unique_states_list = list(all_trajectory_states)
state_to_index = {state: i for i, state in enumerate(unique_states_list)}

logging.info(f"Collected {len(unique_states_list)} unique states from {len(cell_trajectories_to_plot)} cell trajectories")

# Prepare state array for t-SNE
states_array = np.array([np.array(state) for state in unique_states_list])

# Run t-SNE on all states
logging.info("Running t-SNE on all trajectory states...")
tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(states_array) - 1))
state_coords = tsne.fit_transform(states_array)

# Map each state to its t-SNE coordinate
state_tsne_coords = {
    state: state_coords[state_to_index[state]]
    for state in unique_states_list
}

# Create 3D visualization
logging.info("Creating interactive 3D visualization with Plotly...")

# Collect all trajectory states with their pseudotimes
all_coords = []
all_pseudotimes = []

logging.info("Collecting all trajectory states...")
trajectory_point_count = 0
for cell_num, trajectory_states in cell_trajectories_to_plot.items():
    for state in trajectory_states:
        state_id = state_library[state]
        if state not in state_tsne_coords:
            logging.warning(f"State {state} not in t-SNE coordinates")
            continue
            
        coord = state_tsne_coords[state]
        
        # Get pseudotime for this state
        if state_id in state_pseudotime:
            pseudotime = state_pseudotime[state_id]['pseudotime']
        else:
            pseudotime = 0.0
        
        all_coords.append(coord)
        all_pseudotimes.append(pseudotime)
        trajectory_point_count += 1

all_coords = np.array(all_coords)
all_pseudotimes = np.array(all_pseudotimes)

logging.info(f"Collected {trajectory_point_count} trajectory points from {len(cell_trajectories_to_plot)} cells")
logging.info(f"Plotting {len(all_coords)} trajectory points...")

# Create Plotly figure
fig = go.Figure()

# Add trajectory points colored by pseudotime
fig.add_trace(go.Scatter3d(
    x=all_coords[:, 0],
    y=all_coords[:, 1],
    z=all_coords[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=all_pseudotimes,
        colorscale='RdBu_r',  # Red-Blue reversed (red for high pseudotime)
        showscale=True,
        cmin=0,
        cmax=1,
        colorbar=dict(
            title="Pseudotime<br>(0=far, 1=near)",
            thickness=15,
            len=0.7
        ),
        opacity=0.3,
        line=dict(width=0)
    ),
    hoverinfo='skip',
    name='Trajectory States'
))

# Plot only attractors that are actually reached by the plotted cells
logging.info("Adding attractors...")
attractor_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']

# Get unique attractors actually reached by cells in plot
attractors_in_plot = set()
for cell_num in cell_trajectories_to_plot.keys():
    if cell_num in cell_attractor_mapping:
        attractors_in_plot.add(cell_attractor_mapping[cell_num])

logging.info(f"Found {len(attractors_in_plot)} unique attractors in plotted trajectories")

dark_tsne_theme = dict(
    paper_bgcolor="rgba(26,29,41,1.0)",   # outer background
    plot_bgcolor="rgba(26,29,41,1.0)",
    font=dict(
        color="#C9D1D9",
        size=14
    ),
    title=dict(
        font=dict(
            color="#E6EDF3",
            size=18
        )
    ),
    scene=dict(
        bgcolor="rgba(30,34,48,1.0)",
        xaxis=dict(
            title="t-SNE 1",
            showbackground=True,
            backgroundcolor="rgba(14,17,23,1.0)",
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#C9D1D9"),
            showgrid=False,
        ),
        yaxis=dict(
            title="t-SNE 2",
            showbackground=True,
            backgroundcolor="rgba(14,17,23,1.0)",
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#C9D1D9"),
            showgrid=False,
        ),
        zaxis=dict(
            title="t-SNE 3",
            showbackground=True,
            backgroundcolor="rgba(14,17,23,1.0)",
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#C9D1D9"),
            showgrid=False,
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        ),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C9D1D9")
    )
)


# Update layout for better interactivity
fig.update_layout(**dark_tsne_theme)


# Save as interactive HTML
html_path = f'{stg_output_dir}/attractor_clustering_pseudotime_interactive.html'
fig.write_html(html_path)
logging.info(f"Saved interactive 3D visualization to {html_path}")

# Note: Static PNG export is disabled due to Kaleido issues with large 3D plots
# The interactive HTML is the primary visualization

# Save trajectory statistics
trajectory_stats = []
for cell_num, trajectory_states in cell_trajectories_to_plot.items():
    attractor = cell_attractor_mapping[cell_num]
    attractor_id = state_library[attractor]
    
    # Get starting state info
    starting_state = trajectory_states[0]
    starting_state_id = state_library[starting_state]
    starting_pseudotime = state_pseudotime.get(starting_state_id, {}).get('pseudotime', 0.0)
    starting_coord = state_tsne_coords[starting_state]
    
    # Get attractor info
    attractor_coord = state_tsne_coords[attractor]
    
    trajectory_stats.append({
        'cell_number': cell_num,
        'attractor_id': attractor_id,
        'trajectory_length': len(trajectory_states),
        'starting_state_pseudotime': starting_pseudotime,
        'starting_tsne_x': starting_coord[0],
        'starting_tsne_y': starting_coord[1],
        'starting_tsne_z': starting_coord[2],
        'attractor_tsne_x': attractor_coord[0],
        'attractor_tsne_y': attractor_coord[1],
        'attractor_tsne_z': attractor_coord[2]
    })

traj_stats_df = pd.DataFrame(trajectory_stats)
traj_stats_df.to_csv(f'{stg_output_dir}/trajectory_length_stats.csv', index=False)
logging.info(f"Saved trajectory statistics to {stg_output_dir}/trajectory_length_stats.csv")

logging.info(f"\nAll output files saved to: {stg_output_dir}")
logging.info("Files generated:")
logging.info("  - state_library.csv: Complete library of all unique states")
logging.info("  - cell_attractor_mapping.csv: Which attractor each cell reaches")
logging.info("  - state_pseudotime.csv: Pseudotime values for all states")
logging.info("  - stg_graph.json: STG in JSON format")
logging.info("  - stg_graph.graphml: STG in GraphML format (for Cytoscape)")
logging.info("  - stg_visualization.png: Visual representation of STG")
logging.info("  - attractor_clustering_pseudotime_interactive.html: Interactive 3D visualization (MAIN OUTPUT - OPEN IN BROWSER)")
logging.info("  - trajectory_length_stats.csv: Cell trajectory length statistics")
logging.info("  - stg_summary.csv: Summary statistics")