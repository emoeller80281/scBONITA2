import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file paths relative to the script directory
file_paths = {
    'pickle_files' : os.path.join(script_dir, '../scBONITA_output/pickle_files'),
    'graphml_files' : os.path.join(script_dir, '../scBONITA_output/graphml_files'),
    'rules_output' : os.path.join(script_dir, '../scBONITA_output/rules_output'),
    'importance_score_output' : os.path.join(script_dir, '../scBONITA_output/importance_score_output'),
    'relative_abundance_output' : os.path.join(script_dir, '../scBONITA_output/relative_abundance_output'),
    'pathway_xml_files' : os.path.join(script_dir, '../scBONITA_output/pathway_xml_files'),
    'custom_graphml' : os.path.join(script_dir, '../input/custom_graphml_files'),
    'trajectories' : os.path.join(script_dir, '../scBONITA_output/trajectories'),
    'trajectory_analysis': os.path.join(script_dir, '../scBONITA_output/trajectory_analysis'),
    'metadata' : os.path.join(script_dir, '../input')
}
