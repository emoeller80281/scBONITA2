import logging
import os
import pandas as pd
from user_input_prompts import *

def write_group(dataset_file, col_indices, output_path):
    logging.info(f'\t\tWriting data to the group file...')

    chunk_size = 1000
    # Initialize a variable to track the first chunk
    first_chunk = True

    full_col_indices = [0] + col_indices

    # Use iterator=True to get a TextFileReader object for iteration
    reader = pd.read_csv(dataset_file, usecols=full_col_indices, chunksize=chunk_size, iterator=True)
    for chunk_num, df in enumerate(reader):
        # Rename the first column if needed
        if df.columns[0] == 'Unnamed: 0':
            df.rename(columns={df.columns[0]: ''}, inplace=True)

        # Write the processed chunk to a new CSV file
        # If it's the first chunk, write the header, otherwise append without header
        df.to_csv(output_path, mode='a', index=False, header=first_chunk)

        # After the first chunk is processed, set first_chunk to False
        if first_chunk:
            first_chunk = False

def metadata_parser(metadata_file, metadata_sep, dataset_file, dataset_sep, cell_name_index, group_indices, header, overwrite):
    # If no arguments, prompt the user for the dataset and network name
    with open(metadata_file, 'r') as metadata_file:
        if header == 'y':
            next(metadata_file)

        logging.info(f'\n----- Splitting Data Files -----')

        groups = {}
        
        combinations = []
        # Read the metadata file to assign the cells to their groups
        line_number = 1 if header == 'y' else 0
        for line in metadata_file:
            line_number += 1
            line = line.replace('"', '')
            line = line.strip()
            
            # Skip empty lines
            if not line:
                logging.debug(f'\t\tSkipping empty line {line_number}')
                continue
            
            split_line = line.split(metadata_sep)
            
            try:
                cell_name = split_line[cell_name_index].strip()

            except IndexError:
                logging.error(f'\n\t!!!ERROR on line {line_number}: Cannot access cell_name_index={cell_name_index}')
                logging.error(f'\t   Line has {len(split_line)} columns but trying to access column {cell_name_index}')
                logging.error(f'\t   Line content (first 200 chars): {line[:200]}')
                logging.error('\t   Check separator and cell_name_index settings\n')
                continue

            try:
                group_combination = tuple(split_line[group_index].strip() for group_index in group_indices)
            except IndexError:
                logging.error(f'\n\t!!!ERROR on line {line_number}: Cannot access group_indices={group_indices}')
                logging.error(f'\t   Line has {len(split_line)} columns but trying to access columns {group_indices}')
                logging.error(f'\t   Cell name: {cell_name if "cell_name" in locals() else "N/A"}')
                logging.error(f'\t   Line content (first 200 chars): {line[:200]}')
                logging.error('\t   Skipping this line...\n')
                continue

            if cell_name not in groups:
                groups[cell_name] = []
            groups[cell_name] = group_combination

            if group_combination not in combinations:
                combinations.append(group_combination)
        
        if len(groups) == 0:
            logging.error('\n\t!!!ERROR: No valid cells were found in metadata file!')
            logging.error('\t   Check cell_name_index, group_indices, separator, and file format\n')
            exit(1)

        for group_num, group in enumerate(combinations):
            logging.info(f'\tGroup {group_num + 1}: {", ".join(group)}')

        cell_groups = {}
        
        # Split the data file based on the groups
        with open(dataset_file) as datafile:        
            line_count = 1
            for line in datafile:
                line = [cell.strip() for cell in line.replace('"', '').strip().split(dataset_sep)]
                line = line[1:] # Skip the first column, not a cell
                
                # Append the column indices for each group to a dictionary
                if line_count == 1:
                    cells_not_in_metadata = 0
                    for cell_index, cell in enumerate(line):
                        try:
                            if len(groups[cell]) > 1:
                                group_name = '_'.join(name for name in groups[cell])
                            else:
                                group_name = groups[cell][0]
                            if not group_name in cell_groups:
                                cell_groups[group_name] = [cell_index]
                            else:
                                cell_groups[group_name].append(cell_index)

                        except KeyError as e:
                            cells_not_in_metadata += 1
                            if cells_not_in_metadata <= 5:  # Only show first 5 examples
                                logging.debug(f'\t\tCell {e} in data file not found in metadata, skipping')
                            continue
                    
                    if cells_not_in_metadata > 0:
                        logging.info(f'\t{cells_not_in_metadata} cells from data file not found in metadata (skipped)')
                
                line_count += 1

        # Read in the dataset as a pandas dataframe for each group with only the group columns and write the files out as a csv
        dataset_paths = []
        dataset_groups = []

        logging.info(f'\n----- Saving Group Data Files -----')
        for groups, col_indices in cell_groups.items():
            # Format the output path to include the groups
            formatted_dataset_file = dataset_file.replace('.csv', '')
            group_name = groups
            output_path = f'{formatted_dataset_file}_{group_name}.csv'
            dataset_paths.append(output_path)
            dataset_groups.append(groups)
            if not os.path.exists(output_path):
                write_group(dataset_file, col_indices, output_path)
            else:
                overwrite = overwrite_check(overwrite, output_path)
                if overwrite == True:
                    logging.info(f'\tOverwriting...')
                    write_group(dataset_file, col_indices, output_path)
                else:
                    logging.info(f'\tUsing existing file')
    
    return dataset_paths, dataset_groups, cell_groups
    
