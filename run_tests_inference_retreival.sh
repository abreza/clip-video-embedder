#!/bin/bash

# Define the list of k values to test
k_values=(1)

# Define the list of modes to test
modes=('top') 
# 'top' 'random' 'uniform' 'least')

# Define the output file
output_file="results_infernece_retrieval_captions2.txt"

# Loop over each mode and k value
for mode in "${modes[@]}"
do  
    echo '-------------------------------------------------------------------------------------' >> "$output_file"
    echo "Mode: $mode" >> "$output_file"
    echo '-------------------------------------------------------------------------------------' >> "$output_file"
    
    for k in "${k_values[@]}"
    do
        # Run the script with the specified mode and k value
        output=$(python inference_retrieval.py --mode "$mode" -k "$k")

        # Print a message to indicate progress
        echo "Completed mode=${mode}, k=${k}"

        # Write the output to the output file
        # echo "k: $k" >> "$output_file"
        echo "$output" >> "$output_file"
        echo "" >> "$output_file"
    done
done