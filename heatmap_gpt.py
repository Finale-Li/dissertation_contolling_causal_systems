import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Use the 'ggplot' style
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times}')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'


def read_all_json_files(directory_path):
    all_data = {}  # Initialize an empty dictionary to store the JSON data.

    # Iterate through every file in the specified directory.
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):  # Check if the file is a JSON file.
            file_path = os.path.join(directory_path, filename)  # Get the full path of the file.

            # Open and read the JSON file.
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # Load the file content as JSON.

                # Store the data in the dictionary with the filename as the key.
                all_data[filename] = data

    return all_data


def extract_data(all_data):
    data = {i: {"in": None, "out": None} for i in range(len(all_data))}
    for idx, ppl in enumerate(all_data.values()):
        data[idx]["in"] = [0] + ppl["gpt_decision_history"]
        data[idx]["out"] = [0] + ppl["production_history"]
        success = ppl["success"]
        print(f"{idx}:{success}")
    return data


if __name__ == '__main__':
    name = "gpt3_5"
    name = "gpt4"
    color = {"gpt4": "Blues", "gpt3_5": "Greens"}
    gpt_data = read_all_json_files(name)
    #
    # # Call the function and store the result.
    data_extracted = extract_data(gpt_data)
    flatted_out = list(itertools.chain.from_iterable([value["out"][:-1] for value in data_extracted.values()]))
    flatted_in = list(itertools.chain.from_iterable([value["in"][1:] for value in data_extracted.values()]))
    data = np.stack([np.array(flatted_out), np.array(flatted_in)], axis=-1)
    # heatmap, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=(12, 12))

    heatmap = np.zeros([12, 12])
    for output in range(1, 13):
        for input in range(1, 13):
            heatmap[output - 1, input - 1] = sum((data[..., 0] == output * 1000) * (data[..., 1] == input * 100))

    # Create the plot
    fig, ax = plt.subplots(dpi=150)
    cax = ax.imshow(heatmap.T, origin='lower', cmap=color[name], interpolation='none', aspect='equal', vmax=70, vmin=0)

    # Set the grid

    ax.set_xticks(np.arange(0, 12, 1))  # Setting minor ticks
    ax.set_yticks(np.arange(0, 12, 1))  # Setting minor ticks
    x_tick_labels = [f"{1000 * i}" if i % 2 == 0 else '' for i in range(1, 13)]
    y_tick_labels = [f"{100 * i}" if i % 2 == 0 else '' for i in range(1, 13)]
    ax.set_xticklabels(x_tick_labels, fontdict={"size": 12})
    ax.set_yticklabels(y_tick_labels, fontdict={"size": 12})
    ax.grid(True)  # Configuring the grid

    # Add a color bar
    colorbar = plt.colorbar(cax, label='Frequency')
    plt.xlabel('Previous Trial Production ($p_{t-1}$)', fontdict={"size": 12})
    plt.ylabel('Workforce Input ($w_t$)', fontdict={"size": 12})
    plt.savefig(f"heatmap_{name}.png", dpi=200)
