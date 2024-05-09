import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap

# Use the 'ggplot' style
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times}')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rc('text.latex', preamble=r'\usepackage{bm}')
import pandas as pd


def extract_participants(xls, num_people=49):
    human_data = {i: {"in": [], "out": []} for i in range(num_people)}
    for day in range(30 + 1):
        day_in = xls[f"trial{day}_in"]
        day_out = xls[f"trial{day}_out"]
        for person in range(num_people):
            human_data[person]["in"].append(day_in[person])
            human_data[person]["out"].append(day_out[person])
    print(xls["total_reward"])
    return human_data


if __name__ == '__main__':
    human_data_xls = pd.read_excel("human_data_without_repitition.xlsx")
    prefix_for_image = "human"
    #
    # # Call the function and store the result.
    data_extracted = extract_participants(human_data_xls)
    flatted_out = list(itertools.chain.from_iterable([value["out"][:-1] for value in data_extracted.values()]))
    flatted_in = list(itertools.chain.from_iterable([value["in"][1:] for value in data_extracted.values()]))
    data = np.stack([np.array(flatted_out), np.array(flatted_in)], axis=-1)

    heatmap = np.zeros([12, 12])
    for output in range(1, 13):
        for input in range(1, 13):
            heatmap[output - 1, input - 1] = sum((data[..., 0] == output * 1000) * (data[..., 1] == input * 100))

    # Create the plot
    fig, ax = plt.subplots(dpi=150)
    cax = ax.imshow(heatmap.T, origin='lower', cmap='Reds', interpolation='none', aspect='equal', vmax=70)

    # Set the grid

    ax.set_xticks(np.arange(0, 12, 1))  # Setting minor ticks
    ax.set_yticks(np.arange(0, 12, 1))  # Setting minor ticks
    x_tick_labels = [f"{1000 * i}" if i % 2 == 0 else '' for i in range(1, 13)]
    y_tick_labels = [f"{100 * i}" if i % 2 == 0 else '' for i in range(1, 13)]
    ax.set_xticklabels(x_tick_labels, fontdict={"size": 12, 'weight': 'bold',})
    ax.set_yticklabels(y_tick_labels, fontdict={"size": 12, 'weight': 'bold',})
    ax.grid(True)  # Configuring the grid

    # Add a color bar
    plt.colorbar(cax, label='Frequency')
    plt.xlabel('Previous Trial Production ($p_{t-1}$)', fontdict={"size": 12, 'weight': 'bold',})
    plt.ylabel('Workforce Input ($w_t$)', fontdict={"size": 12, 'weight': 'bold',})
    plt.savefig("heatmap_human.png", dpi=200)