from .model_utils import *
from .utils import *

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas

"""Credit: Naysan Saran Modification: [M Shaf Khattak].(https://github.com/SHaf373)

License

This project is licensed under the GPL V3 licence."""

class Node():

    def __init__(
        self, center, radius, label,
        facecolor='#2653de', edgecolor='#e6e6e6',
        ring_facecolor='#a3a3a3', ring_edgecolor='#a3a3a3',
        **kwargs
        ):
        """
        Initializes a Markov Chain Node(for drawing purposes)
        Inputs:
            - center : Node (x,y) center
            - radius : Node radius
            - label  : Node label
        """
        self.center = center
        self.radius = radius
        self.label  = label

        # For convinience: x, y coordinates of the center
        self.x = center[0]
        self.y = center[1]

        # Drawing config
        self.node_facecolor = facecolor
        self.node_edgecolor = edgecolor

        self.ring_facecolor = ring_facecolor
        self.ring_edgecolor = ring_edgecolor
        self.ring_width = 0.03

        self.text_args = {
            'ha': 'center',
            'va': 'center',
            'fontsize': kwargs.get("node_fontsize", 12)
        }


    def add_circle(self, ax):
        """
        Add the annotated circle for the node
        """
        circle = mpatches.Circle(self.center, self.radius)
        p = PatchCollection(
            [circle],
            edgecolor = self.node_edgecolor,
            facecolor = self.node_facecolor
        )
        ax.add_collection(p)
        ax.annotate(
            self.label,
            xy = self.center,
            color = '#ffffff',
            **self.text_args
        )


    def add_self_loop(self, ax, prob=None, direction='up', annotate = True, percentages = False):
        """
        Draws a self loop
        """
        if direction == 'up':
            start = -30
            angle = 180
            ring_x = self.x
            ring_y = self.y + self.radius
            prob_y = self.y + 1.3*self.radius
            x_cent = ring_x - self.radius + (self.ring_width/2)
            y_cent = ring_y - 0.15
        else:
            start = -210
            angle = 0
            ring_x = self.x
            ring_y = self.y - self.radius
            prob_y = self.y - 1.4*self.radius
            x_cent = ring_x + self.radius - (self.ring_width/2)
            y_cent = ring_y + 0.15

        # Add the ring
        ring = mpatches.Wedge(
            (ring_x, ring_y),
            self.radius,
            start,
            angle,
            width = self.ring_width
        )
        # Add the triangle (arrow)
        offset = 0.2
        left   = [x_cent - offset, ring_y]
        right  = [x_cent + offset, ring_y]
        bottom = [(left[0]+right[0])/2., y_cent]
        arrow  = plt.Polygon([left, right, bottom, left])

        p = PatchCollection(
            [ring, arrow],
            edgecolor = self.ring_edgecolor,
            facecolor = self.ring_facecolor
        )
        ax.add_collection(p)

        # Probability to add?
        if prob and annotate:
            text = f"{prob*100 if percentages else prob:.1f}".rstrip("0").rstrip(".")
            text += "%" if percentages else ""
            ax.annotate(text, xy=(self.x, prob_y), color='#000000', **self.text_args)

"""Credit: Naysan Saran Modification: [M Shaf Khattak].(https://github.com/SHaf373)

License

This project is licensed under the GPL V3 licence."""

class MarkovChain:

    def __init__(self, M, labels, **kwargs):
        """
        Initializes a Markov Chain (for drawing purposes)
        Inputs:
            - M         Transition Matrix
            - labels    State Labels
            - kwargs    Keywords to modify how data is displayed, specifically:
                        annotate          if False, probabilities aren't shown
                        arrow_edgecolor
                        arrow_facecolor
                        arrow_head_width
                        arrow_width
                        fontsize          affects transition probability labels
                        node_edgecolor
                        node_facecolor
                        node_fontsize     affects node labels
                        node_radius
                        percentages       bool, if True probabilites should be
                                          displayed as percentages instead of decimals
                        transparency_func function to determine transparency of arrows (default: alpha = prob)
        """

        np.set_printoptions(precision=3, suppress=True)

        if M.shape[0] < 2:
            raise Exception("There should be at least 2 states")
        if M.shape[0] != M.shape[1]:
            raise Exception("Transition matrix should be square")
        if M.shape[0] != len(labels):
            raise Exception("There should be as many labels as states")

        # save args
        self.M = M
        self.n_states = M.shape[0]
        self.labels = labels

        self.save_kwargs(**kwargs)

        # Build the network
        self.build_network()

    def save_kwargs(self, **kwargs):

        # save the dictionary
        self.kwargs = kwargs

        # Colors
        self.arrow_facecolor = self.kwargs.get("arrow_facecolor", '#a3a3a3')
        self.arrow_edgecolor = self.kwargs.get("arrow_edgecolor", '#a3a3a3')
        self.node_facecolor = self.kwargs.get("node_facecolor", '#2693de')
        self.node_edgecolor = self.kwargs.get("node_edgecolor", '#e6e6e6')

        # Drawing config
        self.node_radius = self.kwargs.get("node_radius", 0.60)
        self.arrow_width = self.kwargs.get("arrow_width", 0.1)
        self.arrow_head_width = self.kwargs.get("arrow_head_width", 0.22)
        self.text_args = {
            'ha': 'center',
            'va': 'center',
            'fontsize': self.kwargs.get("fontsize", 14)
        }
        self.scale_xlim = self.kwargs.get("scale_xlim", 1)
        self.scale_ylim = self.kwargs.get("scale_ylim", 1)

        # How to represent the probabilities
        self.percentages = self.kwargs.get("percentages", False)
        self.annotate_probabilities = self.kwargs.get("annotate", True)
        self.transparency_func = self.kwargs.get(
            "transparency_func", lambda p: p)

        # Additional config options
        self.self_arrows = self.kwargs.get("self_arrows", True)
        self.n_columns = self.kwargs.get('n_columns')

    def set_node_centers(self):
        """
            Spread the nodes evenly around in a circle using Euler's formula
            e^(2pi*i*k/n), where n is the number of nodes and k is the
            index over which we iterate. The real part is the x coordinate,
            the imaginary part is the y coordinate. Then scale by n for more room.

            self.node_centers is a numpy array of shape (n,2)
        """

        # For legibility, we use n below
        n = self.n_states

        # generate the evenly spaced coords on the unit circle
        unit_circle_coords = np.fromfunction(lambda x, y:
                                             (1-y)*np.real(np.exp(2 * np.pi * x/n * 1j))
                                             + y *
                                             np.imag(
                                                 np.exp(2 * np.pi * x/n * 1j)),
                                             (n, 2))

        self.figsize = (n*2+2, n*2+2)
        self.xlim = (-n-1, n+1)
        self.ylim = (-n-1, n+1)
        self.xlim = tuple([self.scale_xlim * l for l in self.xlim])
        self.ylim = tuple([self.scale_ylim * l for l in self.ylim])

        # Scale by n to have more room
        self.node_centers = unit_circle_coords * n

        # For legibility, we use n below
        n = self.n_states

        # generate the evenly spaced coords on the unit grid with n_columns
        n_columns = self.n_columns
        n_rows = n // n_columns
        grid_coords = np.vstack([[x - n_columns/4, n_rows - 1 - y]
                                 for x in range(n_columns)
                                 for y in range(n_rows)])

        self.figsize = (n*2+4, n*2+4)
        self.xlim = (-1*(n+2), n+2)
        self.ylim = (-2, n*2+2)
        self.xlim = tuple([self.scale_xlim * l for l in self.xlim])
        self.ylim = tuple([self.scale_ylim * l for l in self.ylim])

        # Scale by n to have more room
        self.node_centers = grid_coords * n

    def build_network(self):
        """
        Loops through the matrix, add the nodes
        """
        # Position the node centers
        self.set_node_centers()

        # Set the nodes
        self.nodes = [Node(self.node_centers[i],
                           self.node_radius,
                           self.labels[i],
                           **{**self.kwargs,
                              "facecolor": (self.node_facecolor
                                            if isinstance(self.node_facecolor,
                                                          str)
                                            else self.node_facecolor[i])
                              }
                           )
                      for i in range(self.n_states)]

    def add_arrow(self, ax,
                  node1, node2,
                  prob=None, width=None,
                  head_width=None,
                  annotate=True,
                  arrow_spacing=0.15,
                  transparency_func=None):
        """
        Add a directed arrow between two nodes

        Keywords:

        annotate:                if True, probability is displayed on top of the arrow
        arrow_spacing:           determines space between arrows in opposite directions
        head_width:              width of arrow head
        prob:                    probability of going from node1 to node2
        transparency_func:       function to determine transparency of arrows
        width:                   width of arrow body
        """

        if width is None:
            width = self.arrow_width
        if head_width is None:
            head_width = self.arrow_head_width
        if transparency_func is None:
            transparency_func = self.transparency_func

        # x,y start of the arrow, just touching the starting node
        x_start = node1.x + node1.radius * \
            (node2.x-node1.x)/np.linalg.norm(node2.center-node1.center)
        y_start = node1.y + node1.radius * \
            (node2.y-node1.y)/np.linalg.norm(node2.center-node1.center)

        # find the arrow length so it just touches the ending node
        dx = node2.x-x_start - node2.radius * \
            (node2.x-node1.x)/np.linalg.norm(node2.center-node1.center)
        dy = node2.y-y_start - node2.radius * \
            (node2.y-node1.y)/np.linalg.norm(node2.center-node1.center)

        # calculate offset so arrows in opposite directions are separate

        x_offset = dy / np.sqrt(dx**2+dy**2) * arrow_spacing
        y_offset = -dx / np.sqrt(dx**2+dy**2) * arrow_spacing

        arrow = mpatches.FancyArrow(
            x_start + x_offset,
            y_start + y_offset,
            dx,
            dy,
            width=width,
            head_width=head_width,
            length_includes_head=True
        )

        # Check if arrow overlaps any other nodes on the way
        # Use curved arrow instead
        if ((node1.x == node2.x) and
            np.any([((center[0] == node1.x)
                     and (((center[1] > node1.y)
                           and (center[1] < node2.y))
                          or ((center[1] < node1.y)
                              and (center[1] > node2.y))))
                    for center in self.node_centers])):
            style=mpatches.ArrowStyle('simple',
                                      head_length=2*head_width,
                                      head_width=head_width,
                                      tail_width=width)
            arrow = mpatches.FancyArrowPatch(
                
                posA=(x_start + x_offset, y_start + y_offset),
                posB=(x_start + x_offset + dx, y_start + y_offset + dy),
                shrinkA=0, shrinkB=0,
                arrowstyle=style,
                mutation_scale=1,
                connectionstyle="arc3, rad=0.4")
            arrow.set_linewidth(0.2)

        
        p = PatchCollection(
            [arrow],
            edgecolor=self.arrow_edgecolor,
            facecolor=self.arrow_facecolor,
            alpha=transparency_func(prob)
        )
        ax.add_collection(p)

        # Add label of probability at coordinates (x_prob, y_prob)
        x_prob = x_start + 0.2*dx + 1.2 * x_offset
        y_prob = y_start + 0.2*dy + 1.2 * y_offset
        if prob and annotate:
            text = f"{prob*100 if self.percentages else prob:.1f}".rstrip(
                "0").rstrip(".")
            text += "%" if self.percentages else ""
            ax.annotate(text, xy=(x_prob, y_prob),
                        color='#000000', **self.text_args)

    def draw(self, img_path=None):
        """
        Draw the Markov Chain
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Set the axis limits
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

        # Draw the nodes
        for node in self.nodes:
            node.add_circle(ax)

        # Add the transitions
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                # self loops
                if self.self_arrows and i == j and self.M[i, i] > 0:
                    self.nodes[i].add_self_loop(ax,
                                                prob=self.M[i, j],
                                                direction='up' if self.nodes[i].y >= 0 else 'down',
                                                annotate=self.annotate_probabilities,
                                                percentages=self.percentages)

                # directed arrows
                elif self.M[i, j] > 0 and i!=j:
                    self.add_arrow(ax,
                                   self.nodes[i],
                                   self.nodes[j],
                                   prob=self.M[i, j],
                                   annotate=self.annotate_probabilities)

        plt.axis('off')
        # Save the image to disk?
        if img_path:
            plt.savefig(img_path)
        plt.show()

def plot_strategy_distribution(data, # The dataset containing data on parameters and the strategy distribution
                               strategy_set, # The strategies to plot from the dataset
                               x="pr", # The parameter to place on the x-axis of the plot
                               x_label='Risk of an AI disaster, pr', # the x-axis label
                               title='Strategy distribution', # the plot title
                               thresholds=["threshold_society_prefers_safety",
                                           "threshold_risk_dominant_safety"], # A list of threshold names in data
                               cmap=plt.colormaps["tab10"],
                               ) -> None:
    """Plot the strategy distribution as we vary `x`."""

    fig, ax = plt.subplots()
    ax.stackplot(data[x],
                 [data[strategy + "_frequency"] for strategy in strategy_set],
                 labels=strategy_set,
                 colors=[cmap(i) for i in range(cmap.N)],
                 alpha=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Proportion')

    # Add threshold boundaries to convey dilemma region
    if thresholds!=None:        
        plt.vlines([data[name].values[0] for name in thresholds],
                    0,
                    0.995,
                    colors=[f"C{2+i}" for i in range(len(thresholds))],
                    linewidth=3)

def plot_heatmap(table, # A pivot table, created using `pandas.pivot` function
                 figure_object=None,
                 xlabel="x",
                 ylabel="y",
                 zlabel="z",
                 cmap='inferno',
                 zmin=0,
                 zmax=1,
                 zcenter=None,
                 norm=None,
                 interpolation=None,
                 set_colorbar=True,
                 set_labels=True,
                ):
    """Plot heatmap using the index, columns, and values from `table`."""
    if figure_object==None:
        heatmap, ax = plt.subplots()
    else:
        heatmap, ax = figure_object
    im = ax.imshow(table.values,
                   cmap=cmap,
                   norm=norm,
                   extent=[table.columns.min(),
                           table.columns.max(),
                           table.index.min(),
                           table.index.max()],
                   vmin=zmin,
                   vmax=zmax,
                   interpolation=interpolation,
                   origin='lower',
                   aspect='auto')
    if set_labels:
        ax.set(xlabel=xlabel,
            ylabel=ylabel)
    if set_colorbar:
        cbar = heatmap.colorbar(im)
        cbar.ax.set_ylabel(zlabel)
    return heatmap, ax, im

def select_unique_values(data):
    """
    Selects a subset of unique values from the given array.

    If the number of unique values is greater than 10, it selects 5 evenly spaced values.

    Args:
    data (np.array): The input array from which to select unique values.

    Returns:
    np.array: An array of selected unique values.

    Example:
    >>> select_unique_values(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    array([ 1,  3,  5,  7,  9, 11])
    """
    categories = data.unique()
    categories = categories[np.argsort(categories)]
    if len(categories) > 10:
        categories = categories[::len(categories)//5]
    return categories

def plot_generic_grid(df, x_col, y_col, data_col):
    """
    Plot a generic grid of subplots for the given data.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    data_col (str): The column name for the data to plot.
    
    Returns:
    fig (matplotlib.figure.Figure): The resulting figure.
    
    Notes:
    - The function will plot a grid of empty subplots for the given data. Plots
      can be added to the subplots later.
    - The given data in data_col should be numerical.
    - The data in x_col and y_col can be either numerical or categorical.
    - The x-axis will be labeled with unique values from the x_col. If there are
      more than 10 unique values, we show 5 evenly spaced values.
    - The y-axis will be labeled with unique values from the y_col. If there are
      more than 10 unique values, we show 5 evenly spaced values.
    """
    categories_x = select_unique_values(df[x_col])
    categories_y = select_unique_values(df[y_col])
    
    fig, axs = plt.subplots(len(categories_y), len(categories_x), figsize=(10, 10))

    hist_range = [df[data_col].min(), df[data_col].max()]
    for i, _ in enumerate(categories_y):
        for j, _ in enumerate(categories_x):
            # Skip drawing any plots for the time being
            # Customize x-axis ticks and labels
            # Remove bounding box
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)
            axs[i, j].yaxis.set_visible(False)  # Hide y-axis  
            if i == 0 and j == 0:  # Top left subplot
                axs[i, j].set_xticks(hist_range)
                axs[i, j].set_xticklabels([f'{hist_range[0]:.2f}', f'{hist_range[1]:.2f}'])
            else:
                axs[i, j].set_xticks([])

    # Add labels at the top and left side of the overall plot
    i = 0
    for ax, col in zip(axs[0], categories_x):
        label = f"{col:.1f}" if isinstance(col, (int, float)) else col
        ax.annotate(f"{x_col}={label}" if i == 0 else f"{label}",
                    xy=(0.5, 1), xytext=(0, 15),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        i+=1

    i = 0
    for ax, row in zip(axs[:, 0], categories_y):
        label = f"{row:.1f}" if isinstance(row, (int, float)) else row
        ax.annotate(f"{y_col}=\n{label}" if i == 0 else f"{label}",
                    xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='right', va='center')
        i += 1

    plt.subplots_adjust(wspace=0.5, hspace=0.2)  # Add more space between the histograms
    return fig, axs

def add_hists_to_grid(axs, df, x_col, y_col, data_col):
    """
    Add histogram plots to the grid of subplots in `axs`.
    
    Args:
    axs (np.array): An array of subplots.
    df (pd.DataFrame): The input DataFrame.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    data_col (str): The column name for the data to plot.
    
    Returns:
    np.array: An array of subplots with the added histogram plots.
    """
    categories_x = select_unique_values(df[x_col])
    categories_y = select_unique_values(df[y_col])
    
    hist_range = [df[data_col].min(), df[data_col].max()]
    for i, category_y in enumerate(categories_y):
        for j, category_x in enumerate(categories_x):
            data = df[(df[x_col] == category_x) & (df[y_col] == category_y)][data_col]
            color = "skyblue"
            axs[i, j].hist(data, bins=10, color = color, edgecolor='black', 
                           density=True, range=hist_range)
            
            
    return axs

def combine_duplicate_x_values(data, freq, tol=1e-6):
    """
    Combine duplicate x-values and sum their frequencies.
    
    Args:
    data (np.array): The input x-values.
    freq (np.array): The input frequencies.
    
    Returns:
    np.array: An array of combined x-values.
    np.array: An array of combined frequencies.
    """
    # First, we consider a tolerance for similar data points to be considered equal
    data = np.round(data, int(-np.log10(tol)))
    unique_data = np.unique(data)
    unique_freq = np.array([np.sum(freq[data == d]) for d in unique_data])
    return unique_data, unique_freq

def add_pdfs_to_grid(axs, df, x_col, y_col, data_col, freq_col):
    """
    Add pdf plots to the grid of subplots in `axs`.
    
    Args:
    axs (np.array): An array of subplots.
    df (pd.DataFrame): The input DataFrame.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    data_col (str): The column name for the data to plot.
    freq_col (str): The column name for the frequencies of the data values.
    
    Returns:
    np.array: An array of subplots with the added pdf plots.
    """
    categories_x = select_unique_values(df[x_col])
    categories_y = select_unique_values(df[y_col])
    
    hist_range = [df[data_col].min(), df[data_col].max()]
    hist_height = df[freq_col].max()
    for i, category_y in enumerate(categories_y):
        for j, category_x in enumerate(categories_x):
            data = df[(df[x_col] == category_x) & (df[y_col] == category_y)][data_col]
            freq = df[(df[x_col] == category_x) & (df[y_col] == category_y)][freq_col]
            # We need to combine any duplicate x-values and sum their frequencies
            data, freq = combine_duplicate_x_values(data, freq, tol=1e-2)
            
            color = "skyblue"
            axs[i, j].set_xlim(hist_range)  # Set the x-limits of the axes
            axs[i, j].set_ylim([0, hist_height])  # Set the y-limits of the axes
            
            # If there is no data for the given category, skip plotting
            if len(data) == 0:
                continue

            # Calculate the width of the bars based on the smallest difference between consecutive x-values
            sorted_data = np.sort(data)
            if len(data) > 1:
                min_diff = np.min(np.diff(sorted_data))
                bar_width = 0.8 * min_diff  # 0.8 is a common choice to leave some space between bars
                x_len = hist_range[1] - hist_range[0]
                # Clip the bar width to be between 1 and 10% of the x-axis length
                bar_width = np.clip(bar_width, 0.01 * x_len, 0.1 * x_len)
            else:
                bar_width = 0.5
            
            axs[i, j].bar(data, freq, color = color, width=bar_width)
            # Add back in x_axis spine which vanishes when plotting bar plots
            # in some cases
            axs[i, j].spines['bottom'].set_visible(True)
            axs[i, j].spines['bottom'].set_linewidth(0.5)
            
    return axs

def test_add_pdfs_to_grid():
    # Create a DataFrame with some dummy data
    mu = 0
    sigma = 1
    M = 500
    x_values =  np.random.normal(mu, sigma, M)
    pdf_values = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(- (x_values - mu)**2 / (2 * sigma**2))
    df = pandas.DataFrame({
        'x_col': np.random.choice(['A', 'B', 'C'], M),
        'y_col': np.random.choice(['X', 'Y', 'Z'], M),
        #  This time we should have a normal distribution
        'data_col': x_values,
        # Each freq value should be between 1 and 0 and add up to 1
        'freq_col': pdf_values,
    })

    # Call setup_grid to get the grid of subplots
    fig, axs = plot_generic_grid(df, 'x_col', 'y_col', 'data_col')

    # Call add_pdfs_to_grid
    axs = add_pdfs_to_grid(axs, df, 'x_col', 'y_col', 'data_col', 'freq_col')

    # Check that the x-limits of each subplot are the same
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(xlim == xlims[0] for xlim in xlims)
    
    # Check that each subplot has a bar plot if M is high (not guaranteed)
    for ax in fig.axes:
        if M > 200:
            assert len(ax.patches) > 0

def test_add_hists_to_grid():
    # Create a DataFrame with some dummy data
    df = pandas.DataFrame({
        'x_col': np.random.choice(['A', 'B', 'C'], 100),
        'y_col': np.random.choice(['X', 'Y', 'Z'], 100),
        'data_col': np.random.normal(0, 1, 100),
    })

    # Call setup_grid to get the grid of subplots
    fig, axs = plot_generic_grid(df, 'x_col', 'y_col', 'data_col')

    # Call add_pdfs_to_grid
    axs = add_hists_to_grid(axs, df, 'x_col', 'y_col', 'data_col')

    # Check that each subplot has a bar plot
    for ax in fig.axes:
        assert len(ax.patches) > 0

    # Check that the x-limits of each subplot are the same
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(xlim == xlims[0] for xlim in xlims)

