import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


def plot(output, index_col,gsom_map = None, file_name="gsom", file_type = ".pdf", figure_label="GSOM Map", max_text=3, max_length=30, cmap_colors="Paired", show_index=True):
    """
    plot GSOM nodes with their clustered data points upto max_text labels
    :param output:
    :param index_col:
    :param file_name:
    :param max_text:
    :param max_length:
    :param cmap_colors:
    :param show_index:
    :return:
    """

    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots()
    if gsom_map:
        ax.plot(gsom_map.node_coordinate[:180,0], gsom_map.node_coordinate[:180,1], 'o',
                color=listed_color_map.colors[0], markersize=2)
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        ax.plot(x, y, 'o', color=listed_color_map.colors[i['hit_count']], markersize=2)
        if show_index:
	        if i['hit_count']>0:
	            label = ", ".join(map(str,i[index_col][0:max_text]))
	        else:
	            label = ""
	        txt = ax.text(x, y,label, ha='left', va='center', wrap=True, fontsize=4)
	        txt._get_wrap_line_width = lambda: max_length  # wrap to n screen pixels

    ax.set_title(figure_label)
    plt.savefig(file_name+file_type)


def _get_color_map(max_count, alpha=0.5, cmap_colors="Reds"):

    np.random.seed(1)

    cmap = cm.get_cmap(cmap_colors, max_count + 1)  # set how many colors you want in color map
    # https://matplotlib.org/examples/color/colormaps_reference.html

    color_list = []
    for ind in range(cmap.N):
        c = []
        for x in cmap(ind)[:3]: c.append(x * alpha)
        color_list.append(tuple(c))

    return colors.ListedColormap(color_list, name='gsom_color_list')
