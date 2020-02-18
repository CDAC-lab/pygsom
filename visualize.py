import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


def show_gsom(output, max_count,index_col,label_col):
    listed_color_map = _get_color_map(max_count, alpha=0.9)

    fig, ax = plt.subplots()
    for index, i in output.iterrows():
        x=i['x']
        y=i['y']
        if i['hit_count']>0:
            c='red'
            label = ", ".join(map(str,i[index_col]))

        else:
            label = ""
            c='yellow'
        ax.plot(x,y, 'o', color=listed_color_map.colors[i['hit_count']],markersize=2)
        ax.annotate(label, (x, y), fontsize=4)
        print("{},{}-{}".format(x, y,label))


    ax.set_title("GSOM Map")
    #plt.show()
    plt.savefig("gsom.png",dpi=1200)


def _get_color_map(max_count, alpha=0.5):

    np.random.seed(1)

    cmap = cm.get_cmap('Reds', max_count + 1)  # set how many colors you want in color map
    # https://matplotlib.org/examples/color/colormaps_reference.html

    color_list = []
    for ind in range(cmap.N):
        c = []
        for x in cmap(ind)[:3]: c.append(x * alpha)
        color_list.append(tuple(c))

    return colors.ListedColormap(color_list, name='gsom_color_list')
