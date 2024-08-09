import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.text(0.5, 1.5, 'No assimilation', horizontalalignment='center')
plt.text(3.5, 1.5, 'Assimilation', horizontalalignment='center')

ellipse = list()

ellipse.append(Ellipse((0.5, 1), width=0.4, height=0.2,
                  facecolor='none', edgecolor='k', linestyle='-', linewidth=2))
ellipse.append(Ellipse((3.5, 1), width=0.2, height=0.1,
                  facecolor='none', edgecolor='k', linestyle='-', linewidth=2))
ellipse.append(Ellipse((0.5, 0.5), width=0.4, height=0.2,
                  facecolor='none', edgecolor='k', linestyle='--', linewidth=2))
ellipse.append(Ellipse((3.5, 0.5), width=0.2, height=0.1,
                  facecolor='none', edgecolor='k', linestyle='--', linewidth=2))

ax = plt.gca()
for item in ellipse:
    ax.add_patch(item)

plt.text(2, 1.1, "Assimilation date (2022-02-26)", horizontalalignment='center')
plt.annotate("", xy=(3.5, 1), xytext=(0.5, 1),
        arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': 'k',
            'linestyle': '-', 'mutation_scale': 30})

plt.text(2, 0.6, "Evaluation date (2022-05-01)", horizontalalignment='center')
plt.annotate("", xy=(3.5, 0.5), xytext=(0.5, 0.5),
        arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': 'k',
            'linestyle': '--', 'mutation_scale': 30})

plt.xlim(0, 4)
plt.ylim(0, 2)

plt.show()
