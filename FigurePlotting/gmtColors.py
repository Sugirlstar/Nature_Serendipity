import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# set the colors

def colormaker(hex_colors):

    rgb_colors = [mcolors.to_rgb(h) for h in hex_colors]

    # make the colormap
    sealand = mcolors.LinearSegmentedColormap.from_list("from_image", rgb_colors)

    # plot the colormap
    data = np.linspace(0, 1, 400).reshape(20, 20)
    plt.imshow(data, cmap=sealand, origin="lower")
    plt.colorbar()
    plt.title("cmap from image palette")
    plt.show()

    return sealand

sealand = colormaker(["#6779ff", "#66ecff", "#66ff9f", "#9fff65", "#ffe9a6", "#ffbca6", "#ffafbe", "#ffc2e5"])
panoply = colormaker(["#2050ff", "#6dc1ff", "#9beeff", "#ceffff", "#ffeb03", "#ff9100", "#ff0000", "#9e0101"])
abyss = colormaker(["#0a0f1a", "#1c2d51", "#1c2d51", "#315a92", "#3e7eba", "#52a8de", "#73c6ec", "#c1e8f7"])
cool = colormaker(["#15efff", "#31cfff", "#4fafff", "#708fff", "#8f70ff", "#af50ff", "#cf30ff", "#ef0fff"])
haxby = colormaker(["#1405af", "#1a66f1", "#45caff", "#89ecae", "#f0ec79", "#f4754b", "#ff9e9e", "#ffebeb"])
# panoply = colormaker(["#2050ff", "", "", "", "", "", "", "", ""])

sealand_rev = colormaker(["#ffc2e5", "#ffafbe", "#ffbca6", "#ffe9a6", "#9fff65", "#66ff9f", "#66ecff", "#6779ff"])
panoply_rev = colormaker(["#9e0101", "#ff0000", "#ff9100", "#ffeb03", "#ceffff", "#9beeff", "#6dc1ff", "#2050ff"])
abyss_rev = colormaker(["#c1e8f7", "#73c6ec", "#52a8de", "#3e7eba", "#315a92", "#1c2d51", "#1c2d51", "#0a0f1a"])
cool_rev = colormaker(["#ef0fff", "#cf30ff", "#af50ff", "#8f70ff", "#708fff", "#4fafff", "#31cfff", "#15efff"])
haxby_rev = colormaker(["#ffebeb", "#ff9e9e", "#f4754b", "#f0ec79", "#89ecae", "#45caff", "#1a66f1", "#1405af"])


