from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Union

# FEA colors
fea_blue = "#3F83E4"
fea_green = "#26D18A"
fea_red = "#E05554"
fea_yellow = "#FFC659"
plot_area = "#f0eada"
patch_area = "#fffaed"

# FEA default axes style
default_ax_kwargs = {
    "facecolor": plot_area,
    "title_kwargs": {"fontweight": "bold", "fontsize": 12, "fontname": "Arial"},
    "xticklabel_kwargs": {"rotation": 45, "ha": "right", "wrap": True, "fontsize": 10},
    "yticklabel_kwargs": {"fontsize": 10},
    "tick_kwargs": {
        "top": False,
        "bottom": False,
        "left": False,
        "right": False,
        "labelleft": True,
        "labelbottom": True,
    },
    "grid_kwargs": {
        "visible": True,
        "color": "#f46c63",
        "linestyle": ":",
        "linewidth": 0.1,
    },
}


def set_rcParams():

    plt.style.use("Solarize_Light2")

    plt.rc("font", size=13)  # controls default text size
    plt.rc("axes", titlesize=10)  # fontsize of the title
    plt.rc("axes", labelsize=13)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=13)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=13)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=13)  # fontsize of the legend


def hide_spines(ax, positions=["top", "right"]):
    """
    Pass a matplotlib axis and list of positions with spines to be removed

    args:
        ax:          Matplotlib axis object
        positions:   Python list e.g. ['top', 'bottom']
    """
    assert isinstance(positions, list), "Position must be passed as a list "

    for position in positions:
        ax.spines[position].set_visible(False)


def set_axis_attributes_and_style(
    axes,
    axes_attributes: Union[dict, list, None] = None,
    kwargs: dict = default_ax_kwargs,
    hide_spine_positions: list = ["top", "right", "left"],
):

    """
    Set axis style using default kwargs
    """

    # extract kwarg subsets
    kwargs = kwargs.copy()
    grid_kwargs = kwargs.pop("grid_kwargs")
    xtick_kwargs = kwargs.pop("xticklabel_kwargs")
    ytick_kwargs = kwargs.pop("yticklabel_kwargs")
    tick_kwargs = kwargs.pop("tick_kwargs")
    title_kwargs = kwargs.pop("title_kwargs")

    # check if single axes or multiple
    if isinstance(axes, Axes):
        axes = [axes]

    # iterate through axes and set style
    for ax in axes:
        ax.set(**kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), **xtick_kwargs)
        ax.set_yticklabels(ax.get_yticklabels(), **ytick_kwargs)
        ax.tick_params(**tick_kwargs)
        ax.set_title(ax.get_title(), **title_kwargs)
        ax.grid(**grid_kwargs)

    # hide spines
    for ax in axes:
        hide_spines(ax, positions=hide_spine_positions)

    if isinstance(axes_attributes, dict):
        axes_attributes = [axes_attributes.copy()]

    if axes_attributes is not None:
        assert len(axes_attributes) == len(
            axes
        ), f"The number of attributes being assigned ({len(axes_attributes)}) must match the number of axis ({len(axes)})"

        for i, ax_att_kwargs in enumerate(axes_attributes):
            axes[i].set(**ax_att_kwargs)
