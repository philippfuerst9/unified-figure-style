from cycler import cycler
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

mpl.use('agg')


def lighten_color(color, amount=0.5):
    """
    https://gist.github.com/ihincks
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Can also darken colors if amount>1.0

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


linestyles = {
    'loosely dotted': (0, (1, 10)),
    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),
    'long dash with offset': (5, (10, 3)),
    'loosely dashed': (0, (5, 10)),
    'dashed': (0, (5, 5)),
    'densely dashed': (0, (5, 1)),
    'loosely dashdotted': (0, (3, 10, 1, 10)),
    'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),
    'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
    'solid': 'solid',  # Same as (0, ()) or '-'
    # 'dotted': 'dotted',  # Same as (0, (1, 1)) or ':'
    # 'dashed': 'dashed',  # Same as '--'
    # 'dashdot': 'dashdot',  # Same as '-.'
    # '-': 'solid',  # Same as (0, ()) or '-'
    # ':': 'dotted',  # Same as (0, (1, 1)) or ':'
    # '--': 'dashed',  # Same as '--'
    # '-.': 'dashdot'
}  # Same as '-.'


def draw_numbers(
    ax,
    matrix,
    xticks,
    yticks,
    cmap_name,
    round_digits=3,
    n_characters=None,
    char_type="float",
    fontsize=14,
    fix_color=None,
    norm="minmax",
    mask=None,
    ha="center",
    va="center",
):
    """
    Draw numbers on a pcolormesh plot.

    Parameters
    ----------
    ax : matplotlib axis
        axis to draw on.
    matrix : np.ndarray
        2D array of values of the rectangular colorplot.
    xticks : np.array
        1d array of x-centers.
    yticks : np.array
        1d array of y-centers.
    cmap_name : str
        Name of a named matplotlib colormap.mro
    round : int, optional
        Number of precision decimals to round to, by default 3
    n_characters : int, optional
        Explicitly limit the number of characters to write, by default None
    char_type : str, optional
        'int' or 'float', by default 'float'
    fontsize : int, optional
        Fontsize of the numbers, by default 14
    fix_color : str, optional
        If set, all numbers will be written in this color, by default None
    norm : str, optional
        either 'minmax' or 'twoslope': minmax to take min/max values of matrix,
        twoslope if the norm should be centered at 0.
    ha : str, optional
        Horizontal alignment of the text, by default 'center'
    va : str, optional
        Vertical alignment of the text, by default 'center'
    """

    # imports
    from matplotlib import cm
    import matplotlib as mpl
    import matplotlib.colors as colors

    # get the extent of the matrix
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    # get cmap, normalization, and a scalar mappable
    cmap = plt.get_cmap(cmap_name)

    delta = np.max([np.abs(1.0 - np.min(matrix)), np.abs(1.0 - np.max(matrix))])
    if norm == "twoslope":
        norm = colors.TwoSlopeNorm(vmin=1 - delta, vcenter=1.0, vmax=1 + delta)
    elif norm == "minmax":
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = norm
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    # loop through the matrix
    for idx_x, xc in enumerate(xticks):
        for idx_y, yc in enumerate(yticks):

            # get the rgba value of the cmap at the matrix position
            rgb = scalar_map.to_rgba(matrix.T[idx_x,
                                              idx_y])[0:3]  # [0:4] gives rgba
            # print(scalar_map.to_rgba(matrix.T[idx_x, idx_y])[0:4])
            # rgb to greyscale scalers, sum=1.0
            scale_r = 0.299
            scale_g = 0.587
            scale_b = 0.114

            # greyscale
            y = scale_r * rgb[0] + scale_g * rgb[1] + scale_b * rgb[2]

            # set the color depending on brightness
            if y > 156. / 255.:  # literature seems to be 186/255 as switch condition
                c = "black"
            else:
                c = "white"
            # if the color is fixed manually, just set it
            if fix_color is not None:
                c = fix_color
            
            # check if int or float:
            if char_type == "int":
                s = str(int(np.round(matrix.T[idx_x, idx_y], round_digits)))
            elif char_type == "float":
                s = str(np.round(matrix.T[idx_x, idx_y], round_digits))
            else:
                s = str(matrix.T[idx_x, idx_y])

            # get the number to write as a string
            if mask is not None:
                # mask all elements which are 1 (True)
                if mask.T[idx_x, idx_y]:
                    s = ""

            # if n_characters is given, cut s to that amount of characters
            if n_characters is not None:
                s = s[:n_characters]

            # write the text
            ax.text(
                xc, yc, s, fontsize=fontsize, ha=ha, va=va, color=c
            )


def write_preliminary(
    *args,
    size=14,
    x=0.2,
    y=0.2,
    string="IceCube Preliminary",
    c="darkred",
    **kwargs
):
    """
    Figure or axes objects to write text to
    """
    import matplotlib
    texts = []
    for arg in args:
        # arg is a full figure
        if isinstance(arg, matplotlib.figure.Figure):
            text = arg.text(
                x,
                y,
                s=string,
                color=c,
                fontweight="bold",
                horizontalalignment='left',
                verticalalignment='bottom',
                size=size,
                **kwargs
            )  # bold somehow is not working?
        else:  # normal axes
            text = arg.text(
                x,
                y,
                s=string,
                color=c,
                fontweight="bold",
                transform=arg.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom',
                size=size,
                **kwargs
            )  # bold somehow is not working?
            texts.append(text)
    return texts


class FigureHandler():
    """Wrapper class for matplotlib figure and axes objects.
    Arguments:
        name: a name for saving the figure
        width: either 'thesis' or 'beamer', or width in points (pt).
        nrows, ncols: number of rows, columns
        r: figure aspect ratio. 'gold'=golden ratio, 'square'= square plot, 
        'subplotsquares' = square * nrows/ncols..
        man_h, man_w: manual global scaling of figure height and width.
        ratios: Build ratio plots (requires even number of nrows).
    """
    def __init__(
        self,
        name,
        width='thesis',
        nrows=1,
        ncols=1,
        r="gold",
        man_h=1,
        man_w=1,
        ratios=False,
        **kwargs
    ):
        plt.clf()
        self.name = name
        self.nrows = nrows
        self.ncols = ncols
        self.ratio = r
        self.man_h = man_h
        self.man_w = man_w
        self.ratio_bool = ratios
        # Golden ratio: https://disq.us/p/2940ij3
        self.gr = (5**.5 - 1) / 2

        # dimensions
        if width == 'thesis':
            self.width = 404.02908  # standard Latex thesis textwidth
        elif width == 'beamer':
            self.width = 307.28987
        else:
            self.width = width

        self.w = None
        self.h = None
        self.__set_w()
        self.__set_h()

        # figure variables
        self.fig = None
        self.axes = None
        self.__init_figure(**kwargs)

    def __pt_to_inch(self, pt):
        # Convert from pt to inches
        return pt / 72.27

    def __set_w(self):
        """
        Figure width, scalable with manual width
        """
        self.w = self.__pt_to_inch(self.width) * self.man_w

    def __set_h(self):
        """
        Figure height, scalable with manual height
        """
        if self.ratio == "gold":
            self.h = self.w * self.gr * (self.nrows / self.ncols) * self.man_h
        elif self.ratio == "subplotsquares":
            self.h = self.w * self.gr * (self.nrows / self.ncols) * self.man_h
        elif self.ratio == "square":
            self.h = self.w * self.man_h

    def __init_figure(self, **kwargs):
        """Initialize the figure
        with the indicated number of subplots and size.
        """
        fig, axs = plt.subplots(
            figsize=[self.w, self.h],
            nrows=self.nrows,
            ncols=self.ncols,
            **kwargs
        )
        self.fig = fig

        if self.ratio_bool:
            # axis handling with ratio plots
            width_ratios = [1 for _ in range(self.ncols)]
            height_ratios = []

            # make the ratio plots smaller
            if self.nrows % 2 == 0:
                for i in range(int(self.nrows / 2.)):
                    height_ratios.append(2)  # 3.5
                    height_ratios.append(1)
            else:
                raise ValueError("nrows must be even for ratio plots")

            # build the subplot grid
            gs1 = gridspec.GridSpec(
                self.nrows,
                self.ncols,
                height_ratios=height_ratios,
                width_ratios=width_ratios
            )
            gs1.update(wspace=0.26, hspace=0.05)

            axes = []
            for i, gs in enumerate(gs1):
                # ratio plots sharex with the plot above them
                if len(axes) != 0 and (i + 1) % 2 == 0:
                    #print(f"ax No i+1 = {i+1} sharing x axis with ax {axes[-1]}")
                    # ax = plt.subplot(gs, sharex=axes[-1])
                    ax = plt.subplot(gs)
                else:
                    #print(f"ax no i+1 = {i+1} not sharing x axis.")
                    ax = plt.subplot(gs)
                axes.append(ax)

            for i_row in range(self.nrows):
                if i_row % 2 == 0:
                    i_start = i_row * self.ncols
                    i_stop = (i_row + 1) * self.ncols
                    for ax in axes[i_start:i_stop]:
                        plt.setp(ax.get_xticklabels(), visible=False)

            # plt.setp(fig.axes[0].get_xticklabels(), visible=False)
            # plt.setp(fig.axes[1].get_xticklabels(), visible=False)
            print(axes)
            print("first axes")
            print(axes[0])
        if not self.ratio_bool:
            # axis handling without ratio plots
            if self.nrows == 1 and self.ncols == 1:
                axes = axs
            else:
                axes = []
                for ax in axs.reshape(-1):
                    axes.append(ax)
        self.fig = fig
        self.axes = axes
        del fig, axes

    def tight_layout(self):
        self.fig.tight_layout()

    def show(self):
        #self.set_figure()
        return self.fig

    def clear(self):
        plt.clf()
        #self.set_figure()

    def save(
        self,
        path=".",
        format=None,
        dpi=500,
        bbox_inches='tight',
        bbox_extra_artists=None
    ):
        fname = path + "/" + self.name
        if format is not None:
            self.fig.savefig(
                fname + "." + format,
                format=format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                bbox_extra_artists=bbox_extra_artists
            )
        else:
            self.fig.savefig(
                fname + ".pdf",
                format="pdf",
                bbox_inches=bbox_inches,
                bbox_extra_artists=bbox_extra_artists
            )
            self.fig.savefig(
                fname + ".png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                bbox_extra_artists=bbox_extra_artists
            )


# Some plotting functionality


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window_2d(a, shape):  # rolling window for 2D array
    shape = (a.shape[0] - shape[0] +
             1, ) + (a.shape[1] - shape[1] + 1, ) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# custom step function to include the last bin
def full_step(
    x_bins, y, label=None, color="black", alpha=1, ax=None, linestyle="-"
):
    if ax != None:
        ax.step(
            x_bins[:-1],
            y,
            color=color,
            label=label,
            where="post",
            alpha=alpha,
            linestyle=linestyle
        )
        ax.hlines(
            y[-1],
            x_bins[-2],
            x_bins[-1],
            color=color,
            alpha=alpha,
            linestyle=linestyle
        )
    else:
        plt.step(
            x_bins[:-1],
            y,
            color=color,
            label=label,
            where="post",
            alpha=alpha,
            linestyle=linestyle
        )
        plt.hlines(
            y[-1],
            x_bins[-2],
            x_bins[-1],
            color=color,
            alpha=alpha,
            linestyle=linestyle
        )


def plot_hist(ax, hist, bins, yerror, **kwargs):
    l = ax.errorbar(
        np.mean(rolling_window(bins, 2), axis=1),
        hist,
        yerr=yerror,
        drawstyle="steps-mid",
        **kwargs
    )  #, capsize=2.0, capthick=1
    return l


def plot_ratio_single_err(ax, hist, bins, yerror, hist_baseline, **kwargs):
    yerror_ratio = yerror / hist_baseline
    l = ax.errorbar(
        np.mean(rolling_window(bins, 2), axis=1),
        hist / hist_baseline,
        yerr=yerror_ratio,
        drawstyle="steps-mid",
        capsize=2.0,
        capthick=1,
        **kwargs
    )
    return l


def get_ratio_error(hist, hist_baseline, sigma_hist, sigma_baseline):
    return np.sqrt(
        sigma_baseline**2 / hist_baseline**2 +
        hist**2 / hist_baseline**4 * sigma_hist**2
    )


def plot_ratio_double_err(
    ax, hist, hist_baseline, sigma_hist, sigma_baseline, bins, **kwargs
):
    yerror = get_ratio_error(hist, hist_baseline, sigma_hist, sigma_baseline)
    l = ax.errorbar(
        np.mean(rolling_window(bins, 2), axis=1),
        hist / hist_baseline,
        yerr=yerror,
        drawstyle="steps-mid",
        capsize=2.0,
        capthick=1,
        **kwargs
    )
    return l


def full_step(
    x_bins, y, label=None, color="black", alpha=1, ax=None, linestyle="-"
):
    if ax != None:
        ax.step(
            x_bins[:-1],
            y,
            color=color,
            label=label,
            where="post",
            alpha=alpha,
            linestyle=linestyle
        )
        ax.hlines(
            y[-1],
            x_bins[-2],
            x_bins[-1],
            color=color,
            alpha=alpha,
            linestyle=linestyle
        )
    else:
        plt.step(
            x_bins[:-1],
            y,
            color=color,
            label=label,
            where="post",
            alpha=alpha,
            linestyle=linestyle
        )
        plt.hlines(
            y[-1],
            x_bins[-2],
            x_bins[-1],
            color=color,
            alpha=alpha,
            linestyle=linestyle
        )
