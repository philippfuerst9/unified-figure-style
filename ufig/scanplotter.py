"""Implements a plotting wrapper around NNMFits ResultHandler classes.
"""
import math
import yaml
import numpy as np
from ufig.figure_helpers import FigureHandler
import importlib.resources as resources


class ScanPlotter():
    """Plotting wrapper around NNMFits ResultHandler classes.
    """
    def __init__(self, override_parameter_plot_config=None):
        """
        Initialize a Plotter.

        Parameters:
        - scan_hdl_dict (dict): Dictionary containing scan handlers:
            [name][asimov_hdl] = asimov scan handler
            [name][pseudoexp_hdl] = pseudoexp scan handler
            [name][asimov_settings] = kwargs, e.g. [color...], passed to plt.plot()
            [name][pseudoexp_settings] = kwargs, e.g. [color...], passed to ptl.stairs()
            [name][injection_points] = dict of injected parameters, or None
        - override_parameter_plot_config (dict, optional): Override the parameter plot config. Defaults to None.
        """
        self.scan_suite_dict = {}
        self.scan_names = []
        if override_parameter_plot_config is not None:
            self.parameter_plot_config = override_parameter_plot_config
        else:
            with resources.files("ufig").joinpath("parameter_plot_config.yaml"
                                                 ).open('rb') as f:
                self.parameter_plot_config = yaml.safe_load(f)["Parameters"]

    @classmethod
    def from_dict(cls, scan_hdl_dict, override_parameter_plot_config=None):
        """
        Initialize a Plotter from a dictionary.

        Parameters:
        - scan_hdl_dict (dict): Dictionary containing scan handlers:
            [name][asimov_hdl] = asimov scan handler
            [name][pseudoexp_hdl] = pseudoexp scan handler
            [name][asimov_settings] = kwargs, e.g. [color...], passed to plt.plot()
            [name][pseudoexp_settings] = kwargs, e.g. [color...], passed to ptl.stairs()
            [name][injection_points] = dict of injected parameters, or None

        Returns:
        - cls: Plotter instance.
        """
        plotter = cls(
            override_parameter_plot_config=override_parameter_plot_config
        )
        plotter.scan_suite_dict = scan_hdl_dict
        plotter.scan_names = list(scan_hdl_dict.keys())
        return plotter

    @staticmethod
    def find_injection_points(fit_configuration_file):
        """
        Find the injection points for a fit configuration, or None if no
        input params are given.
        Assumes all injected params are explicitly written down in the 
        analysis config part of the file (!)

        Parameters:
        - fit_configuration_file (str): Path to the fit configuration file.

        Returns:
        - dict: Dictionary containing the injection points.
        """
        with open(fit_configuration_file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config["analysis"].get("input_params", None)

    def add_injection_points(self, scan_name, injection_points):
        """
        Add injection points to a scan.

        Parameters:
        - scan_name (str): Name of the scan.
        - injection_points (dict): Dictionary containing the injection points.

        Returns:
        - None
        """
        self.scan_suite_dict[scan_name]["injection_points"] = injection_points

    def add_scan(
        self,
        name,
        asimov_hdl=None,
        pseudoexp_hdl=None,
        asimov_settings=None,
        pseudoexp_settings=None
    ):
        """
        Add a scan to the plotter.

        Parameters:
        - name (str): Name of the scan.
        - asimov_hdl (ScanHandler, optional): Asimov scan handler. Defaults to None.
        - pseudoexp_hdl (ScanHandler, optional): Pseudoexperiment scan handler. Defaults to None.
        - asimov_settings (dict, optional): Settings for the asimov scan plots. Defaults to None.
        - pseudoexp_settings (dict, optional): Settings for the pseudoexp plots. Defaults to None.

        Returns:
        - None
        """
        # Raise an Error if both handlers are None
        if asimov_hdl is None and pseudoexp_hdl is None:
            raise ValueError("Both asimov_hdl and pseudoexp_hdl are None.")

        self.scan_suite_dict[name] = {
            "asimov_hdl": asimov_hdl,
            "pseudoexp_hdl": pseudoexp_hdl,
            "asimov_settings": asimov_settings,
            "pseudoexp_settings": pseudoexp_settings
        }
        # add name to scan_names list:
        self.scan_names.append(name)

    def plot_pseudoexp_in_subplot(
        self, scan_name, param, ax, nbins=10, **kwargs
    ):
        """
        Plot a pseudoexperiment.

        Parameters:
        - name (str): Name of the scan.
        - savepath (str, optional): Path to save the figure. Defaults to None.
        - **kwargs: Additional keyword arguments, passed to pyplot.stairs()

        Returns:
        - None
        """
        scan_suite = self.scan_suite_dict[scan_name]
        pseudoexp_hdl = scan_suite.get("pseudoexp_hdl")

        # do the Pseudoexp plot if pseudoexp hdl is not None:
        if pseudoexp_hdl is not None:
            if self.parameter_plot_config[param].get("xlims") is not None:
                bin_edges = np.linspace(
                    *self.parameter_plot_config[param]["xlims"], nbins
                )
                hist, bin_edges = pseudoexp_hdl.get_param_hist(
                    param, bins=bin_edges, density=True
                )

            else:
                hist, bin_edges = pseudoexp_hdl.get_param_hist(
                    param, density=True
                )

            actual_kwargs = self.scan_suite_dict[scan_name]["pseudoexp_settings"
                                                           ].copy()
            actual_kwargs.update(kwargs)
            # plot (usually on a second y axis)
            ax.stairs(hist, bin_edges, **actual_kwargs)

    def plot_asimov_scan_in_subplot(
        self,
        scan_name,
        param,
        ax,
        ylabel=r"$-2\Delta \log \mathcal{L}$",
        delta_y=0,
        remove_peaks=False,
        n_iter=2,
        default_ylims=True,
        **kwargs
    ):
        """
        Plot an asimov scan into an axis.

        Parameters:
        - scan_name (str): Name of the scan.
        - param (str): Parameter to plot.
        - ax (matplotlib.axis): Axis to plot into.
        - ylabel (str, optional): Label for the y-axis. Defaults to r"$-2Delta log mathcal{L}$".
        - delta_y (float, optional): Add a constant value to the y values. Defaults to 0.
        - remove_peaks (bool, optional): Remove peaks from the asimov scans. Defaults to False.
        - n_iter (int, optional): Number of iterations to remove peaks. Defaults to 2.
        - default_ylims (bool, optional): Use default y limits. Defaults to True.
        - **kwargs: Additional keyword arguments, passed to pyplot.plot()
        """
        asimov_hdl = self.scan_suite_dict[scan_name].get("asimov_hdl")

        if asimov_hdl is None:
            return

        x, y = asimov_hdl.get_scan_xy(param)

        # if for some reason the freefit is not properly converged, it can be
        # manually fixed here
        y += delta_y

        if remove_peaks:
            x, y = self.remove_peaks(x=x, y=y, n_iter=n_iter)
        # add all asimov settings but override them with given kwargs:
        # make a deepcopy of the asimov settings:
        actual_kwargs = self.scan_suite_dict[scan_name]["asimov_settings"].copy(
        )
        actual_kwargs.update(kwargs)
        ax.plot(x, y, **actual_kwargs)
        if default_ylims:
            ax.set_ylim(*self.parameter_plot_config[param]["ylims"])
        ax.set_ylabel(ylabel)

    def plot_injected_par_in_subplot(
        self, scan_name, param, ax, override_injection_points=None, **kwargs
    ):
        """
        Plot injected parameters.

        Returns:
        - None
        """
        scan_suite = self.scan_suite_dict[scan_name]
        injection_points = scan_suite.get("injection_points", None)
        if override_injection_points is not None:
            injection_points = override_injection_points
        if injection_points is None:
            return
        ax.axvline(
            injection_points[param], color="black", linestyle="--", **kwargs
        )

    def plot_additional_pars_in_subplot(
        self, scan_name, param, ax, secondary_params=None, **kwargs
    ):
        """
        Plot additional parameters.

        Returns:
        - None
        """
        scan_suite = self.scan_suite_dict[scan_name]
        asimov_hdl = scan_suite.get("asimov_hdl")

        if asimov_hdl is not None:
            if secondary_params is None:
                secondary_params = sorted(list(asimov_hdl.get_freefit().keys()))
            for param_secondary in secondary_params:
                if param_secondary in [param, "fit_success", "llh"]:
                    continue

                x, y = asimov_hdl.get_scan_xy_secondary(
                    param,
                    param_secondary,
                )

                ax.plot(
                    x,
                    y,
                    label=self.parameter_plot_config[param_secondary]["label"],
                    color=self.parameter_plot_config[param_secondary]["color"],
                    linestyle=self.parameter_plot_config[param_secondary]
                    ["linestyle"],
                    **kwargs
                )

    @staticmethod
    def remove_peaks(x, y, n_iter=2):
        """
        Remove x, y value pairs where the y values is greater than both adjacent points

        Parameters:
        - x (np.array): x values
        - y (np.array): y values
        - n_iter (int): number of iterations to remove peaks

        Returns:
        - np.array: x values with peaks removed
        - np.array: y values with peaks removed
        """
        for _ in range(n_iter):
            # find peaks
            peaks = np.where((y[1:-1] > y[0:-2]) & (y[1:-1] > y[2:]))[0]
            # remove peaks
            x = np.delete(x, peaks + 1)
            y = np.delete(y, peaks + 1)
        return x, y

    def get_fitted_pars_list(self, scan_name):
        # determine parameters to plot from the first scan in the list
        tmp_scan = self.scan_suite_dict[scan_name].get(
            "asimov_hdl", None
        )
        if tmp_scan is not None:
            fitres = sorted(list(tmp_scan.get_freefit().keys()))
            fit_params = [
                p for p in fitres if p != 'llh' and p != 'fit_success'
            ]

        # if no asimov exists, look for parameters in pseudo-experiments:
        elif tmp_scan is None:
            tmp_scan = self.scan_suite_dict[scan_name].get(
                "pseudoexp_hdl", None
            )
            if tmp_scan is not None:
                fit_params = tmp_scan.get_param_names()
            else:
                raise ValueError("No scan found in scans_to_plot")
        fit_params = sorted(fit_params)
        return fit_params

    def plot_scan_matrix(
        self,
        name,
        scans_to_plot,
        params_to_plot=None,
        do_asimov=True,
        do_pseudoexp=True,
        nbins=10,
        do_add_pars=False,
        plot_inject=False,
        override_injection_points=None,
        nrows=None,
        ncols=None,
        remove_peaks=False,
        n_iter=2,
        default_xlims=True,
        default_ylims=True,
        **kwargs
    ):
        """
        Plot a matrix of all scanned parameters.

        Parameters:
        - name (str): Name of the figure.
        - scans_to_plot (list): List of scan names to plot.
        - params_to_plot (list, optional): List of parameters to plot. Defaults to None.
        - do_asimov (bool, optional): Plot asimov scans. Defaults to True.
        - do_pseudoexp (bool, optional): Plot pseudoexperiments. Defaults to True.
        - nbins (int, optional): Number of bins for the pseudoexp histograms. Defaults to 10.
        - do_add_pars (bool, optional): Plot additional parameters. Defaults to False.
        - plot_inject (bool, optional): Plot injected parameters. Defaults to False.
        - override_injection_points: Override the injection points. Defaults to None.
        - nrows (int, optional): Number of rows. Defaults to None.
        - ncols (int, optional): Number of columns. Defaults to None.
        - remove_peaks (bool, optional): Remove peaks from the asimov scans. Defaults to False.
        - n_iter (int, optional): Number of iterations to remove peaks. Defaults to 2.
        - default_xlims (bool, optional): Use default x limits. Defaults to True.
        - default_ylims (bool, optional): Use default y limits. Defaults to True.
        - **kwargs: Additional keyword arguments for the FigureHandler.

        Returns:
        - FigureHandler: Figure handler.
        - handles (list): List of handles (for the legend, might be empty)
        - labels (list): List of labels (for the legend, might be empty)
        """

        # check configuration
        if do_pseudoexp and do_add_pars:
            raise ValueError(
                "do_pseudoexp and do_add_pars cannot be True at the same time."
            )

        # check what is to be plotted
        # Case 1: asimov scan
        if do_asimov and not do_pseudoexp:
            case = "asimov_only"
        # Case 2: pseudoexp scan
        elif do_pseudoexp and not do_asimov:
            case = "pseudoexp_only"
        # Case 3: asimov and pseudoexp scan
        elif do_pseudoexp and do_asimov:
            case = "both"
        elif do_asimov and do_add_pars:
            case = "add_pars"

        # assume all scans scanned the same parameters, take the first scan
        fit_params = self.get_fitted_pars_list(scans_to_plot[0])

        # if no params are given, just plot all fitted parameters
        if params_to_plot is None:
            params_to_plot = fit_params

        print(params_to_plot)
        # determine matrix size
        nrows, ncols = self.find_closest_factors(len(params_to_plot))

        # create figure using FigureHandler:
        fig_hdl = FigureHandler(name, nrows=nrows, ncols=ncols, **kwargs)
        axes = fig_hdl.axes
        # Initialize an empty dictionary to store handles and labels
        handles_labels = {}

        for i, ax in enumerate(axes):
            if i < len(params_to_plot):
                param = params_to_plot[i]

                # case-dependent axis creation:
                if case == "asimov_only":
                    ax2 = None
                elif case == "pseudoexp_only":
                    ax2 = ax
                elif case == "both" or case == "add_pars":
                    ax2 = ax.twinx()
                    ax2.set_zorder(
                        ax.get_zorder() - 1
                    )  # Set the z-order of ax2 to be one less than ax
                    ax.patch.set_visible(
                        False
                    )  # Make the background of ax transparent so ax2 is visible


                for scan_name in scans_to_plot:

                    # Plot asimov scan
                    if do_asimov:
                        # check if the params was actually scanned/configured:
                        if param in self.scan_suite_dict[scan_name][
                            "asimov_hdl"].get_scan_list():
                            self.plot_asimov_scan_in_subplot(
                                scan_name,
                                param,
                                ax,
                                ylabel=None,
                                remove_peaks=remove_peaks,
                                n_iter=n_iter,
                                default_ylims=default_ylims,
                            )

                    # Plot pseudoexp histogram
                    if do_pseudoexp:
                        # pseudoexp can always be plotted if the param was 
                        # configured:
                        if param in self.scan_suite_dict[scan_name][
                            "pseudoexp_hdl"].get_param_names():
                            # as function of all fit parameters
                            self.plot_pseudoexp_in_subplot(
                                scan_name, param, ax2, nbins=nbins
                            )

                    # Plot injected points (for each scan)
                    if plot_inject:
                        self.plot_injected_par_in_subplot(scan_name, param, ax, override_injection_points=override_injection_points)

                    # Plot additional parameters
                    if do_add_pars:
                        self.plot_additional_pars_in_subplot(
                            scan_name, param, ax2
                        )

                # always add parameter x-axis label:
                ax.set_xlabel(self.parameter_plot_config[param]["label"])

                # case-dependent ylabeling:
                if case == "asimov_only" and i % ncols==0:
                    # Asimov, ylabels on rightmost axes:
                    ax.set_ylabel(r"$-2\Delta \log \mathcal{L}$")
                elif case == "pseudoexp_only" and i % ncols==0:
                    # Pseudoexp, ylabels on rightmost axes:
                    ax2.set_ylabel("PDF")
                elif case == "both":
                    # Asimov left, Pseudoexp right:
                    if i % ncols == 0:
                        ax.set_ylabel(r"$-2\Delta \log \mathcal{L}$")
                    if i % ncols == ncols - 1:
                        ax2.set_ylabel("PDF")
                elif case == "add_pars":
                    # Asimov left, additional parameters right:
                    if i % ncols == 0:
                        ax.set_ylabel(r"$-2\Delta \log \mathcal{L}$")
                    if i % ncols == ncols - 1:
                        ax2.set_ylabel("Parameter Value")

                if default_xlims:
                    ax.set_xlim(*self.parameter_plot_config[param]["xlims"])

                h, l = ax.get_legend_handles_labels()
                if ax2 is not None:
                    # collect all ax2 legend objects
                    h2, l2 = ax2.get_legend_handles_labels()
                    h.extend(h2)
                    l.extend(l2)
                # update (replace existing, but don't create duplicates)
                # legend items
                handles_labels.update(dict(zip(l, h)))
                # convert back to lists
                handles, labels = list(handles_labels.values()), list(
                    handles_labels.keys()
                )
            else:
                ax.axis('off')
        return fig_hdl, handles, labels

    @staticmethod
    def find_closest_factors(n):
        """Find a combination of rows, columns which is approx. square
        and leaves small numbers of empty plots.
        """
        root = math.sqrt(n)
        a1 = math.floor(root)
        a2 = math.ceil(root)

        while True:
            # Check if a1 * a2 is equal or greater to N
            if a1 * a2 >= n:
                if a1 * a2 <= n + 2:
                    return a1, a2
                else:
                    a2 -= 1
            else:
                a2 += 1
                if a1 * a2 >= n:
                    return a1, a2
