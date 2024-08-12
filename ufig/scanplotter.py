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
    def from_dict(
        cls, scan_hdl_dict, override_parameter_plot_config=None
    ):
        """
        Initialize a Plotter from a dictionary.

        Returns:
        - cls: Plotter instance.
        """
        plotter = cls(override_parameter_plot_config=override_parameter_plot_config)
        plotter.scan_suite_dict = scan_hdl_dict
        plotter.scan_names = list(scan_hdl_dict.keys())
        return plotter

    @staticmethod
    def find_injection_points(fit_configuration_file):
        """
        Find the injection points for a fit configuration.
        Assumes all injected params are explicitly written down in the 
        analysis config part of the file (!)

        Parameters:
        - fit_configuration_file (str): Path to the fit configuration file.

        Returns:
        - dict: Dictionary containing the injection points.
        """
        with open(fit_configuration_file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config["analysis"]["input_params"]

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
        self, scan_name, param, ax, plot_inject=False, **kwargs
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

        # if plot inject, raise not implemented error:
        if plot_inject:
            raise NotImplementedError(
                "Plotting of injected parameters for pseudoexp is not implemented yet."
            )

        # do the Pseudoexp plot if pseudoexp hdl is not None:
        if pseudoexp_hdl is not None:
            if self.parameter_plot_config[param].get("xlims") is not None:
                bin_edges = np.linspace(
                    *self.parameter_plot_config[param]["xlims"], 10
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
        plot_inject=False,
        ylabel=r"$-2\Delta \log \mathcal{L}$",
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
        - plot_inject (bool, optional): Plot injected parameters. Defaults to False.
        - ylabel (str, optional): Label for the y-axis. Defaults to r"$-2\Delta \log \mathcal{L}$".
        - remove_peaks (bool, optional): Remove peaks from the asimov scans. Defaults to False.
        - n_iter (int, optional): Number of iterations to remove peaks. Defaults to 2.
        - default_ylims (bool, optional): Use default y limits. Defaults to True.
        - **kwargs: Additional keyword arguments, passed to pyplot.plot()
        """
        asimov_hdl = self.scan_suite_dict[scan_name].get("asimov_hdl")

        if asimov_hdl is None:
            return

        x, y = asimov_hdl.get_scan_xy(param)

        if remove_peaks:
            x, y = self.remove_peaks(x=x, y=y, n_iter=n_iter)
        # add all asimov settings but override them with given kwargs:
        # make a deepcopy of the asimov settings:
        actual_kwargs = self.scan_suite_dict[scan_name]["asimov_settings"
                                                        ].copy()
        actual_kwargs.update(kwargs)
        ax.plot(x, y, **actual_kwargs)
        if default_ylims:
            ax.set_ylim(*self.parameter_plot_config[param]["ylims"])
        ax.set_ylabel(ylabel)

        if plot_inject:
            injection_points = self.scan_suite_dict[scan_name].get(
                "injection_points", None
            )
            if injection_points is not None:
                value = injection_points[
                    param
                ]  # this has to exist, otherwise config is probably wrong
                ax.axvline(value, color="black", linestyle="-", zorder=10)

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

    def remove_peaks(self, x, y, n_iter=2):
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
            x = np.delete(x, peaks+1)
            y = np.delete(y, peaks+1)
        return x, y

    def plot_scan_matrix(
        self,
        name,
        scans_to_plot,
        params_to_plot=None,
        do_asimov=True,
        do_pseudoexp=True,
        do_add_pars=False,
        plot_inject=False,
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
        - do_asimov (bool, optional): Plot asimov scans. Defaults to True.
        - do_pseudoexp (bool, optional): Plot pseudoexperiments. Defaults to True.
        - do_add_pars (bool, optional): Plot additional parameters. Defaults to False.
        - plot_inject (bool, optional): Plot injected parameters. Defaults to False.
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

        # determine parameters to plot from the first scan in the list
        tmp_scan = self.scan_suite_dict[scans_to_plot[0]].get(
            "asimov_hdl", None
        )
        if tmp_scan is not None:
            fitres = sorted(list(tmp_scan.get_freefit().keys()))
            fit_params = [
                p for p in fitres if p != 'llh' and p != 'fit_success'
            ]

        # if no asimov exists, look for parameters in pseudo-experiments:
        elif tmp_scan is None:
            tmp_scan = self.scan_suite_dict[scans_to_plot[0]].get(
                "pseudoexp_hdl", None
            )
            if tmp_scan is not None:
                fit_params = tmp_scan.get_param_names()
            else:
                raise ValueError("No scan found in scans_to_plot")
        fit_params = sorted(fit_params)

        # if no params are given, just plot all fitted parameters
        if params_to_plot is None:
            params_to_plot = fit_params

        # determine matrix size
        nrows, ncols = self.find_closest_factors(len(fit_params))

        # check that injected parameters are equal for all asimov scans:
        if plot_inject:
            injection_points = self.scan_suite_dict[scans_to_plot[0]].get(
                "injection_points", None
            )
            if injection_points is not None:
                for scan_name in scans_to_plot:
                    if self.scan_suite_dict[scan_name].get(
                        "injection_points", None
                    ) != injection_points:
                        raise ValueError(
                            "Injected parameters are not equal for all scans."
                        )

        # create figure using FigureHandler:
        fig_hdl = FigureHandler(name, nrows=nrows, ncols=ncols, **kwargs)
        axes = fig_hdl.axes
        # Initialize an empty dictionary to store handles and labels
        handles_labels = {}

        for i, ax in enumerate(axes):
            if i < len(params_to_plot):
                param = params_to_plot[i]

                # create a second y-axis if do_pseudoexp or do_add_pars is True:
                if do_pseudoexp or do_add_pars:
                    ax2 = ax.twinx()
                    ax2.set_zorder(
                        ax.get_zorder() - 1
                    )  # Set the z-order of ax2 to be one less than ax
                    ax.patch.set_visible(
                        False
                    )  # Make the background of ax transparent so ax2 is visible
                else:
                    ax2 = None

                for scan_name in scans_to_plot:
                    if do_asimov:
                        # plot injected parameters only for the first scan:
                        if plot_inject and scan_name == scans_to_plot[0]:
                            plot_inject_c = True
                        else:
                            plot_inject_c = False

                        # check if the params was actually scanned:
                        if param in self.scan_suite_dict[scan_name][
                            "asimov_hdl"].get_scan_list():
                            self.plot_asimov_scan_in_subplot(
                                scan_name,
                                param,
                                ax,
                                plot_inject=plot_inject_c,
                                ylabel=None,
                                remove_peaks=remove_peaks,
                                n_iter=n_iter,
                                default_ylims=default_ylims,
                            )
                    if do_pseudoexp:
                        # pseudoexp can always be plotted
                        # as function of all fit parameters
                        self.plot_pseudoexp_in_subplot(scan_name, param, ax2)
                    if do_add_pars:
                        self.plot_additional_pars_in_subplot(
                            scan_name, param, ax2
                        )

                # always add parameter x-axis label:
                ax.set_xlabel(self.parameter_plot_config[param]["label"])

                # set ylabel (None if not left)
                if i % ncols == 0:
                    ax.set_ylabel(r"$-2\Delta \log \mathcal{L}$")

                # set ylabel on ax2 only if do_pseudoexp and only on the rightmost column plots:
                if do_pseudoexp and i % ncols == ncols - 1:
                    ax2.set_ylabel("PDF")
                # set ylabel on ax2 only if do_add_pars and only on the rightmost column plots:
                if do_add_pars and i % ncols == ncols - 1:
                    ax2.set_ylabel("Parameter Value")
                # always set default xlimits:
                if default_xlims:
                    ax.set_xlim(*self.parameter_plot_config[param]["xlims"])

                h, l = ax.get_legend_handles_labels()
                if ax2 is not None:
                    # move asimov scans to front
                    # ax.set_zorder(ax2.get_zorder()+1)

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
