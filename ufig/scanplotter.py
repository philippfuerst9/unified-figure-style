"""Implements a plotting wrapper around NNMFits ResultHandler classes.
"""
import math
import yaml
import numpy as np
from ufig import FigureHandler


class ScanPlotter():
    """Plotting wrapper around NNMFits ResultHandler classes.
    """
    def __init__(self, parameter_plot_config="parameter_plot_config.yaml"):
        """
        Initialize a Plotter.

        Parameters:
        - scan_hdl_dict (dict): Dictionary containing scan handlers:
            [name][asimov_hdl] = asimov scan handler
            [name][pseudoexp_hdl] = pseudoexp scan handler
            [name][scan_settings] = kwargs, e.g. [color...]
            [name][injection_points] = dict of injected parameters, or None
        - parameter_plot_config (str): Path to the parameter settings file.
        """
        self.scan_suite_dict = {}
        self.scan_names = []
        with open(parameter_plot_config, "r", encoding="utf-8") as stream:
            self.parameter_plot_config = yaml.safe_load(stream)[
                "Parameters"]  # Fluxes part ist not needed for this use case

    @classmethod
    def from_dict(
        cls, scan_hdl_dict, parameter_plot_config="parameter_plot_config.yaml"
    ):
        """
        Initialize a Plotter from a dictionary.

        Returns:
        - cls: Plotter instance.
        """
        plotter = cls(parameter_plot_config=parameter_plot_config)
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
        self, name, asimov_hdl=None, pseudoexp_hdl=None, scan_settings=None
    ):
        """
        Add a scan to the plotter.

        Parameters:
        - name (str): Name of the scan.
        - asimov_hdl (ScanHandler, optional): Asimov scan handler. Defaults to None.
        - pseudoexp_hdl (ScanHandler, optional): Pseudoexperiment scan handler. Defaults to None.
        - scan_settings (dict, optional): Settings for the scan. Defaults to None.

        Returns:
        - None
        """
        # Raise an Error if both handlers are None
        if asimov_hdl is None and pseudoexp_hdl is None:
            raise ValueError("Both asimov_hdl and pseudoexp_hdl are None.")

        self.scan_suite_dict[name] = {
            "asimov_hdl": asimov_hdl,
            "pseudoexp_hdl": pseudoexp_hdl,
            "scan_settings": scan_settings
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
        - **kwargs: Additional keyword arguments for the FigureHandler.

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
            # add a second y axis to ax and plot the histogram there:
            ax.stairs(
                hist, bin_edges, **scan_suite["pseudoexp_settings"], **kwargs
            )
            ax.set_ylabel("PDF")

    def plot_asimov_scan_in_subplot(
        self, scan_name, param, ax, plot_inject=False, **kwargs
    ):
        """
        Plot a scan.

        Returns:
        - None
        """
        asimov_hdl = self.scan_suite_dict[scan_name].get("asimov_hdl")

        if asimov_hdl is not None:
            x, y = asimov_hdl.get_scan_xy(param)
            ax.plot(
                x, y, **self.scan_suite_dict[scan_name]["asimov_settings"],
                **kwargs
            )
            ax.set_ylim(*self.parameter_plot_config[param]["ylims"])
            ax.set_ylabel(r"$-2\Delta \log \mathcal{L}$")

            if plot_inject:
                injection_points = self.scan_suite_dict[scan_name].get(
                    "injection_points", None
                )
                if injection_points is not None:
                    value = injection_points[
                        param
                    ]  # this has to exist, otherwise config is probably wrong
                    ax.axvline(
                        value, color="black", linestyle="-", label="Injected"
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
            ax.set_ylabel("Parameter Value")

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
        - **kwargs: Additional keyword arguments for the FigureHandler.
        """

        if do_pseudoexp and do_add_pars:
            raise ValueError(
                "do_pseudoexp and do_add_pars cannot be True at the same time."
            )

        # determine parameters
        tmp_scan = self.scan_suite_dict[scans_to_plot[0]].get(
            "asimov_hdl", None
        )
        if tmp_scan is not None:
            fitres = sorted(list(tmp_scan.get_freefit().keys()))
            fit_params = [
                p for p in fitres if p != 'llh' and p != 'fit_success'
            ]
        elif tmp_scan is None:
            tmp_scan = self.scan_suite_dict[scans_to_plot[0]].get(
                "pseudoexp_hdl", None
            )
            if tmp_scan is not None:
                fit_params = tmp_scan.get_param_names()
            else:
                raise ValueError("No scan found in scans_to_plot")
        fit_params = sorted(fit_params)

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
                            "injection_points", None) != injection_points:
                        raise ValueError(
                            "Injected parameters are not equal for all scans."
                        )

        # create figure using FigureHandler:
        fig_hdl = FigureHandler(name, nrows=nrows, ncols=ncols, **kwargs)
        axes = fig_hdl.axes

        for i, ax in enumerate(axes):
            if i < len(params_to_plot):
                param = params_to_plot[i]
                if do_pseudoexp or do_add_pars:
                    ax2 = ax.twinx()
                for scan_name in scans_to_plot:
                    if do_asimov:
                        # plot injected parameters only for the first scan:
                        plot_inject = plot_inject and scan_name == scans_to_plot[0]

                        # check if the params was actually scanned:
                        if param in self.scan_suite_dict[scan_name][
                                "asimov_hdl"].get_scan_list():
                            self.plot_asimov_scan_in_subplot(
                                scan_name, param, ax, plot_inject=plot_inject
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

                # add legend only to first plot for now:
                if i == 0:
                    # collect all labels from ax, ax2:
                    handles, labels = ax.get_legend_handles_labels()
                    if do_pseudoexp or do_add_pars:
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        handles += handles2
                        labels += labels2
                    ax.legend(handles, labels)
            else:
                ax.axis('off')
        return fig_hdl

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
