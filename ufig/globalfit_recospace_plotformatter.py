import numpy as np

class PlotFormatter():
    def __init__(self):
        # define allowed things
        self.det_configs = [
            "IC86_pass2_SnowStorm_v2_tracks",
            "IC86_pass2_SnowStorm_v2_cscd_cascade"
        ]
        self.dimensions = ["energy", "zenith", "ra"]
        self.sum_indices = {
            "energy": (1, 2),
            "zenith": (0, 2),
            "ra": (0, 1),
        }

        # set default binnings
        self.binnings = {
            "IC86_pass2_SnowStorm_v2_tracks": {},
            "IC86_pass2_SnowStorm_v2_cscd_cascade": {},
        }
        self.binnings["IC86_pass2_SnowStorm_v2_tracks"]["energy"] = np.logspace(
            2.5, 7, 46
        )
        self.binnings["IC86_pass2_SnowStorm_v2_tracks"]["zenith"] = np.linspace(
            -1, 0.0872, 34
        )[::-1]  # flip because hists are saved in zenith (0=upgoing, pi=horizontal)
        self.binnings["IC86_pass2_SnowStorm_v2_tracks"]["ra"] = np.linspace(
            0, 6.28319, 181
        )
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "energy"] = np.logspace(2.8, 7, 22)
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "zenith"] = np.sort([-1., 0.2, 0.6, 1.])[::-1]
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "ra"] = np.linspace(0, 6.28319, 19)

        # set default labels
        self.reco_labels = {
            "IC86_pass2_SnowStorm_v2_tracks": {},
            "IC86_pass2_SnowStorm_v2_cscd_cascade": {}, 
        }
        self.reco_labels["IC86_pass2_SnowStorm_v2_tracks"]["energy"] = "Truncated Energy [GeV]"
        self.reco_labels["IC86_pass2_SnowStorm_v2_tracks"]["zenith"] = "SplineMPE cos($\Theta$)"
        self.reco_labels["IC86_pass2_SnowStorm_v2_tracks"]["ra"] = "SplineMPE RA [rad]"
        self.reco_labels["IC86_pass2_SnowStorm_v2_cscd_cascade"]["energy"] = "Monopod Energy [GeV]"
        self.reco_labels["IC86_pass2_SnowStorm_v2_cscd_cascade"]["zenith"] = "Monopod cos($\Theta$)"
        self.reco_labels["IC86_pass2_SnowStorm_v2_cscd_cascade"]["ra"] = "Monopod RA [rad]"
    
    def add_reco_xlabel(self, ax, det_conf, dimension):
        ax.set_xlabel(self.reco_labels[det_conf][dimension])
        return self.reco_labels[det_conf][dimension]

    def add_reco_ylabel(self, ax, det_conf, dimension):
        """For 2D plots"""
        ax.set_ylabel(self.reco_labels[det_conf][dimension])
        return self.reco_labels[det_conf][dimension]

    def apply_settings(self, ax, det_conf, dimension):

        # always
        self.add_reco_xlabel(ax, det_conf, dimension)
        ax.set_yscale("log")
        ax.set_ylabel("Expected Events")
        ax.legend()
        ax.set_xlim(
            min(self.binnings[det_conf][dimension]),
            max(self.binnings[det_conf][dimension])
        )
        # energy
        if dimension == "energy":
            ax.set_ylim(1e-2, 1e7)
            ax.set_xscale("log")
            if det_conf == "IC86_pass2_SnowStorm_v2_tracks":
                ax.legend(
                    bbox_to_anchor=(1.1, 1.0), facecolor="white", framealpha=1
                )
