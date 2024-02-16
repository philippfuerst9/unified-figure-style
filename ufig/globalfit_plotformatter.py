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
        )
        self.binnings["IC86_pass2_SnowStorm_v2_tracks"]["ra"] = np.linspace(
            0, 6.28319, 181
        )
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "energy"] = np.logspace(2.8, 7, 22)
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "zenith"] = np.sort([-1., 0.2, 0.6, 1.])
        self.binnings["IC86_pass2_SnowStorm_v2_cscd_cascade"][
            "ra"] = np.linspace(0, 6.28319, 19)

    def apply_settings(self, ax, dimension, det_conf):

        # always
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
                ax.set_xlabel("Truncated Energy [GeV]")
                ax.legend(
                    bbox_to_anchor=(1.1, 1.0), facecolor="white", framealpha=1
                )

            elif det_conf == "IC86_pass2_SnowStorm_v2_cscd_cascade":
                ax.set_xlabel("Monopod Energy [GeV]")

        # zenith
        if dimension == "zenith":
            if det_conf == "IC86_pass2_SnowStorm_v2_tracks":
                ax.set_xlabel("SplineMPE cos($\Theta$)")
            elif det_conf == "IC86_pass2_SnowStorm_v2_cscd_cascade":
                ax.set_xlabel("Monopod cos($\Theta$)")

        # Right Ascension
        if dimension == "ra":
            if det_conf == "IC86_pass2_SnowStorm_v2_tracks":
                ax.set_xlabel("SplineMPE RA [rad]")
            elif det_conf == "IC86_pass2_SnowStorm_v2_cscd_cascade":
                ax.set_xlabel("Monopod RA [rad]")