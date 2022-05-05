from src.Params import *

from src.NeuronTypes import *


class ParamsIzhikevich(Params):
    """
    This class contains Izhikevich parameters.

    * alpha describes the timescale of recovery;
    * beta describes the sensitivity of recovery to the subthreshold fluctuations of potential;
    * gamma describes the after-spike reset value of potential;
    * zeta describes the after-spike reset of recovery.

    :param peak_potential: potential at which spikes terminate.
    :type peak_potential: float

    :type alpha_E: float
    :type beta_E: float
    :type gamma_E: float
    :type zeta_E: float
    :type alpha_I: float
    :type beta_I: float
    :type gamma_I: float
    :type zeta_I: float
    """

    def __init__(
            self,
            peak_potential: float,
            alpha_E: float, beta_E: float, gamma_E: float, zeta_E: float,
            alpha_I: float, beta_I: float, gamma_I: float, zeta_I: float
    ):

        self.peak_potential = peak_potential
        self.alpha = {
            NeuronTypes.E: alpha_E,
            NeuronTypes.I: alpha_I
        }
        self.beta = {
            NeuronTypes.E: beta_E,
            NeuronTypes.I: beta_I
        }
        self.gamma = {
            NeuronTypes.E: gamma_E,
            NeuronTypes.I: gamma_I
        }
        self.zeta = {
            NeuronTypes.E: zeta_E,
            NeuronTypes.I: zeta_I
        }

