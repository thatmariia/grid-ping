from src.NeuronTypes import *


class ParamsIzhikevich:
    """
    This class contains Izhikevich parameters.

    :param peak_potential: potential at which spikes terminate.
    :type peak_potential: float

    :param alpha_E: describes the timescale of recovery for excitatory neurons.
    :type alpha_E: float

    :param beta_E: describes the sensitivity of recovery to the subthreshold fluctuations of potential for excitatory neurons.
    :type beta_E: float

    :param gamma_E: describes the after-spike reset value of potential for excitatory neurons.
    :type gamma_E: float

    :param zeta_E: describes the after-spike reset of recovery for excitatory neurons.
    :type zeta_E: float

    :param alpha_I: describes the timescale of recovery for inhibitory neurons.
    :type alpha_I: float

    :param beta_I: describes the sensitivity of recovery to the subthreshold fluctuations of potential for inhibitory neurons.
    :type beta_I: float

    :param gamma_I: describes the after-spike reset value of potential for inhibitory neurons.
    :type gamma_I: float

    :param zeta_I: describes the after-spike reset of recovery for inhibitory neurons.
    :type zeta_I: float


    :ivar peak_potential: potential at which spikes terminate.
    :ivar alpha: describes the timescale of recovery.
    :ivar beta: describes the sensitivity of recovery to the subthreshold fluctuations of potential.
    :ivar gamma: gamma describes the after-spike reset value of potential.
    :ivar zeta: zeta describes the after-spike reset of recovery.
    """

    def __init__(
            self,
            peak_potential: float,
            alpha_E: float, beta_E: float, gamma_E: float, zeta_E: float,
            alpha_I: float, beta_I: float, gamma_I: float, zeta_I: float
    ):

        self.peak_potential: float = peak_potential
        self.alpha: dict[NeuronTypes, float] = {
            NeuronTypes.EX: alpha_E,
            NeuronTypes.IN: alpha_I
        }
        self.beta: dict[NeuronTypes, float] = {
            NeuronTypes.EX: beta_E,
            NeuronTypes.IN: beta_I
        }
        self.gamma: dict[NeuronTypes, float] = {
            NeuronTypes.EX: gamma_E,
            NeuronTypes.IN: gamma_I
        }
        self.zeta: dict[NeuronTypes, float] = {
            NeuronTypes.EX: zeta_E,
            NeuronTypes.IN: zeta_I
        }

