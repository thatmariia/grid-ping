"""Constants"""

"""Network connectivity constants"""

# excitatory  to excitatory
EE = 0.004
sEE = 0.4

# excitatory  to inhibitory
EI = 0.07
sEI = 0.3

# inhibitory to excitatory
IE = -0.04
sIE = 0.3

# inhibitory to inhibitory
II = -0.015
sII = 0.3

"""Izhikevich neuron params"""

# for timescale of recovery variable u
a_EXCIT = 0.02
a_INHIBIT = 0.1

# for sensitivity of u to sub-threshold oscillations of v
b_EXCIT = 0.2
b_INHIBIT = 0.2

# for membrane voltage after spike (after-spike reset of v)
c_EXCIT = -65
c_INHIBIT = -65

# for after-spike reset of recovery variable u
d_EXCIT = 8
d_INHIBIT = 2

# for initial values of v = voltage (membrane potential)
v_INIT = -65

"""Gaussian input"""

# Gaussian excitatory input
GAUSSIAN_EXCIT_INPUT = 1.5

# Gaussian inhibitory input
GAUSSIAN_INHIBIT_INPUT = 1.5

"""Synaptic constants"""

# TODO
DECAY_AMPA = 1.0
RISE_AMPA = 0.1

# TODO
DECAY_GABA = 4.0
RISE_GABA = 0.1