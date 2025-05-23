import numpy as np
import scipy as sp


def norm_lognorm_scale(scale, shape):
    normed_scale = scale / np.exp(shape ** 2 / 2.0)
    return normed_scale


def get_lognorm_dist(scale, shape):
    normed_scale = norm_lognorm_scale(scale, shape)
    dist = sp.stats.lognorm(scale=normed_scale, s=shape)
    return dist


def get_discrete_dist(values, relative_probs):
    ret = stats.rv_discrete(name='discrete', values=(values, relative_probs))
    return ret
