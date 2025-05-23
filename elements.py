import numpy as np
import scipy as sp

from estimage.entities import estimate

import utils


class Elements:
    def __init__(self, dom, scales, parameters):
        self.scales = scales
        self.dom = dom
        self.parameters = dict()
        self._assign_parameters(parameters)

    def _assign_parameters(self, parameters):
        raise NotImplementedError
        
    def compute_elements(self):
        raise NotImplementedError

    def compute_coefficient_of_variance(self):
        raise NotImplementedError


class LognormElements(Elements):
    def _assign_parameters(self, parameters):
        self.parameters["good_lognorm_shape"] = parameters.GOOD_LOGNORM_SHAPE

    def compute_elements(self):
        shape = self.parameters["good_lognorm_shape"]
        homs = np.zeros((len(self.scales), self.dom.size))
        for ii, scale in enumerate(self.scales):
            dist = utils.get_lognorm_dist(scale, shape)
            vals = dist.pdf(self.dom)
            vals *= scale
            homs[ii, :] = vals
        return homs

    def compute_coefficient_of_variance(self):
        shape = self.parameters["good_lognorm_shape"]
        return self.lognorm_coefficient_of_variance(shape)

    @staticmethod
    def lognorm_coefficient_of_variance(shape):
        return np.sqrt(np.exp(shape ** 2) - 1)
        

class PertElements(Elements):
    def _assign_parameters(self, parameters):
        self.parameters["good_lognorm_shape"] = parameters.GOOD_LOGNORM_SHAPE
        self.parameters["gamma"] = parameters.GAMMA

    def compute_elements(self):
        homs = np.zeros((self.scales.size, self.dom.size))
        for ii, scale in enumerate(self.scales):
            est = self.get_corresponding_estimate(scale)
            _, vals = est.get_pert(len(self.dom,), self.dom)
            vals *= scale
            homs[ii, :] = vals
        return homs

    def get_corresponding_estimate(self, scale):
        shape = self.parameters["good_lognorm_shape"]
        gamma = self.parameters["gamma"]
        lognorm_scale = utils.norm_lognorm_scale(scale, shape)
        est = self.produce_estimate_from_lognorm(lognorm_scale, shape, gamma)
        return est

    def compute_coefficient_of_variance(self):
        est = self.get_corresponding_estimate(1)
        return 100 * est.sigma / est.expected

    @staticmethod
    def produce_estimate_from_lognorm(lognorm_scale, shape, gamma):
        expected = np.exp(shape ** 2 / 2.0) * lognorm_scale
        # lognorm_scale = expected / np.exp(shape ** 2 / 2.0)
        mode = lognorm_scale / np.exp(shape ** 2)
        variance = (np.exp(shape ** 2) - 1) * lognorm_scale ** 2 * np.exp(shape ** 2)
        optimistic, pessimistic = estimate.calculate_o_p_ext(mode, expected, variance, gamma)
        return estimate.Estimate.from_triple(mode, optimistic, pessimistic, gamma)


class GaussElements(Elements):
    def _assign_parameters(self, parameters):
        self.parameters["good_gauss_shape"] = parameters.GOOD_GAUSS_SHAPE

    def compute_elements(self):
        shape = self.parameters["good_gauss_shape"]
        homs = np.zeros((self.scales.size, self.dom.size))
        for ii, scale in enumerate(self.scales):
            vals = sp.stats.norm(loc=scale, scale=scale * shape).pdf(self.dom)
            vals *= scale
            homs[ii, :] = vals
        return homs

    def compute_coefficient_of_variance(self):
        shape = self.parameters["good_gauss_shape"]
        return 100 * shape


class ElementsFactory:
    def __init__(self, dom, scales, parameters):
        self.dom = dom
        self.scales = scales
        self.parameters = parameters

    def get_elements(self, mode):
        modes = dict(
            l=LognormElements,
            p=PertElements,
            g=GaussElements,
        )
        ret_factory = modes[mode](self.dom, self.scales, self.parameters)
        ret = ret_factory.compute_elements()
        return ret
