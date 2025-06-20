import numpy as np
import scipy as sp

import pylab as pyl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import estimage
from estimage.entities import estimate
import demos, elements, utils

# Execute me as
# PYTHONPATH=$HOME/git/estimage/ ipython --pylab


def get_slice(dom, val_start, val_end):
    idx_start = np.argmin(np.abs(dom - val_start))
    idx_end = np.argmin(np.abs(dom - val_end))
    return slice(idx_start, idx_end)


class Plotter:
    def __init__(self):
        self.dpi = 300
        self.figsize = np.array((10, 7), dtype=float) * 0.5
        self.current_xlim = 0
        self.current_ylim = 0

    def start(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)

    def store_limits(self):
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

    def apply_limits(self):
        self.ax.set_xlim(self.current_xlim)
        self.ax.set_ylim(self.current_ylim)

    def set_prob_axes(self):
        self.ax.set_ylabel("probability density")
        self.ax.set_yticklabels([])

    def gen_first(self):
        stem_tpl = "01-{}-{}.png"

        dom = np.linspace(0, 25, 500)

        steak_dist = sp.stats.norm(loc=8, scale=0.7)
        steak = steak_dist.pdf(dom)
        print(f"Steak CoV: {100 * steak_dist.std() / steak_dist.mean():.2g}%")
        call_lognorm_shape = 0.32
        call = sp.stats.lognorm(scale=11.5, s=call_lognorm_shape).pdf(dom)
        print(f"Call CoV: {100 * elements.LognormElements.lognorm_coefficient_of_variance(call_lognorm_shape):.2g}%")

        self.start()

        self.ax.fill_between(dom, 0, call, fc="blue", alpha=0.5, label="call")
        self.ax.fill_between(dom, 0, steak, fc="red", alpha=0.5, label="steak")

        self.store_limits()
        self.ax.set_xlabel("time / minutes")
        self.set_prob_axes()
        self.ax.grid()
        self.ax.legend(loc="upper right")

        self.fig.savefig(stem_tpl.format(2, "steak_and_call"), dpi=self.dpi)
        self.ax.clear()

        self.ax.fill_between(dom, 0, steak, fc="red", alpha=0.5, label="steak")

        self.apply_limits()
        self.ax.set_xlabel("time / minutes")
        self.set_prob_axes()
        self.ax.grid()
        self.ax.legend(loc="upper right")

        self.fig.savefig(stem_tpl.format(1, "steak_only"), dpi=self.dpi)
        plt.close(self.fig)

    def gen_second(self):
        stem_tpl = "02-{}-{}.png"
        dom = np.linspace(-1, 15, 500)

        demor = demos.RegDemo(dom)
        demor.GOOD_GAUSS_SHAPE = 0.05
        homs = demor.get_homs("g")
        hom_five = homs[3]

        self.start()
        self.ax.fill_between(dom, 0, hom_five, fc="blue", alpha=0.5, label="5sp")

        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()
        self.fig.savefig(stem_tpl.format(1, "gauss_thin"), dpi=self.dpi)

        self.ax.clear()

        self.ax.fill_between(dom, 0, homs.sum(0), fc="blue", alpha=0.5, label="Any SP")
        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()
        self.fig.savefig(stem_tpl.format(2, "all_gauss_thin"), dpi=self.dpi)

        demor.GOOD_GAUSS_SHAPE = 0.15
        homs = demor.get_homs("g")
        thin_hom_five = hom_five
        hom_five = homs[3]

        self.ax.clear()
        self.ax.fill_between(dom, 0, hom_five, fc="blue", alpha=0.5, label="5 SP")
        self.ax.plot(dom, thin_hom_five, "--", color="black", label="Thin 5 SP")
        self.store_limits()

        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()
        self.fig.savefig(stem_tpl.format(3, "gauss_thick"), dpi=self.dpi)

        self.ax.clear()

        self.ax.fill_between(dom, 0, homs[0], fc="blue", alpha=0.5, label="Any SP")
        for hom in homs[1:]:
            self.ax.fill_between(dom, 0, hom, fc="blue", alpha=0.5)
        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()
        self.fig.savefig(stem_tpl.format(4, "all_gauss_thick"), dpi=self.dpi)

        demor.GOOD_GAUSS_SHAPE = 0.26
        homs = demor.get_homs("g")
        thick_hom_five = homs[3]

        self.ax.clear()
        self.ax.fill_between(dom, 0, thick_hom_five, fc="blue", alpha=0.5, label="5 SP")
        self.ax.plot(dom, thin_hom_five, ":", color="black", label="Thin 5 SP")
        self.ax.plot(dom, hom_five, "--", color="black", label="Medium 5 SP")
        self.store_limits()

        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()
        self.fig.savefig(stem_tpl.format(3, "gauss_really_thick"), dpi=self.dpi)

        plt.close(self.fig)

    def gen_third(self):
        demor = demos.RegDemo
        demor.GOOD_GAUSS_SHAPE = 0.15
        self._gen_third(demor, 3, "g", "gauss_thin_add")
        demor.GOOD_GAUSS_SHAPE = 0.26
        self._gen_third(demor, 4, "g", "gauss_thick_add")
        self._gen_third(demor, 5, "l", "lognorm_add")
        demor.GAMMA = 4
        self._gen_third(demor, 6, "p", "pert_std_add")
        demor.GAMMA = 12
        self._gen_third(demor, 7, "p", "pert_mod_add")
        self._gen_gauss_vs_lognorm(8, demor)
        self._gen_lognorm_autopsy(9, demor)

    def _plot_increment(self, dom, old, hom, incr):
        incr = round(incr)
        self.ax.clear()

        self._plot_unknown_task()
        if old.max() > 0:
            self.ax.fill_between(dom, 0, old, fc="blue", alpha=0.3, label=f"Up to {incr} SP")
        self.ax.fill_between(dom, old, old + hom, fc="blue", alpha=0.5, label=f"{incr} SP")

    def _groom_axes(self):
        self.apply_limits()
        self.ax.grid()
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("size / SP")
        self.set_prob_axes()

    def _plot_unknown_task(self):
        sparse_dom_lead = np.array((-1, 0.5, 1))
        sparse_hom_lead = np.array((0, 0, 1))
        sparse_dom_middle = np.array((1, 7.5))
        sparse_hom_middle = np.array((1, 1))
        sparse_dom_trail = np.array((7.5, 12, 15))
        sparse_hom_trail = np.array((1, 0, 0))
        self.ax.plot(sparse_dom_lead, sparse_hom_lead, "--", color="grey")
        self.ax.plot(sparse_dom_trail, sparse_hom_trail, "--", color="grey")
        self.ax.plot(sparse_dom_middle, sparse_hom_middle, "-", color="grey")
        self.ax.fill_between(sparse_dom_middle, sparse_hom_middle, "--", fc="grey", alpha=0.5, label="unknown task")

    def _gen_gauss_vs_lognorm(self, number, demo_t):
        stem_tpl = "%02d-{}-{}.png" % number
        dom = np.linspace(-1, 13, 500)

        demo = demo_t(dom, 5)

        homs_gauss = demo.get_homs("g")
        homs_lognorm = demo.get_homs("l")
        idx_of_5sp = 3

        self.start()

        self.ax.plot(dom, homs_gauss[idx_of_5sp], "--", color="black", label="5 SP Thick Gauss")
        self.ax.fill_between(dom, 0, homs_lognorm[idx_of_5sp], fc="blue", alpha=0.6, label="5 SP Lognorm")
        self.ax.axvline(5, color="springgreen", label="5 SP")
        self.store_limits()
        self._groom_axes()

        self.fig.savefig(stem_tpl.format(1, "gauss_vs_lognorm"), dpi=self.dpi)

        plt.close(self.fig)

    def _gen_lognorm_autopsy(self, number, demo_t):
        stem_tpl = "%02d-{}-{}.png" % number
        dom = np.linspace(-1, 13, 500)

        demo = demo_t(dom, 5)

        homs_lognorm = demo.get_homs("l")
        idx_of_5sp = 3
        idx_of_3sp = idx_of_5sp - 1
        mask_below = dom < demo.scales[idx_of_3sp]
        idx_of_8sp = idx_of_5sp + 1
        mask_above = dom > demo.scales[idx_of_8sp]
        mask_inter = 1 - (mask_below + mask_above)
        mask_inter = mask_inter.astype(bool)
        hom_5sp = homs_lognorm[idx_of_5sp]
        total_sum = hom_5sp.sum()

        self.start()

        ratio_below = hom_5sp[mask_below].sum() / total_sum
        ratio_above = hom_5sp[mask_above].sum() / total_sum
        ratio_inter = hom_5sp[mask_inter].sum() / total_sum
        self.ax.fill_between(dom[mask_below], 0, hom_5sp[mask_below], fc="green", alpha=0.6, label=f"Below 3 SP - {100 * ratio_below:.2g}%")
        self.ax.fill_between(dom[mask_inter], 0, hom_5sp[mask_inter], fc="blue", alpha=0.6, label=f"In Between - {100 * ratio_inter:.2g}%")
        self.ax.fill_between(dom[mask_above], 0, hom_5sp[mask_above], fc="red", alpha=0.6, label=f"Above 8 SP - {100 * ratio_above:.2g}%")
        self.store_limits()
        self._groom_axes()

        self.fig.savefig(stem_tpl.format(1, "lognorm_analyzed"), dpi=self.dpi)

        plt.close(self.fig)

    def _gen_third(self, demo_t, number, mode, series_desc):
        stem_tpl = "%02d-{}-{}.png" % number
        dom = np.linspace(-1, 15, 500)

        demo = demo_t(dom, 6)
        homs = demo.get_homs(mode)

        self.start()

        self._plot_unknown_task()
        self.ax.set_ylim(self.ax.get_ylim()[0], 1.4)
        self.store_limits()

        self._groom_axes()
        self.fig.savefig(stem_tpl.format(1, "unknown"), dpi=self.dpi)

        hsum = homs.sum(0)
        coef = np.median(hsum[(dom > 1.5) * (dom < 8)])
        homs /= coef
        old = np.zeros_like(demo.dom)
        for ii, hom in enumerate(homs):

            self._plot_increment(dom, old, hom, demo.scales[ii])
            if ii > 2:
                self.ax.plot(dom, homs[3], "--", color="black", label="5 SP")
            self._groom_axes()

            self.fig.savefig(stem_tpl.format(2 + ii, series_desc), dpi=self.dpi)

            old += hom

        plt.close(self.fig)

    def _convolve_and_crop(self, dom, hom1, hom2):
        dom3, hom3 = estimage.utilities.eco_convolve(dom, hom1, dom, hom2)
        hom = sp.interpolate.interp1d(dom3, hom3, kind="linear", bounds_error=False, fill_value=(0, 0))(dom)
        return hom

    def _plot_steps(self, dom, hom, rolling_hom, rng):
        self.ax.clear()
        self.ax.fill_between(dom, 0, hom, color="orange", label="8 SP")
        for ii in range(rng):
            self.ax.fill_between(dom, 0, rolling_hom[ii], color="blue", label=f"{ii + 1} x 2 SP")
        # gauss_hom = sp.stats.norm(loc=scale8 / 4 * 5, scale=produce_estimate_from_lognorm(scale2, demo.GOOD_LOGNORM_SHAPE).sigma * np.sqrt(rng)).pdf(dom)
        # self.ax.plot(dom, gauss_hom / gauss_hom.sum() * hom_two.sum(), "--", color="red", label="gauss")
        self._groom_axes()

    def gen_fourth(self):
        stem_tpl = "08-{}-{}.png"
        dom = np.linspace(-1, 18, 500)

        demo = demos.RegDemo(dom, 5)
        homs = demo.get_homs("l")

        self.start()

        scale8 = demo._get_scales()[-1]
        scale2 = scale8 / 4.0
        hom_two = elements.LognormElements(dom, [scale2], demo).compute_elements()[0]
        hom8 = homs[-1]
        rolling_hom = np.zeros((5, len(dom)))
        rolling_hom[0, :] = hom_two
        for ii in range(1, 5):
            rolling_hom[ii] = self._convolve_and_crop(dom, rolling_hom[0], rolling_hom[ii - 1])
            rolling_hom[ii] /= rolling_hom[ii].sum()
        rolling_hom[0] /= rolling_hom[0].sum()
        rolling_hom *= hom8.sum()

        self.ax.fill_between(dom, 0, homs[-1], color="orange", label="8 SP")
        self.ax.fill_between(dom, 0, rolling_hom[0], color="blue", label="1 x 2 SP")
        self.ax.plot(dom, rolling_hom[3], "--", color="blue", label="4 x 2 SP")
        print((rolling_hom[3] * dom).sum())
        print((homs[-1] * dom).sum())
        self.store_limits()
        self._groom_axes()
        self.fig.savefig(stem_tpl.format(1, "eight_and_two"), dpi=self.dpi)

        self._plot_steps(dom, homs[-1], rolling_hom, 2)
        self.fig.savefig(stem_tpl.format(2, "and_four"), dpi=self.dpi)

        self._plot_steps(dom, homs[-1], rolling_hom, 3)
        self.fig.savefig(stem_tpl.format(3, "and_six"), dpi=self.dpi)

        self._plot_steps(dom, homs[-1], rolling_hom, 4)
        self.fig.savefig(stem_tpl.format(4, "and_eight"), dpi=self.dpi)

        self._plot_steps(dom, homs[-1], rolling_hom, 5)
        self.fig.savefig(stem_tpl.format(5, "and_ten"), dpi=self.dpi)

        plt.close(self.fig)

    def gen_fifth(self):
        # Ideal Estimator, composition of estimates
        stem_tpl = "08-{}-{}.png"
        dom = np.linspace(-1, 19, 501)

        num_scales = 6
        demo = demos.RegDemo(dom, num_scales)
        demo.GOOD_LOGNORM_SHAPE = 0.26
        homs = demo.get_homs("l")

        sample = 5
        sample_index = np.argmin(np.abs(dom - sample))

        self.start()
        base_color = np.array((0.8, 0.25, 0))
        color_inc = np.array((-0.5 / demo.size, 0.7 / demo.size, 0.85 / demo.size))
        ratios = np.zeros(demo.size)
        for idx in range(demo.size):
            norming = demo.scales[idx]
            norming = 1
            ratios[idx] = homs[idx, sample_index] / demo.scales[idx] / norming
            print(f"{idx}: {sum(homs[idx]) / demo.scales[idx]:.2g}")
        ratios /= ratios.sum()
        for ii, scale in enumerate(demo.scales):
            print(f"{scale:.2g}: {ratios[ii] * 100:.2g}%")

        print(f"{sample:.2g} -> {sum(ratios * demo.scales)}")

        cumsum = np.zeros_like(dom)
        for idx, scale in enumerate(demo.scales):
            increment = homs[idx] / scale
            self.ax.fill_between(dom, cumsum, cumsum + increment, fc=base_color + color_inc * idx, label=f"{demo.scales[idx]:.2g}")
            cumsum += increment
        self._groom_axes()
        self.fig.savefig(stem_tpl.format(6, f"composition_of_{sample}"), dpi=self.dpi)

        self.ax.clear()
        for idx, scale in enumerate(demo.scales):
            self.ax.plot(dom, homs[idx] / scale, color=base_color + color_inc * idx, label=f"{demo.scales[idx]:.2g}")
        sample_scale_index = 3
        between_sample_and_neighbor = (demo.scales[sample_scale_index] + demo.scales[sample_scale_index - 1]) / 2.0
        self.ax.axvline(between_sample_and_neighbor, color="gray", ls="--", label=f"{between_sample_and_neighbor:.2g}")
        self._groom_axes()
        self.fig.savefig(stem_tpl.format(6, f"composition_of_{sample}-2"), dpi=self.dpi)

        self.ax.clear()
        homs_normed = homs.copy() / demo.scales[:, np.newaxis] ** 2
        homs_sums = np.sum(homs_normed, 0)
        homs_normed /= homs_sums[np.newaxis]
        cumsum = np.zeros_like(homs_normed[0])
        for idx, scale in enumerate(demo.scales):
            self.ax.fill_between(dom, cumsum, cumsum + homs_normed[idx], fc=base_color + color_inc * idx, label=f"{demo.scales[idx]:.2g}")
            cumsum += homs_normed[idx]
        self.ax.axvline(between_sample_and_neighbor, color="gray", ls="--", label=f"{between_sample_and_neighbor:.2g}")
        self._groom_axes()
        self.fig.savefig(stem_tpl.format(6, f"composition_of_{sample}-x"), dpi=self.dpi)

        self.ax.clear()
        expected_estimate_values = np.zeros_like(dom)
        nonzero_mask = sum(homs, 0) > 0
        norming = demo.scales.reshape(num_scales, 1)
        # norming = 1
        nonzero_homs = homs.copy()[:, nonzero_mask] / demo.scales.reshape(num_scales, 1) / norming
        nonzero_homs /= sum(nonzero_homs, 0)
        expected_estimate_values[nonzero_mask] = sum(demo.scales.reshape(num_scales, 1) * nonzero_homs, 0)

        # TODO: for a set of Dom values, calculate expected value (we know that already) and stdev (calculate using custom discrete distribution)

        self.ax.plot(dom[nonzero_mask], expected_estimate_values[nonzero_mask], color="blue", label=f"Expected SPs")
        self.ax.plot(dom[nonzero_mask], dom[nonzero_mask], color="black", label=f"SPs")
        self.ax.grid()
        self.ax.set_xlabel("size / SP")
        self.ax.set_ylabel("size / SP")
        self.fig.savefig(stem_tpl.format(6, f"composition_of_{sample}-3"), dpi=self.dpi)
        plt.close(self.fig)

    def gen_sixth(self):
        stem_tpl = "09-{}-{}.png"
        dom = np.linspace(-1, 19, 501)

        GOOD_LOGNORM_SHAPE = 0.26

        sample = 5
        dist = utils.get_lognorm_dist(scale=sample, shape=GOOD_LOGNORM_SHAPE)
        mean = dist.mean()
        var = dist.var()
        normed_scale = utils.norm_lognorm_scale(sample, GOOD_LOGNORM_SHAPE)
        mode = np.exp(np.log(normed_scale) - GOOD_LOGNORM_SHAPE**2)

        self.start()
        for pert_lambda in [2, 4, 8, 16, 32, 64]:
            o, p = estimate.calculate_o_p_ext(mode, mean, var, pert_lambda)
            pert = estimate.Estimate.from_triple(mode, o, p, pert_lambda)
            self.ax.plot(dom, dist.pdf(dom), "--", color="black", label=f"lognorm")
            self.ax.fill_between(dom, 0, pert.get_pert(dom=dom)[1], fc="blue", alpha=0.5, label=f"pert {pert_lambda}")
            self.ax.axvline(mode, color="springgreen")
            self.ax.grid()
            self.ax.set_xlabel("size / SP")
            self.ax.set_ylabel("Probability Density")
            self.ax.legend()
            self.fig.savefig(stem_tpl.format(1, f"lognorm_PERT-{pert_lambda:02d}"), dpi=self.dpi)
            self.ax.clear()
        plt.close(self.fig)

    def gen_seventh(self):
        stem_tpl = "10-{}-{}.png"
        self.start()

        src = 0.5

        self._compare_estimates(5, 3, 8, src)
        self.fig.savefig(stem_tpl.format(1, f"lognorm-like"), dpi=self.dpi)
        self.ax.clear()

        self._compare_estimates(5, 2, 8, src)
        self.fig.savefig(stem_tpl.format(2, f"gauss-like"), dpi=self.dpi)
        self.ax.clear()

        self._compare_estimates(5, 4, 9, src)
        self.fig.savefig(stem_tpl.format(3, f"skewed"), dpi=self.dpi)
        self.ax.clear()

        plt.close(self.fig)

    def _compare_estimates(self, m, o, p, l0, l1=64):
        e1 = estimate.Estimate.from_triple(m, o, p, l0)
        o2, p2 = estimate.calculate_o_p_ext(m, e1.expected, e1.variance, l1)
        e2 = estimate.Estimate.from_triple(m, o2, p2, l1)
        dist = e2._get_rv()

        dom = np.linspace(dist.ppf(0.001) - 1, dist.ppf(0.999) + 1, 500)
        hom_orig = e1.get_pert(dom=dom)[1]
        hom_2 = e2.get_pert(dom=dom)[1]

        ratio_below = dist.cdf(o)
        ratio_above = 1 - dist.cdf(p)
        ratio_inter = 1 - ratio_below - ratio_above
        mask_below = dom < o
        mask_inter = (dom >= o) * (dom <= p)
        mask_above = dom > p
        self.ax.plot(dom, hom_orig, "--", color="black", alpha=0.6, label=f"Original setting")
        self.ax.fill_between(dom[mask_below], 0, hom_2[mask_below], fc="green", alpha=0.6, label=f"Below - {100 * ratio_below:.2g}%")
        self.ax.fill_between(dom[mask_inter], 0, hom_2[mask_inter], fc="blue", alpha=0.6, label=f"In Between - {100 * ratio_inter:.2g}%")
        self.ax.fill_between(dom[mask_above], 0, hom_2[mask_above], fc="red", alpha=0.6, label=f"Above - {100 * ratio_above:.2g}%")
        self.store_limits()
        self._groom_axes()

        print(f"{o=} {o2=} {p=} {p2=} {e1.expected} {e2.expected}")


dom = np.linspace(0, 20, 2000)

demof = demos.FibDemo(dom)
demor = demos.RegDemo(dom)

plotter = Plotter()
# plotter.gen_first()
# plotter.gen_second()
# plotter.gen_third()
# plotter.gen_fourth()
plotter.gen_fifth()
# plotter.gen_sixth()
# plotter.gen_seventh()

# TODO: How to estimate a 4SP or 5SP task?
