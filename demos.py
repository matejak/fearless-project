import numpy as np

import elements


def cover_area(functions, sl=None):
    coverage = np.empty_like(functions)
    if not sl:
        sl = slice(0, None)
    selection = np.array([f[sl] for f in functions])
    qr = np.linalg.qr(selection.T)
    bases = qr.Q.T
    how = qr.R
    target = np.ones_like(bases[0])
    coefs = np.zeros(len(bases))
    for ii, base in enumerate(bases):
        coefs[ii] = np.dot(base, target)
    coefs @= np.linalg.inv(how).T
    for ii, base in enumerate(bases):
        coverage[ii] = coefs[ii] * functions[ii]
    return coverage


class Demo:
    GOOD_LOGNORM_SHAPE = 0.27
    GOOD_GAUSS_SHAPE = 0.27 * 5
    GOOD_GAUSS_SHAPE = 0.27 * 5
    COLOR_CAROUSEL = ("lightsteelblue", "cornflowerblue", "royalblue", "dodgerblue")
    COLOR_CAROUSEL2 = ("springgreen", "limegreen", "seagreen", "lawngreen")

    def __init__(self, dom, size=7):
        self.dom = dom
        self.size = size

    def _restrict_dom(self):
        scales = self._get_scales()
        meaningful_values = self.dom > scales[0] * 0.5
        meaningful_values * self.dom < scales[-1] * 1.2
        self.dom = self.dom[meaningful_values]

    @property
    def scales(self):
        return self._get_scales()

    def get_homs(self, mode):
        factory = elements.ElementsFactory(self.dom, self.scales, self)
        homs = factory.get_elements(mode)
        #homs /= homs.sum(0).max()
        return homs

    def get_cover_homs(self, mode, start, end):
        homs = self.get_homs(mode)

        cover_homs = cover_area(homs, get_slice(self.dom, start, end))
        return cover_homs

    def show_layers(self, mode, limits=None):
        fig = pyl.figure()
        ax = fig.add_subplot(111)
        if limits:
            start, end = limits
            homs = self.get_cover_homs(mode, start, end)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_betweenx((0, 1.0), start, end, transform=trans, color="orange", alpha=0.3)
        else:
            homs = self.get_homs(mode)
        base = np.zeros_like(homs[0])
        scales = self.scales
        for ii, hom in enumerate(homs):
            color_index = ii % len(self.COLOR_CAROUSEL)
            ax.fill_between(self.dom, base, base + hom, fc=self.COLOR_CAROUSEL[color_index], label=f"{scales[ii]:.2g}")
            ax.axvline(scales[ii], ls="--", color="k")
            base += hom
        ax.grid()
        ax.legend(loc="upper right")
        pyl.show()

    def show_homs(self, mode, limits=None):
        fig = pyl.figure()
        ax = fig.add_subplot(111)
        if limits:
            start, end = limits
            homs = self.get_cover_homs(mode, start, end)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_betweenx((0, 1.0), start, end, transform=trans, color="orange", alpha=0.3)
        else:
            homs = self.get_homs(mode)
        scales = self.scales
        mean = np.zeros_like(homs[0])
        for ii, hom in enumerate(homs):
            color_index = ii % len(self.COLOR_CAROUSEL)
            scale = scales[ii]
            ax.plot(self.dom, hom, color=self.COLOR_CAROUSEL[color_index], label=f"{scale:.2g}")
            ax.axvline(scales[ii], ls="--", color="k")
            mean += scale * hom
        ax.plot(self.dom, homs.sum(0), ls="--", color="orchid", label="coverage")
        ax.plot(self.dom, mean / self.dom, ls="-", color="orange", label="mean")
        ax.grid()
        ax.legend(loc="upper right")
        pyl.show()

    def show_comparison(self, mode1, mode2, limits=None):
        fig = pyl.figure()
        ax = fig.add_subplot(111)
        if limits:
            start, end = limits
            homs1 = self.get_cover_homs(mode1, start, end)
            homs2 = self.get_cover_homs(mode2, start, end)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_betweenx((0, 1.0), start, end, transform=trans, color="orange", alpha=0.3)
        else:
            homs1 = self.get_homs(mode1)
            homs2 = self.get_homs(mode2)
        scales = self.scales
        for ii, hom in enumerate(homs1):
            color_index = ii % len(self.COLOR_CAROUSEL)
            ax.plot(self.dom, hom, color=self.COLOR_CAROUSEL[color_index], label=f"{scales[ii]:.2g}")
            ax.plot(self.dom, -homs2[ii], color=self.COLOR_CAROUSEL2[color_index])
            ax.axvline(scales[ii], ls="--", color="k")
        ax.plot(self.dom, homs1.sum(0), ls="--", color="orchid", label="coverage")
        ax.plot(self.dom, -homs2.sum(0), ls="--", color="goldenrod")
        ax.grid()
        ax.legend(loc="upper right")
        pyl.show()


class FibDemo(Demo):
    def _get_scales(self):
        return np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])[:self.size]


class RegDemo(Demo):
    def _get_scales(self):
        natural_scales = (np.ones(self.size) * 1.618) ** np.arange(self.size)
        return natural_scales * 5 / natural_scales[3]


class Reg2Demo(Demo):
    def _get_scales(self):
        natural_scales = (np.ones(self.size) * 2) ** np.arange(self.size)
        return natural_scales * 5 / natural_scales[3]


class EqDemo(Demo):
    def _get_scales(self):
        natural_scales = np.arange(self.size) + 1
        return natural_scales * 5 / natural_scales[3]
