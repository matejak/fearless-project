import numpy as np
import scipy as sp

import pylab as pyl
import matplotlib.pyplot as plt


def make_rect_payoff(lead_time, fade_away_time):
    def fun(dom):
        shifted_dom = dom - lead_time
        ret = np.ones_like(shifted_dom, dtype=float)
        ret[shifted_dom < 0] *= 0
        ret[shifted_dom > fade_away_time] *= 0
        return ret
    return fun


def make_trianular_payoff(lead_time, fade_away_time, peak_ratio=0.15):
    scale = fade_away_time - lead_time
    fun = sp.stats.triang(loc=lead_time, scale=scale, c=peak_ratio)
    return fun.pdf


def make_attenuation(dom, rate):
    ret = rate ** dom
    ret[dom < 0] = 1
    return 1.0 / ret


def make_organic_payoff(lead_time, raise_rate, die_rate):
    def fun(dom):
        shifted_dom = dom - lead_time
        mask = shifted_dom > 0
        relevant_dom = shifted_dom[mask]
        ret = np.zeros_like(dom, dtype=float)
        base = 1.0 - 1.0 / (raise_rate ** relevant_dom)
        atten = make_attenuation(relevant_dom, die_rate)
        ret[mask] = base
        ret[mask] *= atten
        return ret
    return fun


def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dom = np.linspace(0, 48, 200)
    fun = make_trianular_payoff(1, 40)
    fun = make_rect_payoff(2, 35)
    fun = make_organic_payoff(2, 1.5, 1.01)

    early = fun(dom)
    late = fun(dom - 3)
    plot_payoff_diff(ax, dom, early, late)
    ax.grid()
    ax.legend()
    plt.show()


def plot_payoff_diff(ax, dom, early, late):
    early_wins_area = early > late
    ax.plot(dom, early, color="black", label="Early execution")
    ax.fill_between(dom[early_wins_area], early[early_wins_area], late[early_wins_area], label="Value lost")
    late_wins_area = early < late
    ax.fill_between(dom[late_wins_area], early[late_wins_area], late[late_wins_area], label="Value gained")
