import collections

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


def make_attenuation_starting_at(start, dom, rate):
    dom = dom - start
    ret = rate ** dom
    ret[dom < 0] = 1
    return 1.0 / ret


def make_attenuation(dom, rate):
    return make_attenuation_starting_at(0, dom, rate)


def make_degradation(rate):
    return lambda dom: make_attenuation(dom, rate)


def make_organic_payoff(lead_time, raise_rate, die_rate, die_offset=0):
    def fun(dom):
        shifted_dom = dom - lead_time
        mask = shifted_dom > 0
        relevant_dom = shifted_dom[mask]
        ret = np.zeros_like(dom, dtype=float)
        base = 1.0 - 1.0 / (raise_rate ** relevant_dom)
        atten_mask = shifted_dom - die_offset > 0
        atten = make_attenuation(shifted_dom[atten_mask] - die_offset, die_rate)
        ret[mask] = base
        ret[atten_mask] *= atten
        return ret
    return fun


def make_hanning_payoff(lead_time, inertia_time, full_time):
    def fun(dom):
        shifted_dom = dom - lead_time
        mask = shifted_dom > 0
        ret = np.zeros_like(dom, dtype=float)
        raising = mask * (shifted_dom < inertia_time)
        num_r = sum(raising)
        ret[raising] = np.hanning(num_r * 2 + 1)[0:num_r]
        stable = (shifted_dom > inertia_time) * (shifted_dom < inertia_time + full_time)
        ret[stable] = 1
        dropping = (shifted_dom > inertia_time + full_time) * (shifted_dom < 2 * inertia_time + full_time)
        num_r = sum(dropping)
        ret[dropping] = np.hanning(num_r * 2 + 1)[num_r + 1:]
        return ret
    return fun


def plot(callback, delay, general_degradation=None, delay_degradation=1):
    fig = plt.figure()

    dom = np.linspace(0, 300, 500)
    if general_degradation is None:
        general_degradation = lambda x: 1

    early = callback(dom)
    early *= general_degradation(dom)
    late = callback(dom - delay)
    late *= general_degradation(dom) / delay_degradation
    plot_payoff_diff(fig, dom, early, late)
    plt.show()


def plot_tea_blob_ending_synchronously(degradation_coef=1):
    fig = plt.figure()

    dom = np.linspace(0, 300, 500)

    degradation = make_attenuation(dom, degradation_coef)
    early = make_hanning_payoff(3, 40, 120)(dom) * degradation
    late = make_hanning_payoff(23, 40, 100)(dom) * degradation
    plot_payoff_diff_value_time(fig, dom, early, late, late * 0.9)
    plt.show()


def plot_bread_blob_ending_synchronously(degradation_coef=1):
    fig = plt.figure()

    dom = np.linspace(0, 300, 500)

    degradation = make_attenuation(dom, degradation_coef)
    degradation = 1
    early = make_organic_payoff(3, 1.2, 1.04, 90)(dom) * degradation
    late = make_organic_payoff(13, 1.2, 1.04, 80)(dom) * degradation
    plot_payoff_diff_value_time(fig, dom, early, late, late * 0.9)
    plt.show()


def plot_tea_rect():
    fun = make_rect_payoff(2, 20)
    plot(fun, 3)


def plot_bread_rect():
    fun = make_rect_payoff(2, 20)
    plot(fun, 3, general_degradation=make_degradation(1.005), delay_degradation=1.1)


def plot_tea_rect_degraded():
    fun = make_rect_payoff(2, 35)
    plot(fun, 3, make_degradation(1.005))


def plot_tea_organic():
    fun = make_organic_payoff(2, 1.5, 1.01)
    plot(fun, 3)


def plot_tea_organic_degrading():
    fun = make_organic_payoff(2, 1.5, 1.01)
    degradation = lambda dom: make_attenuation(dom, 1.01)
    plot(fun, 3, degradation)


vis = collections.namedtuple("vis", ("color", "quantity"))


def plot_payoff_diff_value_time(fig, dom, early, late, late_nontea):
    early_wins_area = early > late_nontea
    ax_values = fig.add_subplot(211)
    ax_values.plot(dom, early, color="black", label="Early execution")
    ax_values.plot(dom, late_nontea, "--", color="black", label="Late execution")

    ax_values.fill_between(dom[early_wins_area], early[early_wins_area], late[early_wins_area], color="red", alpha=0.5, label="Lateness")
    ax_values.fill_between(dom[early_wins_area], late[early_wins_area], late_nontea[early_wins_area], color="orange", alpha=0.5, label="Degradation")

    ax_values.grid()
    ax_values.legend()

    ax_summary = fig.add_subplot(212)
    stuff = dict()

    lost_late = (early - late)[early_wins_area].sum()
    lost_degraded = (late - late_nontea)[early_wins_area].sum()
    stuff["late"] = vis(color="red", quantity=lost_late)
    stuff["degraded"] = vis(color="orange", quantity=lost_degraded)
    stuff["CoD"] = vis(color="black", quantity=lost_late + lost_degraded)
    for key, val in stuff.items():
        ax_summary.bar(key, val.quantity, color=val.color)

    ax_summary.grid()


def plot_payoff_diff(fig, dom, early, late):
    ax_values_zoomed = fig.add_subplot(311)
    early_wins_area = early > late
    zoom_dst = 20
    beginning = dom < zoom_dst
    ax_values_zoomed.plot(dom[beginning], early[beginning], color="black", label="Early execution")
    ax_values_zoomed.plot(dom[beginning], late[beginning], "--", color="black", label="Late execution")
    ax_values_zoomed.fill_between(dom[early_wins_area * beginning], early[early_wins_area * beginning], late[early_wins_area * beginning], color="red", alpha=0.5, label="Value lost")
    late_wins_area = early < late
    ax_values_zoomed.fill_between(dom[late_wins_area * beginning], early[late_wins_area * beginning], late[late_wins_area * beginning], color="green", alpha=0.5, label="Value gained")
    ax_values_zoomed.grid()
    ax_values_zoomed.legend()

    ax_values = fig.add_subplot(312)
    ax_values.plot(dom, early, color="black", label="Early execution")
    ax_values.fill_between(dom[early_wins_area], early[early_wins_area], late[early_wins_area], color="red", alpha=0.5, label="Value lost")
    late_wins_area = early < late
    ax_values.fill_between(dom[late_wins_area], early[late_wins_area], late[late_wins_area], color="green", alpha=0.5, label="Value gained")
    ax_values.grid()
    ax_values.legend()

    ax_summary = fig.add_subplot(313)
    stuff = dict()

    gained = (late - early)[late_wins_area].sum()
    lost = (early - late)[early_wins_area].sum()
    stuff["gained"] = vis(color="green", quantity=gained)
    stuff["lost"] = vis(color="red", quantity=lost)
    stuff["CoD"] = vis(color="black", quantity=-(gained - lost))
    for key, val in stuff.items():
        ax_summary.bar(key, val.quantity, color=val.color)

    ax_summary.grid()
