# standard library
import math


# dependent packages
import numpy as np
import xarray as xr


def decide_z(
    maskrange: tuple,
    spec: xr.DataArray,
    rest_freq: float,
) -> tuple:
    mask = (maskrange[0] < spec["ch"]) & (spec["ch"] < maskrange[1])
    nume = spec * spec["ch"]
    masked_n = nume[mask]
    weight = np.sum(masked_n)
    deno = np.sum(spec[mask])
    obs_freq = float(weight / deno)
    redshift = float(rest_freq / obs_freq - 1)
    return redshift, obs_freq


def get_integ_int(
    obs_freq: float,
    spec: xr.DataArray,
    chbin: int,
    maskrange: tuple,
) -> tuple:
    lam = 1 / (3.34 * spec.ch)
    beamsz = (lam / 50) * (180 / math.pi) * 3600 * 1.2
    idx = (4 * math.pi * 1.33 * 10 ** (-4)) * (3.34 * obs_freq)
    a_eff = 0.80 * math.exp(-(idx ** 2))
    mask = (maskrange[0] < spec["ch"]) & (spec["ch"] < maskrange[1])
    ch_wid = (
        (2.99 * 10 ** 5) * ((2.5 * 10 ** 9) / 2 ** (15) * chbin) / (obs_freq * 10 ** 9)
    )
    f_nu = (spec / a_eff) * beamsz ** 2 * spec.ch ** 2 / (1.22 * 10 ** 6)
    integ = ch_wid * f_nu
    masked_integ = integ[mask]
    integ_summed = np.sum(masked_integ)

    ch_use = ~(mask)
    rms_f = f_nu[ch_use]
    noise_f = rms_f.std("ch")
    integ_noise = ch_wid * np.array(noise_f)
    sum_noise = integ_noise * math.sqrt(len(masked_integ))

    return integ_summed, sum_noise


def Dlpen(redshift, giveAnswerInMeters=False):
    from astropy.cosmology import Planck15 as cosmo

    Mpc = 3.0857e22
    Dl = cosmo.luminosity_distance(redshift).value
    if giveAnswerInMeters:
        return Dl * Mpc
    else:
        return Dl


def get_luminosity(
    redshift: float,
    integ_summed: float,
    sum_noise: float,
    obs_freq: float,
) -> tuple:
    D_L = Dlpen(redshift)
    L_co = 1.04 * 10 ** (-3) * integ_summed * obs_freq * D_L ** 2
    L_co_er = 1.04 * 10 ** (-3) * sum_noise * obs_freq * D_L ** 2

    return L_co, L_co_er


def get_prime_luminosity(
    redshift: float,
    integ_summed: float,
    sum_noise: float,
    obs_freq: float,
) -> tuple:
    D_L = Dlpen(redshift)
    L_co_prime = (
        3.25
        * 10 ** 7
        * integ_summed
        * obs_freq ** (-2)
        * D_L ** 2
        * (1 + redshift) ** (-3)
    )
    L_co_prime_er = (
        3.25
        * 10 ** 7
        * sum_noise
        * obs_freq ** (-2)
        * D_L ** 2
        * (1 + redshift) ** (-3)
    )

    return L_co_prime, L_co_prime_er


def convert_CI_to_molecular_mass(
    redshift: float,
    integ_summed: float,
    sum_noise: float,
) -> tuple:
    D_L = Dlpen(redshift)
    M_h2 = (
        (1376 / (redshift + 1))
        * D_L ** 2
        * (1 / 3)
        * (1 / 0.793)
        * (1 / 0.5)
        * (integ_summed)
    )
    M_h2_er = (
        (1376 / (redshift + 1))
        * D_L ** 2
        * (1 / 3)
        * (1 / 0.793)
        * (1 / 0.5)
        * (sum_noise)
    )

    return M_h2, M_h2_er
