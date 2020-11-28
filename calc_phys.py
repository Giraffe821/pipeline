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
