# import pytest

import h5py
import pandas as pd
import numpy as np

df = pd.read_csv("tests/stark_map_25D_MHz.csv")
Ef = np.array(df["E-field (V/cm)"])
f_shifts = np.array(df["25D_{5/2}:5/2"])

from ..lineshapes import HoltsmarkLine
from ..lineshapes import Poly, _fast_lorentzian_sum

hl = HoltsmarkLine(Ef, f_shifts)

Np = 300
freq = np.linspace(-1200, 400, Np)
Efg = np.linspace(0, 45, 30)
Lwg = np.linspace(20, 50, 10)
Ehg = np.linspace(0.01, 20, 20)
hl.build_lut(freq, Efg, Ehg, Lwg)

Ef0 = 20
Lw = 20
Eh = 20

pol = Poly(np.arange(7))  # 7th order polynomial

# warm up numba jit compiler
hl.stark_map.Efield2freq(np.linspace(0, max(Ef), 1000))
pol.polyval(Ef)
_fast_lorentzian_sum(freq, np.linspace(0, 1, 10000), np.linspace(0, 2, 10000), 10, 2)
hl.line2d(freq, Ef0, Lw, Eh)


def test_fast_lorentzian_sum(benchmark):
    result = benchmark(
        _fast_lorentzian_sum,
        freq,
        np.linspace(0, 1, 10000),
        np.linspace(0, 2, 10000),
        10,
        2,
    )


def test_polyval(benchmark):
    result = benchmark(pol.polyval, np.linspace(0, max(Ef), 1000))


def test_stark_map_Efield2freq_within_tabulated(benchmark):
    result = benchmark(hl.stark_map.Efield2freq, np.linspace(0, max(Ef), 1000))


def test_stark_map_Efield2freq_outside_tabulated(benchmark):
    result = benchmark(
        hl.stark_map.Efield2freq, np.linspace(max(Ef) + 1, 2 * (max(Ef) + 1), 1000)
    )


def test_build_lut(benchmark):
    result = benchmark(hl.build_lut, freq, Efg, Ehg, Lwg)


def test_line2d(benchmark):
    result = benchmark(hl.line2d, freq, Ef0, Lw, Eh)


def test_line2d_lut(benchmark):
    result = benchmark(hl.line_lut, freq, Ef0, Lw, Eh)

def test_spectal_map_with_lut(benchmark):
    result = benchmark(hl.line_lut, freq, np.linspace(0,49), Lw, Eh/10)

def test_line2d_fidelity():
    f_shifts = np.array(df["25D_{5/2}:5/2"])

    hl = HoltsmarkLine(Ef, f_shifts)
    lsh_bf = hl.line2d(freq, 60.0, 20.0, 0.02, bruteforce=True)
    lsh_fast = hl.line2d(freq, 60.0, 20.0, 0.02)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 60.0, 20.0, 20, bruteforce=True)
    lsh_fast = hl.line2d(freq, 60.0, 20.0, 20)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 10.0, 20.0, 20, bruteforce=True)
    lsh_fast = hl.line2d(freq, 10.0, 20.0, 20)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 10.0, 20.0, .02, bruteforce=True)
    lsh_fast = hl.line2d(freq, 10.0, 20.0, 0.02)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    f_shifts = np.array(df["25D_{5/2}:1/2"])

    hl = HoltsmarkLine(Ef, f_shifts)
    lsh_bf = hl.line2d(freq, 60.0, 20.0, 0.02, bruteforce=True)
    lsh_fast = hl.line2d(freq, 60.0, 20.0, 0.02)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 60.0, 20.0, 20, bruteforce=True)
    lsh_fast = hl.line2d(freq, 60.0, 20.0, 20)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 20.0, 20.0, 20, bruteforce=True)
    lsh_fast = hl.line2d(freq, 20.0, 20.0, 20)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

    lsh_bf = hl.line2d(freq, 20.0, 20.0, .02, bruteforce=True)
    lsh_fast = hl.line2d(freq, 20.0, 20.0, 0.02)
    assert (np.abs((lsh_bf-lsh_fast)/lsh_bf).max() <= 0.001) | (np.abs(lsh_bf-lsh_fast).max() <= 0.001)

