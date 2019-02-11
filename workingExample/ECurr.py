# Tutorial 2.7. Spin textures
# ===========================
#
# Physics background
# ------------------
#  - Spin textures
#  - Skyrmions
#
# Kwant features highlighted
# --------------------------
#  - operators
#  - plotting vector fields

from math import sin, cos, tanh, pi
import itertools
import numpy as np
import tinyarray as ta
import matplotlib.pyplot as plt
import time

import kwant
import generalOperator
import generalOperatorStringsNotCalledNoPrints

import pstats, cProfile

sigma_0 = ta.array([[1, 0], [0, 1]])
sigma_x = ta.array([[0, 1], [1, 0]])
sigma_y = ta.array([[0, -1j], [1j, 0]])
sigma_z = ta.array([[1, 0], [0, -1]])

# vector of Pauli matrices σ_αiβ where greek
# letters denote spinor indices
sigma = np.rollaxis(np.array([sigma_x, sigma_y, sigma_z]), 1)



def field_direction(pos, r0, delta):
    x, y = pos
    r = np.linalg.norm(pos)
    r_tilde = (r - r0) / delta
    theta = (tanh(r_tilde) - 1) * (pi / 2)

    if r == 0:
        m_i = [0, 0, -1]
    else:
        m_i = [
            (x / r) * sin(theta),
            (y / r) * sin(theta),
            cos(theta),
        ]

    return np.array(m_i)


def scattering_onsite(site, r0, delta, J):
    m_i = field_direction(site.pos, r0, delta)
    return J * np.dot(m_i, sigma)


def lead_onsite(site, J):
    return J * sigma_z


lat = kwant.lattice.square(norbs=2)

def make_system(L=80):

    syst = kwant.Builder()

    def square(pos):
        return all(-L/2 < p < L/2 for p in pos)

    syst[lat.shape(square, (0, 0))] = scattering_onsite
    syst[lat.neighbors()] = -sigma_0

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)),
                         conservation_law=-sigma_z)

    lead[lat.shape(square, (0, 0))] = lead_onsite
    lead[lat.neighbors()] = -sigma_0

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst


def plot_vector_field(syst, params):
    xmin, ymin = min(s.tag for s in syst.sites)
    xmax, ymax = max(s.tag for s in syst.sites)
    x, y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    m_i = [field_direction(p, **params) for p in zip(x.flat, y.flat)]
    m_i = np.reshape(m_i, x.shape + (3,))
    m_i = np.rollaxis(m_i, 2, 0)

    fig, ax = plt.subplots(1, 1)
    im = ax.quiver(x, y, *m_i, pivot='mid', scale=75)
    fig.colorbar(im)
    plt.show()


def plot_densities(syst, densities):
    fig, axes = plt.subplots(1, len(densities))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.map(syst, rho, ax=ax, a=4)
        ax.set_title(title)
    plt.show()


def plot_currents(syst, currents):
    fig, axes = plt.subplots(1, len(currents))
    if not hasattr(axes, '__len__'):
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)
    plt.show()


def main():

    for i in [150,300,400,500,600,800,1000,1500]:
        print('----------------------------------------')
        sys_size=i
        start_time = time.time()
        syst = make_system(sys_size).finalized()
        print("------- system creation: %s seconds ---" % (time.time() - start_time))

        params = dict(r0=20, delta=10, J=1)
        start_time = time.time()
        wf = kwant.wave_function(syst, energy=-1, params=params)
        print("------- wave functions calculation: %s seconds ---" % (time.time() - start_time))
        psi = wf(0)[0]
        size = (syst.cell_size if isinstance(syst, kwant.system.InfiniteSystem) else syst.graph.num_nodes)
#        print(size*2, " ", (time.time() - start_time), file=open("sys_creat-times-OpStrings.dat", "a"))
        psi = np.zeros(size*2, dtype=complex)
        start_time = time.time()
        offECurr = generalOperator.offEnergyCurrentLead(syst, where=[None,None])
#        offECurr = generalOperatorStringsNotCalledNoPrints.offEnergyCurrentLead(syst, where=[None,None])
        print("------- operator initialisation: %s seconds ---" % (time.time() - start_time))
#        print(size*2, " ", (time.time() - start_time), file=open("init-times-OpStrings.dat", "a"))

#        # calculate the expectation values of the operators with 'psi'
        start_time = time.time()
        ecurr = offECurr(psi)
        print("------- operator calculation: %s seconds ---" % (time.time() - start_time))
#        print(size*2, " ", (time.time() - start_time), file=open("call-times-OpStrings.dat", "a"))
        print(ecurr)



if __name__ == '__main__':
    main()
