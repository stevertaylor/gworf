#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""

GWORF.py -- (G)ravitational (W)ave (O)verlap (R)eduction (F)unctions

Copyright (c) 2020 Stephen R. Taylor & Rutger van Haasteren

"""

from __future__ import division

import numpy as np
import scipy.linalg as sl, scipy.special as ss
import os, glob, sys, json
import healpy as hp

pic_c = 299792458     # Speed of light in m/s
pic_pc = 3.08567758e16  # Parsec in meters

class psr(object):
    """
    Lightweight class to hold pulsar meta-data.
    """

    def __init__(self, raj, decj, dist):

        self.raj = raj
        self.decj = decj
        self.dist = dist


def R_SignalResponse(ptapsrs, lmax):
    """
    Given a maximum harmonic resolution (lmax), calculate
    the Signal response matrix. No pixel midway conversion
    
    @param lmax:    Mapping resolution
    """
    # R is an (Npsr x Nl) matrix
    Nalm = nClm_sw_lmax(lmax)
    Npsr = len(ptapsrs)
    R = np.zeros((Npsr, Nalm))

    for pp, psr in enumerate(ptapsrs):
        alm_ind = 0
        for ll in range(2,lmax+1):
            for mm in range(-ll, ll+1):
                Nl = shar.Nl(ll)
                
                #coeff = 2*np.pi * shar.Nl(ll) * (-1)**ll
                coeff = np.sqrt(3*np.pi) * shar.Nl(ll) * (-1)**ll
                phi, theta = float(psr.raj), float(np.pi/2-psr.decj)
                
                R[pp, alm_ind] = coeff * shar.real_sph_harm(mm, ll, phi, theta)
                alm_ind += 1
    return R


def R_SignalResponse_earth(ptapsrs, lmax):
    """
    Given a maximum harmonic resolution (lmax), calculate
    the earth-term signal response matrix. No pixel midway conversion
    
    @param lmax:    Mapping resolution
    """
    # R is an (Npsr x Nl) matrix
    Nalm = nClm_sw_lmax(lmax)
    Npsr = len(ptapsrs)
    R = np.zeros((Npsr, Nalm), dtype=np.complex64)

    for pp, psr in enumerate(ptapsrs):
        alm_ind = 0
        for ll in range(2,lmax+1):
            for mm in range(-ll, ll+1):
                Nl = shar.Nl(ll)
                
                #coeff = 2*np.pi * shar.Nl(ll) * (-1)**ll
                coeff = np.sqrt(3*np.pi) * shar.Nl(ll) * (-1)**ll
                phi, theta = float(psr.raj), float(np.pi/2-psr.decj)
                
                R[pp, alm_ind] = coeff * shar.real_sph_harm(mm, ll, phi, theta)
                alm_ind += 1
    return R


def R_SignalResponse_pulsar(ptapsrs, lmax):
    """Since we cannot do the pulsar term properly, we will just add random
    phases"""
    raise NotImplementedError("Cannot do pulsar term justice in sYlm basis :(")
    R = R_SignalResponse_earth(ptapsrs, lmax)
    phase = np.random.rand(*R.shape) * 2 * np.pi

    return R * np.exp(1j * phase)


def signalResponse(ptapsrs, gwtheta, gwphi, freq=1.0e-9, dirconv=True):
    """Total signal response matrix"""
    F_e = signalResponse_earth(ptapsrs, gwtheta, gwphi, freq=freq, dirconv=dirconv)
    F_p = signalResponse_pulsar(ptapsrs, gwtheta, gwphi, freq=freq, dirconv=dirconv)
    return F_e + F_p


def signalResponse_earth(ptapsrs, gwtheta, gwphi, freq=1.0e-9, dirconv=True):
    """
    Create the signal response matrix
    @param dirconv: True when Omega in direction of source (not prop.)
    @param freq:    Frequency in Hertz
    """
    psrpos_phi = np.array([ptapsrs[ii].raj for ii in range(len(ptapsrs))])
    psrpos_theta = np.array([np.pi/2.0 - ptapsrs[ii].decj for ii in range(len(ptapsrs))])
    psrdist = np.array([0.0 for psr in ptapsrs])

    return signalResponse_fast(psrpos_theta, psrpos_phi, gwtheta, gwphi,
            freq, psrdist, dirconv)


def signalResponse_pulsar(ptapsrs, gwtheta, gwphi, freq=1.0e-9, dirconv=True):
    """
    Create the signal response matrix
    @param dirconv: True when Omega in direction of source (not prop.)
    @param psrdist: List of distances to the pulsars (in pc)
    """
    psrpos_phi = np.array([ptapsrs[ii].raj for ii in range(len(ptapsrs))])
    psrpos_theta = np.array([np.pi/2.0 - ptapsrs[ii].decj for ii in range(len(ptapsrs))])
    psrdist = np.array([psr.dist * pic_pc for psr in ptapsrs])

    return signalResponse_fast(psrpos_theta, psrpos_phi, gwtheta, gwphi,
            freq, psrdist, dirconv)


def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a, freq,
        psrdist, dirconv=True):
    """
    Create the signal response matrix FAST
    @param dirconv: True when Omega in direction of source (not prop.)
    """
    npsrs = len(ptheta_a)

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta, freq, psrdist, dirconv=dirconv)


def createSignalResponse(pphi, ptheta, gwphi, gwtheta, freq, psrdist, dirconv=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location
    @param psrdist: Pulsar distances in meters
    @param dirconv: True when Omega in direction of source (not prop.)

    @return:    Signal response matrix of Earth-term

    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, freq, psrdist, plus=True, dirconv=dirconv)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, freq, psrdist, plus=False, dirconv=dirconv)

    F = np.zeros((Fp.shape[0], 2*Fp.shape[1]), dtype=np.complex64)
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F


def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, freq, psrdist, plus=True, norm=True,
        dirconv=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location
    @param freq:    Frequency in Hz
    @param psrdist: Pulsar distances in meters
    @param plus:    Whether or not this is the plus-polarization
    @param dirconv: True when Omega in direction of source (not of propagation)

    @return:    Signal response matrix of Earth-term
    """
    if dirconv:
        dc = 1.0
    else:
        dc = -1.0


    # Create the direction vectors. First dimension will be collapsed later
    Omega = np.array([-np.sin(gwtheta)*np.cos(gwphi), \
                      -np.sin(gwtheta)*np.sin(gwphi), \
                      -np.cos(gwtheta)])
    
    mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
    nhat = np.array([-np.cos(gwphi)*np.cos(gwtheta), \
                     -np.cos(gwtheta)*np.sin(gwphi), \
                     np.sin(gwtheta)])

    p = np.array([np.cos(pphi)*np.sin(ptheta), \
                  np.sin(pphi)*np.sin(ptheta), \
                  np.cos(ptheta)])

    # Extra phase (for pulsar term)
    inprod = np.sum(Omega * p, axis=0)
    phase = 2*np.pi*freq*psrdist[:,None]*(1.0 + inprod) / pic_c
    #if np.sum(psrdist) > 1e-20:
    #    raise NotImplementedError("")
    
    # There is a factor of 3/2 difference between the Hellings & Downs
    # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # et al. 2014). This factor 'normalises' the correlation matrix
    npixels = Omega.shape[2]
    if norm:
        # Add extra factor of 3/2
        c = np.sqrt(1.5) / np.sqrt(npixels)
    else:
        c = 1.0 / np.sqrt(npixels)

    # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
    # al. (2014), with right-handed coordinate system
    if plus:
        # The sum over axis=0 represents an inner-product
        Fsig = 0.5 * c * (np.sum(nhat * p, axis=0)**2 - np.sum(mhat * p, axis=0)**2) / \
                (1 + dc*np.sum(Omega * p, axis=0))
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / \
                (1 + dc*np.sum(Omega * p, axis=0))

    if np.sum(psrdist) > 1e-20:
        Fsig = Fsig * np.exp(1j*phase)

    return Fsig


def fplus_fcross(psr, gwtheta, gwphi):
    """
    Compute gravitational-wave quadrupolar antenna pattern.
    (From NX01)

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]

    :returns: fplus, fcross
    """

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])
    
    # pulsar location
    ptheta = np.pi/2 - psr.decj
    pphi = psr.raj
    
    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
            np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))

    return fplus, fcross


def almFromClm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    maxl = int(np.sqrt(len(clm)))-1
    nclm = len(clm)

    # Construct alm from clm
    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)
            
            clmindex += 1
    
    return alm


def clmFromAlm(alm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    nalm = len(alm)
    maxl = int(np.sqrt(9.0 - 4.0 * (2.0-2.0*nalm))*0.5 - 1.5)
    nclm = (maxl+1)**2

    # Check the solution
    if nalm != int(0.5 * (maxl+1) * (maxl+2)):
        raise ValueError("Check numerical precision. This should not happen")

    clm = np.zeros(nclm)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                #alm[almindex] += clm[clmindex]
                clm[clmindex] = alm[almindex].real
            elif mm < 0:
                #alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
                clm[clmindex] = - alm[almindex].imag * np.sqrt(2)
            elif mm > 0:
                #alm[almindex] += clm[clmindex] / np.sqrt(2)
                clm[clmindex] = alm[almindex].real * np.sqrt(2)
            
            clmindex += 1
    
    return clm


def mapFromClm_fast(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm)))-1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h


def mapFromClm(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels
    """
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    h = np.zeros(npixels)

    ind = 0
    maxl = int(np.sqrt(len(clm)))-1
    for ll in range(maxl+1):
        for mm in range(-ll, ll+1):
            h += clm[ind] * shar.real_sph_harm(mm, ll, pixels[1], pixels[0])
            ind += 1

    return h


def clmFromMap_fast(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values

    Use Healpix spherical harmonics for computational efficiency
    """
    alm = hp.sphtfunc.map2alm(h, lmax=lmax)
    alm[0] = np.sum(h) * np.sqrt(4*np.pi) / len(h)

    return clmFromAlm(alm)


def clmFromMap(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values
    """
    npixels = len(h)
    nside = hp.npix2nside(npixels)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    clm = np.zeros( (lmax+1)**2 )
    
    ind = 0
    for ll in range(lmax+1):
        for mm in range(-ll, ll+1):
            clm[ind] += np.sum(h * shar.real_sph_harm(mm, ll, 
                                                    pixels[1], pixels[0]))
            ind += 1
            
    return clm * 4 * np.pi / npixels


def orfFromMap_fast(usermap, response):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations

    @param usermap:     Provide a healpix map for GW power
    @param response:    Provide the Earth-term pixel-basis response

    Note: GW directions are in direction of GW propagation
    """
    
    F_e = response

    # Double the power (one for each polarization)
    sh = np.array([usermap, usermap]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))


def SH_CorrBasis(psr_locs, lmax, nside=32):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations

    @param psr_locs:    Location of the pulsars [phi, theta]
    @param lmax:        Maximum l to go up to
    @param nside:       What nside to use in the pixelation [32]
    """
    npsrs = len(psr_locs)
    pphi = psr_locs[:,0]
    ptheta = psr_locs[:,1]

    # Create the pixels
    npixels = hp.nside2npix(nside)    # number of pixels total
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]

    # Create the signal response matrix
    F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)

    # Loop over all (l,m)
    basis = []
    nclm = (lmax+1)**2
    clmindex = 0
    for ll in range(0, lmax+1):
        for mm in range(-ll, ll+1):
            clm = np.zeros(nclm)
            clm[clmindex] = 1.0

            basis.append(getCov(clm, nside, F_e))
            clmindex += 1

    return basis


def c_orf_Ylm(pars, ptapsrs, corrbasis):
    """Get c_orf from the parameters for Ylm model
    
    ##########
    :param pars:        Anisotropy parameterization
    :param ptapsrs:     List of pulsar objects
    :param corrbasis:   Basis of ORFs
    
    """
    npsrs = len(ptapsrs)
    nobs = len(ptapsrs[0].residuals)

    Plm = Plm_from_anipars(pars)

    if len(Plm) != len(corrbasis):
        raise ValueError("Dimension mismatch between Plm & corrbasis")

    c_orf = np.zeros((npsrs, npsrs))
    for ii in range(len(Plm)):
        c_orf += Plm[ii] * corrbasis[ii]

    return c_orf


def c_orf_sqrtYlm(pars, ptapsrs, Fe):
    """Get c_orf from the parameters for sqrtYlm model

    ####
    :param Plm:         Anisotropy power parametes P_{lm}
    :param pars:        Anisotropy parameterization
    :param ptapsrs:     List of pulsar objects
    :param F:           The pixel-basis signal response for the Earth term
    
    """
    npsrs = len(ptapsrs)
    nobs = len(ptapsrs[0].residuals)
    nPlm = len(pars)
    npix = int(0.5*Fe.shape[1])
    nside = hp.npix2nside(npix)

    lmax = int(np.sqrt(nPlm))-1
    
    if not (lmax+1)**2 == nPlm:
        raise ValueError("Number of Plm not a full square")

    # Square root of power, since we square it later
    newpars = np.append(1.0,np.copy(pars[1:]))

    hpwr = mapFromClm_fast(newpars, nside)**2.0
    #print '1', hpwr.min(), hpwr.max(), np.sum(hpwr)
    hpwr /= np.mean(hpwr)
    #print '2', hpwr.min(), hpwr.max(), np.sum(hpwr)
    hpwr *= 10.0**(2.0*pars[0])

    return orfFromMap_fast(usermap=hpwr, response=Fe.real)


def c_orf_queryDisk(pars, ptapsrs, Fe, ndisks, isocorrbasis):
    """Get c_orf from the parameters for queryDisk model

    ##########
    :param pars:        Anisotropy parameterization
    :param ptapsrs:     List of pulsar objects
    :param Fe:          The pixel-basis signal response for the Earth term
    
    """
    npsrs = len(ptapsrs)

    orf_tot = np.zeros((npsrs,npsrs))

    Plm = Plm_from_anipars(np.array([pars[0]]))
    orf_tot += Plm[0] * isocorrbasis[0]
    for ii in range(ndisks):
        params = pars[4*ii+1:4*ii+5]
        m = get_map_from_queryDisk(params, ptapsrs, Fe)
        orf_tot += orfFromMap_fast(usermap=m, response=Fe.real)

    return orf_tot


def c_orf_pointSrc(pars, ptapsrs, Fe, npoints, isocorrbasis):
    """
    Get c_orf from the parameters for pointSrc model

    ##########
    :param pars:        Anisotropy parameterization
    :param ptapsrs:     List of pulsar objects
    :param Fe:          The pixel-basis signal response for the Earth term
    """
    
    npsrs = len(ptapsrs)

    orf_tot = np.zeros((npsrs,npsrs))

    Plm = Plm_from_anipars(np.array([pars[0]]))
    orf_tot += Plm[0] * isocorrbasis[0]
    for ii in range(npoints):
        params = pars[3*ii+1:3*ii+4]
        m = get_map_from_pointSrc(params, ptapsrs, Fe)
        orf_tot += orfFromMap_fast(usermap=m, response=Fe.real)

    return orf_tot