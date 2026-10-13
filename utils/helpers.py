import numpy as np
from ast import literal_eval
import astropy.units as u
from datetime import datetime
from astropy.time import Time
import astropy.constants as const
from astroquery.jplhorizons import Horizons

def getPars(parFile):
    """
    Read in content from the input parameter file 
    and store them in a dictionary for later use
    """

    pars = {}
    with open(parFile, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                if line.rstrip():
                    split_line = line.rstrip().split("#")
                    var, val = split_line[0].split('=')
                    var = var.replace('"','')
                    val = val.replace('"','')
                    #Check if the value is actually an array
                    if (val.strip()[0] == '[') & (val.strip()[-1] == ']'):
                        pars[var.strip()] = np.array(literal_eval(val.strip()))
                    else:
                        #Check for boolean
                        if val.strip() == 'True':
                            pval = True
                        elif val.strip() == 'False':
                            pval = False
                        else:
                            pval = val.strip()
                        pars[var.strip()] = pval

    return pars

def convertObjToPlt(objFile):
    """
    Read in a shape model in .obj format and convert it to .plt format. Save the output file
    """
    pltFile = objFile[:-3]+'.plt'
    nv = 0
    nf = 0
    with open(objFile, 'r') as fobj:
        for line in fobj:
            if line.startswith('v'):
                nv += 1
            if line.startswith('f'):
                nf += 1
    
    with open(objFile, 'r') as fobj:
        with open(pltFile, 'w') as fplt:
            fplt.write(f"{nv:.0f} {nf:.0f}\n")
            for line in fobj:
                lsplit = line.split()
                if line.startswith('v'):
                    fplt.write(f"{lsplit[1]}\t{lsplit[2]}\t{lsplit[3]}]\n")
                if line.startswidth('f'):
                    fplt.write(f"{lsplit[1]}\t{lsplit[2]}\t{lsplit[3]}\n")


def subPoints(lamda, beta, period, phi, t0, tf, objectName, location, point='Earth'):
    """
    Inputs:
    lamda - rotation pole ecliptic longitude (degrees)
    beta - rotation pole ecliptic latitude (degrees)
    period - rotation period (hr)
    phi - reference angle at t0 (deg)
    t0 - reference epoch
    tf - epoch of interest
    point - "Earth" returns sub-Earth point; "Solar" returns sub-solar point

    Outputs:
    (latF, lonF) - latitude and longitude of desired sub-point (degrees)
    """

    t0 = Time(t0, scale='tdb', format='jd')
    tf = Time(tf, scale='utc')

    #Get the Earth-asteroid vector from Horizons
    obj = Horizons(id=str(objectName), location=location, epochs = tf.tdb.jd)

    geoTable = obj.vectors(refplane='ecliptic', aberrations='astrometric')

    #Asteroid-earth vector (au)
    r_e = np.array(
        [geoTable['x'][0],
         geoTable['y'][0],
         geoTable['z'][0]]
    )

    #Retrieve light travel time from Horizons (days)
    ltt = geoTable['lighttime'][0]

    #Calculate time that photons left the asteroid surface
    tEmit = tf.tdb.jd - ltt

    #Retrieve the solar-asteroid vector
    obj = Horizons(id=str(objectName), location='@10', epochs=tEmit)

    solarTable = obj.vectors(refplane='ecliptic', aberrations='geometric')

    #Sun-asteroid vector (au)
    r_s = np.array(
        [solarTable['x'][0],
         solarTable['y'][0],
         solarTable['z'][0]]
    )

    #Select input vector for rotation depending on point type requested
    if point == 'Earth':
        r_ast = -1 * r_e
    elif point == 'Solar':
        r_ast = -1 * r_s
    else:
        raise ValueError("point type must be Earth or Solar")

    #Normalize vector
    d_ast = np.linalg.norm(r_ast)
    dir_ast = r_ast / d_ast

    #Calculate time elapsed from photons leaving asteroid to DAMIT reference time
    dt = tEmit - t0.tdb.jd #days
    dh = dt * 24.0 * u.hr #hours

    #Number of asteroid rotations since reference time
    nRot = dh / period
    #Position of prime meridian
    W = phi.to(u.rad).value + 2 * np.pi * nRot.value
    W %= 2 * np.pi

    #Define rotation matrices

    def rotZ(theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    def rotY(theta):
        return np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

    #Transform eliptic pole coordinates into body-fixed frame
    lamda_rad = lamda.to(u.rad).value
    colat_rad = (90*u.deg - beta).to(u.rad).value

    #Rotate from ecliptic frame to asteroid frame
    R_matrix = rotZ(W) @ rotY(colat_rad) @ rotZ(lamda_rad)

    #Transform vector into asteroid frame
    r_final = R_matrix @ dir_ast

    #Sub-point latitude and longitude
    latF = np.degrees(np.arcsin(r_final[2]))
    lonF0 = np.degrees(np.arctan2(r_final[1], r_final[0]))
    lonF = (lonF0 % 360)

    return latF, lonF