import numpy as np
from ast import literal_eval

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
                        pars[var.strip()] = val.strip()

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

