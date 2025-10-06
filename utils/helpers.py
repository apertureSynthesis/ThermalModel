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