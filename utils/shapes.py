import numpy as np

def readPlate(fileName):

    nvert = 0.
    ntrian = 0.

    x = 0.
    y = 0.
    z = 0.
    i1 = 0.
    i2 = 0.
    i3 = 0.

    with open(fileName, 'r') as fn:
        for line in fn:
            if len(line.split()) == 2:
                nvert = line.split[0]
                ntrian = line.split[1]
                break
    
        vertices = np.zeros((3,nvert))
        triangles = np.zeros((3,ntrian))
    
        i=0
        for line in fn:
            if not (line.endswith(']')):
                break
            x,y,z1 = line.split()
            z = z1.split(']')[0]
            vertices[i,0] = x
            vertices[i,1] = y
            vertices[i,2] = z
            
            i+=1

        i=0
        for line in fn:
            x,y,z = line.split()
            triangles[i,0] = x
            triangles[i,1] = y
            triangles[i,2] = z
            
            i+=1
        
    return vertices, triangles
            