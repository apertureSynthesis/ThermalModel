import numpy as np
from scipy.spatial.transform import Rotation as R

def readPlate(fileName):

    nvert = 0.
    ntrian = 0.

    x = 0.
    y = 0.
    z = 0.


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

def polyConvert(shapes, xs, ys):

    shape = shapes.astype(np.float64)

    n_vert = len(shape) / 3
    xs = np.int(xs)
    ys = np.int(ys)

    #Compute the vertices
    shape[0:1, :] *= np.pi / 180
    vertices = np.zeros((3, n_vert), dtype=float)
    vertices[0, :] = shape[2, :] * np.cos(shape[1, :]) * np.cos(shape[0, :])
    vertices[1, :] = shape[2, :] * np.sin(shape[1, :]) * np.cos(shape[0, :])
    vertices[2, :] = shape[2, :] * np.sin(shape[0, :])
    shape = 0 #release memory

    #Compute the connectivity of the surface polygons
    n_tria = 2 * xs * (ys - 1)
    triangles = np.zeros((3, n_tria), dtype=int)

    #Construct the connectivity array
    sft = xs * (2*ys - 3)

    for i in range(xs):
        #the first scanline
        triangles[:, i] = [0, i+1, ((i+1)%xs)+1]
        #the last scanline
        triangles[:, i+sft] = [n_vert-1, (((i+1) % xs) + (ys-2)*xs)+1, i+(ys-2)*xs+1]

    for i in range(xs*(ys-2)):
        n = i/xs
        m = i-n*xs

        triangles[:,i*2+xs] = [(m+n*xs)+1, (m+(n+1)*xs)+1, (((m+1) % xs)+(n+1)*xs)+1]
        triangles[:,i*2+1+xs] = [(m+n*xs)+1, (((m+1) % xs)+(n+1)*xs)+1, (((m+1) % xs)+n*xs)+1]

    triangles = np.reverse(triangles)

    return vertices, triangles


def meshSphere(step1,step2,radius=1):

    #Number of vertices
    nvert = step1 * (step2 - 1) + 2

    #Step size
    s1 = 360 / step1 #longitude
    s2 = 180 / step2 #latitude

    lon = np.linspace(0, 360-s1, step1)
    lat = np.linspace(-90+s2, 90-s2, step2)

    #Shape model grid
    grid = np.zeros((3, nvert))
    grid[0, 1:nvert-2] = lat
    grid[1, 1:nvert-2] = lon
    grid[2,:] = radius
    grid[:,0] = [-90, 0, radius]
    grid[:, nvert-1] = [90, 0, radius]

    #Convert to triangle plate format
    vertices, triangles = polyConvert(grid, step1, step2)

    return vertices, triangles

def meshEllipsoid(a, b, c, step1, step2):

    vertices, triangles = meshSphere(step1, step2)

    vertices[0, :] *= a
    vertices[1, :] *= b
    vertices[2, :] *= c

    return vertices, triangles

def rd2xyz(ra,dec):

    return np.array([[np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)],[np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)],[np.sin(dec*np.pi/180.)]])

def pyrot2(r,phi,theta):

    rv = R.from_euler('yz',[theta,phi],degrees=True)
    return rv.apply(r)

def vectMatchView(vertices,subEarthLatitude,subEarthLongitude,norAz):
    #Rotate sub-Earth longitude to +x axis
    xvert = pyrot2(vertices, -subEarthLongitude, 0)
    
    #Rotate sub-Earth longitude to +x axis
    zvert = pyrot2(xvert, 0, -90+subEarthLatitude)

    #Rotate sub-Earth latitude to meridian, so that sub-Earth points towards +z axis
    new_vertices = pyrot2(zvert, -90-norAz, 0)

    return vertices
