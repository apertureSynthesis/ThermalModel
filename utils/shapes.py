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

    n_vert = shape.size / 3
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

def xyz2rd(xyz):
    vec = xyz.reshape((1,3))

    x = vec[:,0]
    y = vec[:,1]
    z = vec[:,2]

    rd = np.array([[(np.atan(y,x)+2*np.pi)%(2*np.pi)],[np.atan(z,np.sqrt(x**2+y**2))]]) * np.pi/180

    return rd

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

def meshNormal(vertices,polygons):

    polygons = np.array(polygons)
    surfNum = polygons.size / 3

    vert = np.reshape(vertices[:,polygons], (3,3,surfNum))
    p1 = np.squeeze(vert[:,0,:])
    p2 = np.squeeze(vert[:,1,:])
    p3 = np.squeeze(vert[:,2,:])

    l=p1[1,:]*p2[2:]+p2[1,:]*p3[2,:]+p3[1,:]*p1[2,:]-p1[1,:]*p3[2,:]-p2[1,:]*p1[2,:]-p3[1,:]*p2[2,:]
    l = np.squeeze(l)
    m=p1[2,:]*p2[0,:]+p2[2,:]*p3[0,:]+p3[2,:]*p1[0,:]-p1[2,:]*p3[0,:]-p2[2,:]*p1[0,:]-p3[2,:]*p2[0,:]
    m = np.squeeze(m)
    n=p1[0,:]*p2[1,:]+p2[0,:]*p3[1,:]+p3[0,:]*p1[1,:]-p1[0,:]*p3[1:]-p2[0,:]*p1[1,:]-p3[0,:]*p2[1,:]
    n = np.squeeze(n)
    k = np.sqrt(l*l + m*m + n*n)
    xyz = np.transpose(np.array([[l],[m],[n]])) / np.matmul(np.ones(3), k)
    norm = np.transpose(xyz2rd(np.transpose(xyz)))
    norm[1, :] = 90 - norm[1, :]

    return norm

def meshCenters(vertices, triangles):
    triangles = np.array(triangles)
    nTriangles = triangles.size / 3
    centers = np.sum( np.reshape(vertices[:, triangles], (3,3,nTriangles)), axis=1) / 3
    
    return centers


def meshGeoMap(vertices, triangles, sunpos, range, xres=1, yres=1, xs=256, ys=256):

    #Set up variables
    xscl = np.float64(xres) / range #pixel scale in radians/pixel
    yscl = np.float64(yres) / range #pixel scale in radians/pixel

    xc = xs / 2.
    yc = ys / 2.

    if range < 0:
        obspos = [0, 0, 1e50]
    else:
        obspos = [0, 0, 1.] * range

    nVert = vertices.size / 3.
    nTriangle = triangles.size / 3.

    imap = np.zeros((xs,ys), dtype=float)
    emap = np.zeros((xs,ys), dtype=float)
    amap = np.zeros((xs,ys), dtype=float)
    mask = np.zeros((xs,ys), dtype=int)
    pltmap = np.zeros((xs,ys), dtype=int)
    pltmap[:,:] = -1

    #compute the plate outward unit normal vectors
    normals = meshNormal(vertices, triangles)

    #incidence, emission, and phase angle lists of plates
    centers = meshCenters(vertices, triangles)

    vinc = (np.matmul(np.ones(nTriangle), sunpos)) - centers
    vemi = (np.matmul(np.ones(nTriangle), obspos)) - centers

    inc = np.acos(np.sum(normals*vinc, axis=0) / np.sqrt(np.sum(vinc*vinc, axis=0))) * 180/np.pi
    emi = np.acos(np.sum(normals*vemi, axis=0) / np.sqrt(np.sum(vemi*vemi, axis=0))) * 180/np.pi
    alpha = np.acos(np.sum(vinc*vemi, axis=0) / np.sqrt(np.sum(vinc*vinc, axis=0)*np.sum(vemi*vemi, axis=0))) * 180/np.pi

    #Construct geometry maps for i, e, and alpha
