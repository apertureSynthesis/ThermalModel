import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import map_coordinates

def readObjPlate(fileName):
    """
    Reads coordinates of vertices and triangle facets from a .obj shape file.
    
    Parameters: 
        fileName (string): Name of the shape file

    Returns:
        tuple: (vertices, triangles)
            - vertices: Shape (3, N) array of Cartesian coordinates.
            - triangles: Shape (3, M) array of vertex indices.
    """

    #Check that we are dealing with a .obj file
    if not fileName.lower().endswith('.obj'):
        raise ValueError('Input file must be .obj format')
    nVert = 0
    nTrian = 0
    i=0
    j=0

    with open(fileName, 'r') as fn:
        for line in fn:
            if line.rstrip().split()[0] == 'v':
                nVert += 1
            else:
                nTrian += 1

        vertices = np.zeros((3, nVert))
        triangles = np.zeros((3, nTrian)) 

    with open(fileName, 'r') as fn:
        for line in fn:
            fline = line.rstrip().split()
            if fline[0] == 'v':
                vertices[0, i] = np.float64(fline[1])
                vertices[1, i] = np.float64(fline[2])
                vertices[2, i] = np.float64(fline[3])
                i+=1
            else:
                triangles[0, j] = int(fline[1])-1
                triangles[1, j] = int(fline[2])-1
                triangles[2, j] = int(fline[3])-1
                
                j+=1   
     

    return vertices, triangles.astype(int)  
                  

def readPlate(fileName):
    """
    Reads coordinates of vertices and triangle facets from a .plt shape file.
    
    Parameters: 
        fileName (string): Name of the shape file

    Returns:
        tuple: (vertices, triangles)
            - vertices: Shape (3, N) array of Cartesian coordinates.
            - triangles: Shape (3, M) array of vertex indices.
    """
    nVert = 0.
    nTrian = 0.

    #Check that we are dealing with a .plt file
    if not fileName.lower().endswith('.plt'):
        raise ValueError('Input file must be .plt format')

    with open(fileName, 'r') as fn:
        for line in fn:
            if len(line.split()) == 2:
                nVert = int(line.split()[0])
                nTrian = int(line.split()[1])
                break
    
        vertices = np.zeros((3, nVert))
        triangles = np.zeros((3, nTrian))
    
        i=0
        j=0
        for line in fn:
            if line.rstrip().endswith(']'):
                x,y,z1 = line.rstrip().split()
                z = z1.split(']')[0]
                vertices[0, i] = np.float64(x)
                vertices[1, i] = np.float64(y)
                vertices[2, i] = np.float64(z)

                i+=1
            else:
                x,y,z = line.rstrip().split()
                triangles[0, j] = int(x)-1
                triangles[1, j] = int(y)-1
                triangles[2, j] = int(z)-1
                
                j+=1              
        
    return vertices, triangles.astype(int)

def polyConvert(shape0, xs, ys):
    """
    Converts a spherical shape parameter array into 3D Cartesian vertices 
    and generates the surface polygon connectivity (triangles).
    
    Parameters:
        shape0 (array_like): Shape (3, N) array where:
                             row 0 = latitude (degrees)
                             row 1 = longitude (degrees)
                             row 2 = radius
        xs (int): Number of longitudinal vertices (scanlines).
        ys (int): Number of latitudinal vertices.
        
    Returns:
        tuple: (vertices, triangles)
            - vertices (np.ndarray): Shape (3, N) Cartesian coordinates (X, Y, Z).
            - triangles (np.ndarray): Shape (3, M) connectivity array of vertex indices.
    """
    shape = np.asarray(shape0, dtype=float)
    nVert = shape.shape[1]
    
    xs = int(xs)
    ys = int(ys)
    
    # Convert latitude and longitude from degrees to radians
    lat = np.radians(shape[0, :])
    lon = np.radians(shape[1, :])
    r = shape[2, :]
    
    # Compute the Cartesian vertices
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    
    vertices = np.vstack((x, y, z))
    
    # Compute the connectivity of the surface polygons
    nTrian = 2 * xs * (ys - 1)
    triangles = np.zeros((3, nTrian), dtype=int)
    
    sft = xs * (2 * ys - 3)
    
    # Vectorized computation of the first and last scanlines
    i = np.arange(xs)
    
    # The first scanline (pole to first ring)
    triangles[0, i] = 0
    triangles[1, i] = i + 1
    triangles[2, i] = ((i + 1) % xs) + 1
    
    # The last scanline (last ring to opposite pole)
    triangles[0, i + sft] = nVert - 1
    triangles[1, i + sft] = (((i + 1) % xs) + (ys - 2) * xs) + 1
    triangles[2, i + sft] = i + (ys - 2) * xs + 1
    
    # Vectorized computation for the main body
    iBody = np.arange(xs * (ys - 2))
    n = iBody // xs
    m = iBody % xs
    
    idx1 = iBody * 2 + xs
    idx2 = iBody * 2 + 1 + xs
    
    # First triangle of the quad
    triangles[0, idx1] = (m + n * xs) + 1
    triangles[1, idx1] = (m + (n + 1) * xs) + 1
    triangles[2, idx1] = (((m + 1) % xs) + (n + 1) * xs) + 1
    
    # Second triangle of the quad
    triangles[0, idx2] = (m + n * xs) + 1
    triangles[1, idx2] = (((m + 1) % xs) + (n + 1) * xs) + 1
    triangles[2, idx2] = (((m + 1) % xs) + n * xs) + 1
    
    #Reverse the winding order of the triangles
    triangles = triangles[::-1, :]
    
    return vertices, triangles

def xyz2rd(vec, sph_coord=False):
    """
    Converts Cartesian coordinates (X, Y, Z) to RA and Dec in degrees.
    
    Parameters:
        vec (array_like): Array of shape (3,) or (N, 3) containing (X, Y, Z) coordinates.
        sph_coord (bool): If True, returns Dec in spherical coordinates (0 for North, 180 for South)
            instead of ecliptic coordinates (90 for North, -90 for South).
        
    Returns:
        np.ndarray: Array of shape (N, 2) or (2,) containing [RA, Dec] in degrees.
    """
    vec = np.asarray(vec, dtype=float)
    
    # Handle single vector input
    is_1d = False
    if vec.ndim == 1 and len(vec) == 3:
        vec = vec.reshape(1, 3)
        is_1d = True
        
    x = vec[:, 0]
    y = vec[:, 1]
    z = vec[:, 2]
    
    ra = np.degrees((np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi))
    
    dec = np.degrees(np.arctan2(z, np.hypot(x, y)))
    
    
    if sph_coord:
        dec = 90.0 - dec
        
    # Combine into an (N, 2) array
    rd = np.column_stack((ra, dec))
    
    # If a single 1D vector was passed in, return a 1D array of length 2
    if is_1d:
        return rd[0]
        
    return rd

def rd2xyz(rd):
    """
    Converts RA and Dec in degrees to Cartesian coordinates (X, Y, Z).
    
    Parameters:
        rd (array_like): Array of shape (2, N) containing (RA, Dec) coordinates
        
    Returns:
        np.ndarray: Array of shape (N, 3) or (3,) containing (X, Y, Z) coordinates.
    """
    rd = np.array(rd)

    #Check for single vector input
    if rd.size == 2:
        rd = rd.reshape((1, 2))

    dec = np.radians(rd[:, 1])
    ra = np.radians(rd[:, 0])

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.squeeze(np.array([x,y,z]))

def meshCenters(vertices, triangles):
    """
    Computes the center points of triangular plates.
    
    Parameters:
        vertices (np.ndarray): Shape (3, N) containing (X, Y, Z) coordinates.
        triangles (np.ndarray): Shape (3, M) containing vertex indices for each triangle.
        
    Returns:
        np.ndarray: Shape (3, M) containing the (X, Y, Z) centers of each triangle.
    """

    return np.mean(vertices[:, triangles], axis=1)


def meshSphere(step1, step2, rad=1.0, debug=False):
    """
    Generates a 3D spherical mesh (vertices and triangle connectivity).
    
    Parameters:
        step1 (int): Number of steps in longitude.
        step2 (int): Number of steps in latitude.
        rad (float): Radius of the sphere (default=1.0).
        debug (bool): Optional debug flag.
        
    Returns:
        tuple: (vertices, triangles)
            - vertices: (3, N) numpy array of Cartesian coordinates.
            - triangles: (3, M) numpy array of vertex indices.
    """
    rad = float(rad)
    
    # Number of vertices (intermediate rings + 2 poles)
    nvert = step1 * (step2 - 1) + 2

    # Step size
    s1 = 360.0 / step1  # in longitude
    s2 = 180.0 / step2  # in latitude
    
    # Replicate IDL's makenxy behavior
    lon_1d = np.linspace(0, 360 - s1, step1)
    lat_1d = np.linspace(-90 + s2, 90 - s2, step2 - 1)
    
    # Create the grid. Flattening in default C-order means the last dimension (longitude)
    # varies fastest, perfectly matching the scanline loop logic in polyconvert.
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    lon = lon_grid.flatten()
    lat = lat_grid.flatten()

    # Shape model grid: shape (3, nvert)
    # Row 0: latitude, Row 1: longitude, Row 2: radius
    grid = np.zeros((3, nvert), dtype=float)
    
    # Fill intermediate latitudes and longitudes
    grid[0, 1:nvert-1] = lat
    grid[1, 1:nvert-1] = lon
    grid[2, :] = rad
    
    # Set the South pole at index 0 and North pole at index nvert-1
    grid[:, 0] = [-90.0, 0.0, rad]
    grid[:, nvert-1] = [90.0, 0.0, rad]

    # Convert it to triangulated plate format using the polyconvert function
    vertices, triangles = polyConvert(grid, step1, step2)
    
    return vertices, triangles

def meshEllipsoid(a, b, c, step1, step2):
    """
    Generates a 3D ellipsoid mesh (vertices and triangle connectivity).
    
    Parameters:
        step1 (int): Number of steps in longitude.
        step2 (int): Number of steps in latitude.
        a (float): first axis of the ellipsoid
        b (float): second axis of the ellipsoid
        c (float): third axis of the ellipsoid
        
    Returns:
        tuple: (vertices, triangles)
            - vertices: (3, N) numpy array of Cartesian coordinates.
            - triangles: (3, M) numpy array of vertex indices.
    """
    vertices, triangles = meshSphere(step1, step2)

    vertices[0, :] *= a
    vertices[1, :] *= b
    vertices[2, :] *= c

    return vertices, triangles

def meshNormal(vertices, polygons, vector=True):
    """
    Computes the normal vectors of surface plates.
    
    Parameters:
        vertices (np.ndarray): Shape (3, N) containing (X, Y, Z) coordinates.
        polygons (np.ndarray): Shape (3, M) containing vertex indices for each triangle.
        vector (bool): If True, returns the Cartesian normal vectors.
                       If False, returns the spherical coordinate normals.
                       
    Returns:
        normals (np.ndarray): Shape (3, M) containing the Cartesian or spherical normals
    """
    # Extract the three vertices for all triangles
    p1 = vertices[:, polygons[0, :]]
    p2 = vertices[:, polygons[1, :]]
    p3 = vertices[:, polygons[2, :]]

    #Calculate the unit normal vectors
    vectors = np.cross(p2 - p1, p3 - p1, axis=0)
    k = np.linalg.norm(vectors, axis=0)

    #Handle division by 0
    k[k == 0] = 1.0
    normVec = vectors / k[np.newaxis, :]
    
    # Convert to spherical coordinates
    normSph = xyz2rd(normVec)
    normSph[1, :] = 90.0 - normSph[1, :]
    
    if vector:
        normals = normVec
    else:
        normals = normSph

    return normals

def twoVectors(axDef, indexA, plnDef, indexP):
    """
    Function to define a reference frame from two input vectors. Directly translated from 
    SPICE lib routine twovec.f. Function returns a rotation matrix, mOut, so that any vector
    in the new frame can be expressed as mout @ (x,y,z)

    Parameters:
    axDef (array_like): Shape (3,) is a vector defining one of the principal axes of a coordinate frame
    indexA (int): A number that determines which of the three coordinate axes contains axDef.
        0 for the x-axis, 1 for the y-axis, 2 for the z-axis
    plnDef (array_like): Shape (3,) is a vector defining (with axDef) a principal plane of the coordinate frame.
        Same convention as indexA
    indexP (int): The second axis of the principal frame determined by axDef and plnDef

    Returns:
     rotMatrix (np.ndarray): Shape (3, 3) is a rotation matrix that transforms coordinates given in the input frame
        to the frame defined by axDef, plnDef, indexA, and indexP
    """

    unitMatrix = np.identity(3)
    rotMatrix = np.zeros((3,3))

    #Check for obvious bad inputs
    if (max([indexP, indexA]) > 2) or (min([indexP, indexA])< 0):
        print("The definition indices must lie in the range from 0 to 2. Returning unit matrix.")
        return unitMatrix
    
    if (indexA == indexP):
        print("The values of indexA and indexP must be different. Returning unit matrix")
        return unitMatrix
    
    i1 = indexA
    i2 = (indexA+1) % 3
    i3 = (indexA+2) % 3

    rotMatrix[i1,:] = axDef / np.linalg.norm(axDef)
    if indexP == i2:
        cross = np.cross(axDef, plnDef)
        rotMatrix[i3,:] = cross / np.linalg.norm(cross)
        cross = np.cross(rotMatrix[i3,:], axDef)
        rotMatrix[i2,:] = cross / np.linalg.norm(cross)
    else:
        cross = np.cross(plnDef, axDef)
        rotMatrix[i2,:] = cross / np.linalg.norm(cross)
        cross = np.cross(axDef, rotMatrix[i3,:])
        rotMatrix[i3,:] = cross / np.linalg.norm(cross)

    #Verify that we have a non-zero quantity in one of the columns in 
    #rotMatrix(1,I2) and rotMatrix(1,I3). We only need to check one of them
    #since they are related by a cross product.

    if (rotMatrix[i2,0] == 0) & (rotMatrix[i2,1] == 0) & (rotMatrix[i2,2] == 0):
        print('The input vectors axDef and plnDef were linearly dependent. Returning unit matrix.')
        return unitMatrix
    
    return np.transpose(rotMatrix)

def euRot(r, psi, phi=None, theta=None):
    """
    Rotates a vector by a set of Euler angles
    
    Parameters:
        r (array_like): Vector to be rotated. Can be (lat, lon) in spherical coordinates 
            or (x, y, z) in Cartesian coordinates.
        psi (float, degrees): first Euler angle
        phi (float, degrees): second Euler angle
        theta (float, degrees): third Euler angle
        
    Returns:
        output (np.ndarray): rotated vector
    """   

    #Make sure we have a vector
    r = np.asarray(r, dtype=float)

    #Check if we have spherical or cartesian coordinates
    if r.shape[0] == 2:
        #If spherical, convert to Cartesian
        lon = np.radians(r[0, :])
        lat = np.radians(r[1, :])

        xyz = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])
    else:
        xyz = r

    #Check whether this is a 2D or 3D rotation
    if phi is not None and theta is None:
        a_phi = psi
        a_theta = phi
        rv = R.from_euler('ZY',[a_phi,a_theta],degrees=True)
    elif phi is not None and theta is not None:
        rv = R.from_euler('ZXZ',[psi,theta,phi], degrees=True)
    else:
        raise ValueError("Invalid number of arguments.\n" 
                         "Usage: r' = euRot(r,phi,theta) or r' = euRot(r,psi,phi,theta)")
    
    #Perform the rotation
    xyzRot = rv.apply(xyz.T).T

    if r.shape[0] == 2:
        #If spherical input, return spherical output
        output = np.zeros_like(r)
        output[0, :] = (np.degrees(np.arctan(xyzRot[1,:]/xyzRot[0, :])) + 360) % 360
        output[1, :] = np.degrees(np.arctan(np.hypot(xyzRot[0,:],xyzRot[1,:])/xyzRot[2,:]))
    else:
        output = xyzRot

    return output

def vectMatchView(vertices,subEarthLatitude,subEarthLongitude,norAz):
    """
    Converts an input body-fixed frame vector into the frame used by the 
    meshGeoMap function for the purposes of calculating geometry maps.
    This program takes the sub-Earth (observer) latitude and longitude along
    with the north pole azimuthal angle as the necessary information to perform
    the transofrmation. This transformation aligns a shape model with 
    the viewing geometry of an object during observations.
    
    Parameters:
        vertices (np.ndarray): Shape (3, N) containing (X, Y, Z) coordinates.
        subEarthLatitude (float, degrees): sub-Earth(observer) latitude
        subEarthLongitude (float, degrees): sub-Earth(observer) longitude
        norAz (float, degrees): north pole azimuthal angle
        
    Returns:
        newVertices (np.ndarray): Shape (3, N) rotated vertices
    """   
    #Rotate sub-Earth longitude to +x axis
    xvert = euRot(vertices, -subEarthLongitude, 0)
    
    #Rotate sub-Earth latitude to the meridian, so the sub-Earth point is along the +z-axis
    zvert = euRot(xvert, 0, -90+subEarthLatitude)

    #Rotate to match the orientation
    newVertices = euRot(zvert, -90-norAz, 0)

    return newVertices

def cross2d(u, v):
    """
    Compute the cross product of two 2D vectors

    Parameters:
        u (np.ndarray): Shape (2,) first input vector
        v (np.ndarray): Shape (2,) second input vector

    Returns:
        float: magnitude of the cross product
    """
    return u[0]*v[1] - u[1]*v[0]

def meshGeoMap(vertices, triangles, sunPos, dRange, 
                xres=1.0, yres=1.0, xifov=None, yifov=None, 
                xs=256, ys=256, bench=False, debug=False, 
                noShadow=False, xc=None, yc=None):
    """
    Renders a 3D mesh into 2D geometry maps (incidence, emission, phase, mask, plate IDs).
    Uses the zBuffer method. Each pixel of each plate is compared with the z-value stored in the
    corresponding zBuffer. If it's in front of hte pixel that is in the current zBuffer, it takes
    the place of the previous pixel.

    Note: The resolution of the map can't be too small compared to the resolution of the plate model
    or the error will be large.

    Parameters:
        vertices (np.ndarray): Shape (3, N) containing (X, Y, Z) coordinates.
        triangles (np.ndarray): Shape (3, M) containing vertex indices.
        sunPos (np.ndarray): Shape (3,) containing the position of the Sun in the body-fixed frame of the plate model
        dRange (float): Distance of the observer at the origin of the shape model. 
            Must be positive; negative means the observer is at infinity.
        xres (float), yres(float): (optional) x and y resolution in plate model units/pixel.
            Defaults are 1. When set, they are calculated as xres/dRange and yres/dRange
        xifov (float), yifov (float): (optional) x and y instantaneous FOV (pixel scale) in radiance/pixel.
            These override xres and yres. Defaults are 1/range.
        xs (int), ys (int): (optional) x and y dimensions of the output maps. Default is 256 x 256.
        noShadow (bool): (optional) If set, turns off the shadow-casting search function. Default is conduct shadow-casting search.
        xc (int), yc (int): (optional) The pixel position of the figure center. Default is image center.

        
    Returns:
        iMap (np.ndarray): Shape (xs, ys) incident angle map
        eMap (np.ndarray): Shape (xs, ys) emission angle map
        aMap (np.ndarray): Shape (xs, ys) phase angle map
        pltMap (np.ndarray): Shape (xs, ys) plate indices of the corresponding pixel
            in the map. -1 is no plate coverage
        mask (np.ndarray): Shape (xs, ys) shadow-casting mask. Flags indicate:
            0: No plates at the index
            1: Plate faces away from the Sun
            2: Plate shadowed by other plates
            3: Plate is illuminated with visible pixels

    """
    if debug:
        bench = True

    if bench:
        t0 = time.time()

    xres = float(xres)
    yres = float(yres)
    
    xscl = xres / dRange if xifov is None else float(xifov)
    yscl = yres / dRange if yifov is None else float(yifov)
    
    xc = xs / 2.0 if xc is None else float(xc)
    yc = ys / 2.0 if yc is None else float(yc)

    sunPos = np.array(sunPos, dtype=float)
    if dRange < 0:
        obspos = np.array([0.0, 0.0, 1e50])
    else:
        obspos = np.array([0.0, 0.0, 1.0]) * dRange
        
    nVert = vertices.shape[1]
    nTriangle = triangles.shape[1]

    iMap = np.zeros((xs, ys), dtype=float)
    eMap = np.zeros((xs, ys), dtype=float)
    aMap = np.zeros((xs, ys), dtype=float)
    mask = np.zeros((xs, ys), dtype=np.uint8)
    pltMap = np.full((xs, ys), -1, dtype=int)

    #Obtain normal vectors and plate centers
    normals = meshNormal(vertices, triangles)

    centers = meshCenters(vertices, triangles)

    #Incidence and emission vectors
    vinc = sunPos[:, np.newaxis] - centers
    vemi = obspos[:, np.newaxis] - centers
    
    norm_vinc = np.sqrt(np.sum(vinc**2, axis=0))
    norm_vemi = np.sqrt(np.sum(vemi**2, axis=0))

    #Prevent division by zero
    norm_vinc[norm_vinc == 0] = 1.0
    norm_vemi[norm_vemi == 0] = 1.0
    
    #Calculate incidence, emission, and phase angles
    inc = np.degrees(np.arccos(
        np.clip(np.sum(normals * vinc, axis=0) / norm_vinc, -1.0, 1.0)
        ))
    emi = np.degrees(np.arccos(
        np.clip(np.sum(normals * vemi, axis=0) / norm_vemi, -1.0, 1.0)
        ))
    alpha = np.degrees(np.arccos(
        np.clip(np.sum(vinc * vemi, axis=0) / (norm_vinc * norm_vemi), -1.0, 1.0)
        ))


    zBuffer = np.full((xs, ys), -1e10, dtype=float)

    sclMatrix = np.array([[1/xres, 0., 0.], [0., 1/yres, 0.], [0., 0., 1.]])
    offset = np.array([xc, yc, 0.])
    
    #Calculate line of sight, canvus coordinates and pixel depth (m, n, z)
    los = vertices - np.array([[0], [0], [dRange]])
    vertCan = np.zeros((3, nVert))
    vertCan[0, :] = -np.arctan(los[0, :] / los[2, :]) / xscl
    vertCan[1, :] = -np.arctan(los[1, :] / los[2, :]) / yscl
    vertCan[2, :] = vertices[2, :]
    vertCan += offset[:, np.newaxis]

    #Get canvas normals
    normCan = meshNormal(vertCan, triangles)
    zGrad = np.array([-normCan[0, :] / normCan[2, :], 
                       -normCan[1, :] / normCan[2, :]])

    for j in range(nTriangle):
        #Test all pixels that may be in the range of the current triangle
        #and compute their depths
        a = vertCan[:, triangles[0, j]]
        b = vertCan[:, triangles[1, j]]
        c = vertCan[:, triangles[2, j]]

        #Edge vectors
        ba = b[:2] - a[:2]
        cb = c[:2] - b[:2]
        ac = a[:2] - c[:2]
        
        mMin = int(np.floor(min(a[0], b[0], c[0])))
        mMax = int(np.ceil(max(a[0], b[0], c[0])))
        nMin = int(np.floor(min(a[1], b[1], c[1])))
        nMax = int(np.ceil(max(a[1], b[1], c[1])))
        
        m0, n0, z0 = a[0], a[1], a[2]

        #Center of the plate        
        center = np.array([np.mean([a[0], b[0], c[0]]), np.mean([a[1], b[1], c[1]])])

        #Each possible pixel index
        mRange = range(max(mMin, 0), min(mMax, xs - 1) + 1)
        nRange = range(max(nMin, 0), min(nMax, ys - 1) + 1)

        aDis = np.array([
            [b[1] - a[1], c[1] - b[1], a[1] - c[1]],
            [a[0] - b[0], b[0] - c[0], c[0] - a[0]]
        ])

        cDis = np.array([
            b[0]*a[1] - a[0]*b[1],
            c[0]*b[1] - b[0]*c[1],
            a[0]*c[1] - c[0]*a[1]
        ])

        #Calculate distance from the center to each side of the plate
        dCenter = center @ aDis + cDis
        
        for m in mRange:
            for n in nRange:
                p = np.array([m, n])
                #Calculate distance from each point p to each side of the plate
                Dmn = p @ aDis + cDis

                if np.min(dCenter * Dmn) >= 0: #If True, this pixel is in the plate
                    z = z0 + (m - m0) * zGrad[0, j] + (n - n0) * zGrad[1, j]
                    if z > zBuffer[n, m]: #If True, this takes the place of the previous pixel
                        zBuffer[n, m] = z
                        iMap[n, m] = inc[j]
                        eMap[n, m] = emi[j]
                        aMap[n, m] = alpha[j]
                        pltMap[n, m] = j
                        mask[n, m] = 1 if inc[j] > 90 else 3

    if bench:
        print(f'Image generation done in {time.time() - t0:.4f} seconds')
        t0 = time.time()

    if noShadow:
        return iMap, eMap, aMap, mask, pltMap

    """
    Shadow casting procedure. Each pixels in the image I have their z-depth
    in the zBuffer. This can be used to map back into the real world coordinates
    and test if they are blocked by plates in front of them.
    """
    effpixFlat = np.where(mask.flatten() == 3)[0]
    if len(effpixFlat) == 0:
        print('No non-zero pixels in the image')
        return iMap, eMap, aMap, mask, pltMap

    #Map all pixels back to the scatter frame where the Sun
    #is at the +x direction and the observer in the x-y plane
    #Construct the transformation matrix
    xFormScat = twoVectors(sunPos, 0, [0, 0, 1], 1)
    
    xpix, ypix = np.meshgrid(np.arange(ys), np.arange(xs))
    xpix = xpix.flatten()
    ypix = ypix.flatten()
    zBufFlat = zBuffer.flatten()
    
    #Extract effective pixels
    pts = np.vstack((xpix[effpixFlat], ypix[effpixFlat], zBufFlat[effpixFlat]))
    pts = pts - offset[:, np.newaxis]
    
    sclMatrixInv = np.linalg.inv(sclMatrix)
    pixr = (pts.T @ sclMatrixInv) @ xFormScat
    pixr = pixr.T 
    
    pixelsFlat = mask.flatten()
    
    minY, maxY = np.min(pixr[1, :]), np.max(pixr[1, :])
    minZ, maxZ = np.min(pixr[2, :]), np.max(pixr[2, :])
    
    #Sort and bin pixels by z-depth so that per-triangle
    #shadow tests only need to consider the pixel subset whose
    #z-range overlaps with that triangle
    zSort = np.argsort(pixr[2, :])
    zRange = maxZ - minZ

    #Find a valid quantization level.
    #Start at (unique z levels - 5) and ensure
    #it doesn't exceed 100, then reduce until
    #exactly qLevel unique quantized bins exist
    qLevel = len(np.unique(pixr[2, :])) - 5
    qLevel = max(2, min(100, qLevel))
    
    while True:
        zQuan = np.floor(
            (pixr[2, zSort] - minZ) / zRange * (qLevel -1)
        ).astype(int)
        #Indices of first occurence of each unique quantized z
        zQuanUniq = np.unique(zQuan, return_index=True)[1] 
        if len(zQuanUniq) == qLevel:
            break
        qLevel -= 1
        if qLevel < 2:
            break

    if qLevel < 2:
        mask = pixelsFlat.reshape((xs, ys))
        return iMap, eMap, aMap, mask, pltMap
    
    #zQuanUniq[q] is the index in zSort of the first pixel
    #with quantized z = q

    zQuanUniq = np.concatenate(([0], zQuanUniq, [len(zSort)]))
    
    #Transform vertices back to scattering frame
    vert1 = xFormScat.T @ vertices
    
    for j in range(nTriangle):
        a = vert1[:, triangles[0, j]]
        b = vert1[:, triangles[1, j]]
        c = vert1[:, triangles[2, j]]
        
        yminJ, ymaxJ = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        zminJ, zmaxJ = min(a[2], b[2], c[2]), max(a[2], b[2], c[2])
        xmin = min(a[0], b[0], c[0])
        
        if not (ymaxJ < minY or yminJ > maxY or zmaxJ < minZ or zminJ > maxZ):

            #Map triangle z-extents to quantization bins
            zminJQ = int(np.floor((zminJ - minZ) / zRange * (qLevel - 1)))
            zminJQ = max(0, min(zminJQ, qLevel - 1))
            zmaxJQ = int(np.floor((zmaxJ - minZ) / zRange * (qLevel - 1)))
            zmaxJQ = max(0, min(zmaxJQ, qLevel - 1))

            #Retrieve only pixels whose quantized z falls in [zminJQ, zmaxJQ]
            ind1 = zSort[zQuanUniq[zminJQ + 1]: zQuanUniq[zmaxJQ + 2]]

            if len(ind1) == 0:
                continue
            pixr1 = pixr[:, ind1]
            aDis = np.array([
                [b[2] - a[2], c[2] - b[2], a[2] - c[2]],
                [a[1] - b[1], b[1] - c[1], c[1] - a[1]]
            ])

            cDis = np.array([
                b[1]*a[2]-a[1]*b[2],
                c[1]*b[2]-b[1]*c[2],
                a[1]*c[2]-c[1]*a[2] 
            ])

            
            center = (a[1:3] + b[1:3] + c[1:3]) / 3.0

            dCenter = center @ aDis + cDis
            
            inPltPix = np.where((pixr1[1, :] >= yminJ) & (pixr1[1, :] < ymaxJ) & 
                                (pixr1[2, :] >= zminJ) & (pixr1[2, :] < zmaxJ))[0]
            
            if len(inPltPix) > 0:
                pts_eval = pixr[1:3, inPltPix]

                smx = pts_eval.T @ aDis + cDis
                blocked = np.where(
                    np.min(smx * dCenter, axis=1) > 0
                    )[0]
                
                for p in blocked:
                    idx = inPltPix[p]
                    if pixr[0, idx] < xmin:
                        pixelsFlat[effpixFlat[idx]] = 2

    mask = pixelsFlat.reshape((xs, ys))
    print('Shadow correction performed')

    if bench:
        print(f'Shadow casting correction done in {time.time() - t0:.4f} seconds')

    return iMap, eMap, aMap, mask, pltMap

def tpmSph2Plt(vertices, triangles, tref, soLon, times=None, lats=None, retrograde = False, cubic = True):
    """
    Calculate the temperature profile of a triangular plate shape model
    with the the thermophysical model from KRC for a spherical shape

    Parameters:
        vertices (np.ndarray): Shape (3, N) containing (X, Y, Z) coordinates.
        triangles (np.ndarray): Shape (3, M) containing vertex indices.
        tref (np.ndarray): Shape (Times, Depths, Latitudes) containing the temperature profile from KRC for each
            combination of LST, depth, and latitude
        soLon (float): sub-solar longitude for consideration
        times (np.ndarray): (optional) list of local solar times (LST) to consider
        lats (np.ndarray): (optional) list of sub-solar latitudes to consider
        retrograde (bool): (optional) consider retrograde motion. Default is False
        cubic (bool): (optional) method for interpolation. Default is cubic.

    Returns:
        temperature (np.ndarray): Shape (M, N) containing the temperature of plate M at depth N
    """
    subsLon = ((soLon % 360) + 360) % 360

    tref = np.asarray(tref)
    nTime, nDepth, nLat = tref.shape

    #check latitudes
    if lats is None or len(lats) != nLat:
        if lats is not None:
            print("Invalid latitude array was ignored")
        lats = np.linspace(-90, 90, nLat)
    else:
        lats = np.asarray(lats)

    #Check times
    if times is None or len(times) != nTime+1:
        if times is not None:
            print("Invalid time array was ignored")
        times = np.linspace(0, 360, nTime+1)
    else:
        times = np.asarray(times)

    nT = len(times)
    nTri = triangles.shape[1]

    #Calculate the planetocentric longitude and latitude
    lonLat = xyz2rd(np.transpose(meshNormal(vertices, triangles, vector=True)))

    #Prepare the reference temperature profile
    tRef1 = np.zeros((nT, nDepth, nLat), dtype=float)
    tRef1[:-1, :, :] = tref
    tRef1[-1, :, :] = tref[0, :, :]


    #Calculate local time for all plates
    localTime = subsLon - lonLat[:, 0]
    if retrograde:
        localTime = -localTime
    localTime = (localTime + 360) % 360

    #Local latitude for all plates
    localLat = lonLat[:, 1]

    #Find floating point pixel positions
    tInd = np.arange(nT)
    lInd = np.arange(nLat)

    timePos = np.interp(localTime,times,tInd)
    latPos = np.interp(localLat,lats,lInd)

    #Generate grid of coordinates sampling tref1 from all plates and depths
    #Desire output of shape (nTri, nDepth)
    timeCoords = np.broadcast_to(timePos[:, np.newaxis], (nTri, nDepth))
    latCoords = np.broadcast_to(latPos[:, np.newaxis], (nTri, nDepth))
    zCoords = np.broadcast_to(np.arange(nDepth)[np.newaxis, :], (nTri, nDepth))

    coords = np.stack((timeCoords, zCoords, latCoords))

    #Interpolate the input reference temperature profile
    order = 3 if cubic else 1
    temperature = map_coordinates(tRef1, coords, order=order, mode='nearest')
        
    return temperature

def tpmMapping(tempList, pltMap, depth=None, zz=None, cubic=True, tss0 = None):
    """
    Maps the temperature distribution of a plate shape model onto an image.

    Parameters:
        tempList (np.ndarray): Shape (M, N) containing the temperature of plate M at depth N
        pltMap (np.ndarray): Shape (X, Y) containing the index of each plate for consideration
        depth (np.ndarray): (optional) Shape (Z,) containing the depth for which the temperature is mapped.
            If keyword zz is specified, depth is the same unit as zz. 
        zz (np.ndarray): (optional) Shape (Z,) contains the depth profile
        tss0 (np.ndarray): (optional) Shape (M,) contains the theoretical subsolar temperature
            in the thermophysical model. It scales the dimensionless temperature in tempList to
            the physical temperature in Kelvin.
        cubic (bool): (optional) Specifies whether to use cubic interpolation

    """
    tempList = np.array(tempList)
    if tempList.ndim != 2:
        raise ValueError("Temperature array must be of dimensions (nTri, nDepth)")
    if depth is not None:
        depth = np.array(depth)

    nTri, nDep = tempList.shape

    #Check subsolar temperature
    if tss0 is None:
        tss = np.ones(nTri)
    else:
        tss = tss0
    if len(tss) == 1:
        tss = np.ones(nTri)*tss0
    if len(tss) != nTri:
        print("Ignoring invalid temperature array")
        tss = np.ones(nTri)

    #Process depth array
    if zz is None:
        zArr = np.arange(nDep)
    else:
        zArr = zz
    if zArr.size != nDep:
        print("Ignoring invalid depth array")
        zArr = np.arange(nDep, dtype=float)
    nZ = zArr.size

    #Check depth parameters
    if depth is None:
        depth = np.zeros(1)
    inds = np.where((depth<0) | (depth > max(zArr)))
    if inds[0].size != 0:
        print('Invalid depth array ignored')
        depth = depth[~inds]
        if depth.size == 0:
            print('No valid depth specified. Returning surface temperature.')
            depth = np.zeros(1)
    nDepth = depth.size

    #Check plate map
    if np.nanmax(pltMap) > (nTri-1):
        raise ValueError('Wrong plate map detected, stopping')
    
    #Temperature at each depth

    kind = 'cubic' if (cubic and nDepth >= 4) else 'linear'

    tdFunc = interp1d(zArr, tempList, axis=1, kind=kind, bounds_error=False, fill_value="extrapolate")
    tDep = tdFunc(depth)
    
    #Convert to Kelvin according to subsolar temperature
    tDep *= tss[:, np.newaxis]

    #Map the temperature to the image
    xs, ys = pltMap.shape
    img = np.zeros((xs, ys, nDepth), dtype=float)

    #Mask out background pixels
    mask = pltMap >= 0

    img[mask] = tDep[pltMap[mask]]

    return np.transpose(img, axes=[2, 0, 1])