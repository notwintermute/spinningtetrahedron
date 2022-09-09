import math
import random

import pygame
import numpy as np

resolution_side_length = 900   # size of the window
resX = resolution_side_length  # width of window
resY = resolution_side_length  # height of window
randomRot = False  # random rotation mode

# axis angles (in degrees)
taX = 60  # tetrahedron's roll  (x-axis rotation)
taY = 40  # tetrahedron's pitch (y-axis rotation)
taZ = 20  # tetrahedron's yaw   (z-axis rotation)

s = 300  # side length of tetrahedron
tCx = 0  # x pos of tetrahedron center
tCy = 0  # y pos of tetrahedron center
tCz = 0  # z pos of tetrahedron center
tCenter = np.array([0, 0, 0])

# this generates the points of the tetrahedron (done relative to the center)
A = np.array([tCx - s / 2, tCy - (s / (2 * math.sqrt(3))), tCz - (s * (math.sqrt(6) / 12))])
B = np.array([tCx + s / 2, tCy - (s / (2 * math.sqrt(3))), tCz - s * (math.sqrt(6) / 12)])
C = np.array([tCx, tCy + (s / math.sqrt(3)), tCz - s * (math.sqrt(6) / 12)])
D = np.array([tCx, tCy, tCz + s * ((math.sqrt(6) / 3) - (math.sqrt(6) / 12))])

# this is the "camera" position
cx, cy, cz = 0, 0, 100
cCen = np.array([cx, cy, cz])

# these are the camera normals (for orienting it)
cA = np.array([cx - 1, cy, cz])
cB = np.array([cx, cy, 1 + cz])
cP = [cCen, cA, cB]

# normalized vector pointing towards the tetrahedron from the camera
normlp = tCenter - cCen
normlp = normlp / np.linalg.norm(normlp)
Scp = np.dot(normlp, -cP[0])

# colors of the sides
c1 = (255, 255, 255)  # white
c2 = (255, 0, 0)      # red
c3 = (0, 255, 0)      # green
c4 = (0, 0, 255)      # blue

# colors get stored in a list
cL = [c1, c2, c3, c4]


# function for converting degrees to radians
def dtr(a):
    return (math.pi / 180) * a


# projects the shapes onto the screen
def project(p):
    dist = abs(np.dot(p, normlp) + Scp)
    return p + dist * normlp


# converts screen positions to cartesian coordinates
def stc(inp):
    sx = inp[0] - resX / 2
    sy = -(inp[1] - resY / 2)
    return sx, sy


# converts screen positions to cartesian coordinates
def cts(inp):
    sx = inp[0] + resX / 2
    sy = (resY / 2) - inp[1]
    return sx, sy


pygame.init()
# initialize surface and start the main loop
surface = pygame.display.set_mode((resX, resY))
pygame.display.set_caption('3d')
running = True
# --------------------------------------- Main Loop ---------------------------------------
while running:

    # ------------------------------------- input handling -------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        # ------------------------------------ key press actions ------------------------------------
        if event.type == pygame.KEYDOWN:

            # the following are the rotations about each axis
            if event.key == pygame.K_q:
                taZ += 10  # increases the yaw
            if event.key == pygame.K_e:
                taZ -= 10  # decreases the yaw
            if event.key == pygame.K_w:
                taX -= 10  # decreases the roll
            if event.key == pygame.K_s:
                taX += 10  # increases the roll
            if event.key == pygame.K_a:
                taY -= 10  # decreases the pitch
            if event.key == pygame.K_d:
                taY += 10  # increases the pitch

            # toggles random rotations (so it rotates hands free)
            if event.key == pygame.K_z:
                if randomRot:
                    randomRot = False
                else:
                    randomRot = True

    # ---------------------------------------- Updating Parameters ----------------------------------------
    if randomRot:  # randomly rotates the tetrahedron (a little bit jittery)
        taX += (1/10)*random.randrange(-5, 5)
        taY += (1/10)*random.randrange(-5, 5)
        taZ += (1/10)*random.randrange(-5, 5)

    # rotation handling is done by generating the rotation matrix
    # corresponding to the new pitch/roll/yaw
    # rotation matrix X
    cosX = math.cos(dtr(taX))
    sinX = math.sin(dtr(taX))
    taX = 0
    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    # rotation matrix Y
    cosY = math.cos(dtr(taY))
    sinY = math.sin(dtr(taY))
    taY = 0
    Ry = np.zeros((3, 3))
    Ry[1, 1] = 1
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    # rotation matrix Z
    cosZ = math.cos(dtr(taZ))
    sinZ = math.sin(dtr(taZ))
    taZ = 0
    Rz = np.zeros((3, 3))
    Rz[2, 2] = 1
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ

    # applying the final rotation
    RM = np.matmul(Ry, Rx)
    RM = np.matmul(Rz, RM)
    A = np.matmul(RM, A.transpose()).transpose()
    B = np.matmul(RM, B.transpose()).transpose()
    C = np.matmul(RM, C.transpose()).transpose()
    D = np.matmul(RM, D.transpose()).transpose()

    # these tuples store the side vertices
    s1 = (A, B, C)
    s2 = (A, B, D)
    s3 = (B, C, D)
    s4 = (C, A, D)

    # this list stores each of the tuples of vertices
    sL = [s1, s2, s3, s4]

    # calculating which sides are the closest
    # this is done by calculating their normals
    # and comparing their dist to the camera
    minHold = []
    for i, p in enumerate(sL):
        v1 = p[0] - p[1]
        v2 = p[0] - p[2]
        norml = np.cross(v1, v2)
        norml /= np.linalg.norm(norml)
        Sc = np.dot(norml, -p[0])
        minHold.append([i, abs(np.dot(cP[0], norml) - Sc)])

    # sorts them by distance then reverses
    # they are rendered furthest to closest
    # so there isn't any weird clipping
    minHold.sort(key=lambda x: x[1])
    minHold = minHold[::-1]

    # ---------------------------------------- Rendering ----------------------------------------
    surface.fill((0, 0, 0))

    # projects the sides to the camera
    # then converts from coord to screen pos
    pA = cts(project(A))
    pB = cts(project(B))
    pC = cts(project(C))
    pD = cts(project(D))

    # projected sides
    Ps1 = (pA, pB, pC)  # white
    Ps2 = (pA, pB, pD)  # red
    Ps3 = (pB, pC, pD)  # blue
    Ps4 = (pC, pA, pD)  # green
    PsL = [Ps1, Ps2, Ps3, Ps4]

    # the actual drawing of the polygons
    for i in range(4):
        pygame.draw.polygon(surface, cL[minHold[i][0]], PsL[minHold[i][0]], width=0)

    pygame.display.flip()
