# This script is to create 5 viewpoints (vertices of a regular dodecahedron) around shapes.
import math

if __name__ == '__main__':
    dodecahedron = [[0, 0.5, 1.5],
                    [ -1.1, -0.4, 0.6],
                    [ 1.3, -0.2, 1.1],
                    [0.9, 0, 1.5],
                    [0, -0.1, 0]]

    # get Azimuth, Elevation angles
    # Azimuth varies from -pi to pi
    # Elevation from -pi/2 to pi/2
    view_points = open('./view_points.txt', 'w+')
    for vertice in dodecahedron:
        distance = math.sqrt(vertice[0]**2+vertice[1]**2+vertice[2]**2)
        elevation = math.asin(vertice[2] / distance)
        azimuth = math.atan2(vertice[1], vertice[0])
        view_points.write('%f %f %f %f\n' % (azimuth, elevation, 0., distance))
    view_points.close()


