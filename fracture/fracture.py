import random
import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
from sympy import symbols, Eq, Point3D, Plane, solve, Line3D
import pandas as pd

use('Qt5Agg')


class Fracture:
    """
        Clase para cargar datos de fracturas, calcular propiedades y visualizar en 3D utilizando Plotly       """

    def __init__(self):
        self.inter = None
        self.color_t = None
        self.caras = None
        self.vertex = None
        self.normal = None
        self.M = None
        self.A = None
        self.mp = None
        self.autovalues = None
        self.autovectors = None
        self.axis = None
        self.fig = None
        self.traces = None
        self.tri = None
        self.color_m = None
        self.size = None

    def load(self, filename):
        """
        Carga los datos de la fractura desde un archivo de texto.

        :param filename: Nombre del archivo de texto.
        """

        dataset = []
        with open(filename, 'r') as file:
            for line in file:
                values = [float(value) for value in line.split()]
                dataset.append(values)
        self.M = np.array(dataset)

    def load_matrix(self):
        """
        Calcula la matriz A (matriz de covarianza) de una matriz M y
        los autovalores y autovectores
        """
        print(len(self.M))
        self.M = self.M[400:440]

        self.A = np.dot(self.M.T, self.M)

        self.mp = np.mean(self.M, axis=0)

        self.autovalues, self.autovectors = np.linalg.eig(self.A)

    def get_box(self):
        """
        Calcula los vértices de la caja que encierra la fractura (matriz m)
        """
        points = self.M - self.mp
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(points, rowvar=False))
        points = np.dot(points, eigenvectors)

        # Calcula los límites de la caja en el nuevo sistema de coordenadas
        min_limits = np.min(points, axis=0)
        max_limits = np.max(points, axis=0)

        # Crear puntos de los límites de la caja en el sistema transformado
        self.vertex = np.array(np.meshgrid([min_limits[0], max_limits[0]], [min_limits[1], max_limits[1]],
                                           [min_limits[2], max_limits[2]])).T.reshape(-1, 3)

        # Transformar los vértices de la caja de vuelta al sistema original
        self.vertex = np.dot(self.vertex, eigenvectors.T) + self.mp

    def color_points(self):
        self.color_m = []
        self.size = int(len(self.M) / 20)
        for i in range(self.size):
            r = [random.random(), random.random(), random.random()]
            self.color_m.append(r)

        self.color_m = np.array(self.color_m)

    def print_point(self):
        """
        Representación de los datos en 3D utilizando Matplotlib
        """
        for i in range(self.size):
            m = self.M[i * 20:i * 20 + 20]
            self.axis.scatter(m[:, 0], m[:, 1], m[:, 2], color=self.color_m[i], s=2)
        # self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], color=(0, 0, 0), s=2, label='points')
        # self.axis.scatter(self.mp[0], self.mp[1], self.mp[2], c='b', s=2, label='middle point')

        # Graficar autovectores
        # self.axis.quiver(*self.mp, *self.autovectors, color='pink')
        # self.axis.scatter(self.vertex[:, 0], self.vertex[:, 1], self.vertex[:, 2], color='blue', s=10)

    def build_box(self):
        lines = [
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5],
            [4, 6], [5, 7], [6, 7]
        ]
        for line in lines:
            self.axis.plot(
                [self.vertex[line[0], 0], self.vertex[line[1], 0]],
                [self.vertex[line[0], 1], self.vertex[line[1], 1]],
                [self.vertex[line[0], 2], self.vertex[line[1], 2]],
                color='black'
            )

    def triangularization(self):
        """
         Realizar la triangulación de los puntos, teniento en cuenta que cada 20 puntos se apueden realizar
         triangulaiones con los siguientes 2o puntos.

        self.tri alamacena la dirección de los puntos que forman un triangulo
        """
        self.color_t = []
        self.tri = []
        for i in range(0, self.size - 1):
            for j in range(20):
                if j == 19:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), 0 + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), 0 + (20 * (i + 1)), 0 + (20 * i)])
                else:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), (j + 1) + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), (j + 1) + (20 * (i + 1)), (j + 1) + (20 * i)])

                color = (self.color_m[i] + 2 * self.color_m[i + 1]) / 3
                self.color_t.append(color)
                color = (2 * self.color_m[i] + self.color_m[i + 1]) / 3
                self.color_t.append(color)
        self.color_t = np.array(self.color_t)
        self.tri = np.array(self.tri)

    def norm(self):
        """
        sacar la norma de cada triangulo
        """

        self.normal = []
        aux = 0
        for triangle in self.tri:
            p1, p2, p3 = self.M[triangle]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            if n.all() == 0:
                v1 = p3 - p2
                v2 = p1 - p2
                n = np.cross(v1, v2)
            if n.all() == 0:
                v1 = p1 - p3
                v2 = p2 - p3
                n = np.cross(v1, v2)
            if n.all() == 0:
                n = np.cross(v1, v2)
                print(n, "\n", triangle, p1, p2, '\n')
            self.normal.append(n / np.linalg.norm(n))

    def triangles_color(self):
        self.color_t = []
        for i in range(self.size - 1):
            r = (self.color_m[i] + self.color_m[i + 1]) / 2
            self.color_t.append(r)
        self.color_t = np.array(self.color_t)

    def print_triangles(self):
        """
        graficar triangularizacion
        """
        for s in range(len(self.tri)):
            triangle = np.array([self.M[self.tri[s][0]], self.M[self.tri[s][1]], self.M[self.tri[s][2]],
                                 self.M[self.tri[s][0]]])
            self.axis.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=self.color_t[s])
        ''' 
        for s in self.tri:
            triangle = np.array([self.M[s[0]], self.M[s[1]], self.M[s[2]], self.M[s[0]]])
            self.axis.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=self.color_t)
        '''

    def surface(self):
        self.axis.plot_trisurf(self.M[:, 0], self.M[:, 1], self.M[:, 2],
                               triangles=self.tri, color='lightpink')

    def print_norm(self):
        """
        graficar la norma de los triangulos
        """
        for i in range(len(self.normal)):
            mp = np.mean(self.M[self.tri[i]], axis=0)
            v = 0.5 * self.normal[i]
            self.axis.quiver(mp[0], mp[1], mp[2], v[0], v[1], v[2], color='b')

    def points_in_middle(self, punto, punto_inicio, punto_fin):
        """
           Comprueba si un punto está dentro de un rango definido por dos puntos en cada eje.

           Args:
           - punto: Punto a comprobar.
           - punto_inicio: Punto inicial del rango.
           - punto_fin: Punto final del rango.

           Returns:
           - bool: True si el punto está dentro del rango en todos los ejes, False en caso contrario.
           """
        dentro_x = (punto_inicio[0] <= punto.x <= punto_fin[0]) or (punto_fin[0] <= punto.x <= punto_inicio[0])
        dentro_y = (punto_inicio[1] <= punto.y <= punto_fin[1]) or (punto_fin[1] <= punto.y <= punto_inicio[1])
        dentro_z = (punto_inicio[2] <= punto.z <= punto_fin[2]) or (punto_fin[2] <= punto.z <= punto_inicio[2])
        return dentro_x and dentro_y and dentro_z

    def direction_points(self, punto_inicial, vector_director, distancia):
        """
           Calcula un punto final dado un punto inicial, un vector director y una distancia.

           Args:
           - punto_inicial: Punto inicial.
           - vector_director: Vector director.
           - distancia: Distancia desde el punto inicial al punto final.

           Returns:
           - Punto final calculado.
           """
        norma_vector = np.linalg.norm(vector_director)
        direccion = vector_director / norma_vector
        punto_final = punto_inicial + distancia * direccion
        return punto_final

    def plane_from_points(self):
        """
           Calcula los planos a partir de puntos dados y los guarda en la lista.
        """
        self.caras = []
        points = [[0, 1, 2], [0, 1, 4], [0, 4, 2], [7, 5, 3], [7, 6, 3], [7, 6, 5]]
        points = [[0, 1, 4], [7, 6, 5], [7, 6, 3], [0, 4, 2], [7, 5, 3], [0, 1, 2]]
        for p in points:
            p1, p2, p3 = map(Point3D, (self.vertex[p[0]], self.vertex[p[1]], self.vertex[p[2]]))
            self.caras.append(Plane(p1, p2, p3))

    def line_from_point_and_vector(self, point, vector):
        """
            Crea una línea a partir de un punto y un vector.

            Args:
            - point: Punto inicial de la línea.
            - vector: Vector director de la línea.

            Returns:
            - Línea creada.
        """
        point = Point3D(point)
        direction_point = point + Point3D(*vector)
        line = Line3D(point, direction_point)
        return line

    def intersection_plane_line(self, line):
        """
            Calcula la intersección entre un plano y una línea.

            Args:
            - line: Línea con la que se calcula la intersección.

            Returns:
            - Lista de puntos de intersección, si hay alguno, o None en caso contrario.
        """
        inter = []
        for plane in self.caras:
            intersection = plane.intersection(line)
            if len(intersection) == 1:  # Solo hay un punto de intersección
                # intersection_point = intersection[0]
                inter.append(intersection)
                # return inter
        if len(inter) > 0:
            return inter
        else:
            return None

    def intersection(self):
        """
        Calcula las intersecciones entre las normales de los triángulos y la caja, y guarda los puntos en la lista
        inter.
        """
        for i in self.normal:
            print(i)
        self.inter = []
        self.plane_from_points()
        aux = 0
        for i in range(len(self.normal)):
            mp = np.mean(self.M[self.tri[i]], axis=0)
            v = self.normal[i]
            line = self.line_from_point_and_vector(mp, v)
            op = self.direction_points(mp, v, 1)
            p = self.intersection_plane_line(line)
            print(aux)
            aux += 1
            if p is None: continue
            for pp in p:
                pp = [pp[0].x, pp[0].y, pp[0].z]
                self.inter.append(pp)
        self.inter = np.array(self.inter)
        print('len', len(self.inter), len(self.normal))

    def print_intersection(self):
        self.color_t = self.color_t[:len(self.inter)]
        self.axis.scatter(self.inter[:, 0], self.inter[:, 1], self.inter[:, 2], color=self.color_t, s=2)

    def mainloop(self):
        """
        Método principal para cargar datos, calcular propiedades y visualizar datos.
        """
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')
        self.load("FRAC0006_nrIter27.txt")
        self.load_matrix()
        self.get_box()
        self.build_box()
        self.color_points()
        # self.print_point()
        self.triangularization()
        self.norm()
        # self.print_triangles()
        # self.surface()
        # self.print_norm()
        # self.box_planes()
        self.intersection()
        self.print_intersection()
        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')

        self.axis.legend()
        # plt.xlim(5, 15)
        # plt.ylim(-5, 5)
        plt.show()
