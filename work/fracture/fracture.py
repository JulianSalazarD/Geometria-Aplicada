# Librerias
import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd
from skspatial.objects import Plane, Point, Vector, Line
from skspatial.plotting import plot_3d


use('Qt5Agg')

class Fracture:
    def __init__(self, M):
        self.inter = None
        self.color_t = None
        self.caras = None
        self.vertex = None
        self.normal = None
        self.M = M
        self.A = None
        self.mp = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.traces = None
        self.tri = None
        self.color_m = None
        self.size = None
        self.max_value = 0
        self.x_axis = None
        self.y_axis = None
        self.M_size = None
        self.face = []
        self.normal_vertex = None
        self.points = np.array(
                    [[0, 1, 2, 3],
                    [0, 1, 4, 5],
                    [0, 4, 2, 6],
                    [7, 5, 3, 1],
                    [7, 6, 3, 2],
                    [7, 6, 5, 4]]
                    )
        self.isdegenerate = None
        self.load_data()
        self.build_fracture()

    
    def load_data(self):
        # Carga los datos de la fractura desde un archivo de texto.

        # dataset = []
        # with open(filename, 'r') as file:
        #     for line in file:
        #         values = [float(value) for value in line.split()]
        #         dataset.append(values)
        #         self.M = np.array(dataset)
        self.M = np.array(self.M)
        self.M_size = self.M.shape

    @staticmethod
    def new_plot():
        fig = plt.figure()
        return fig.add_subplot(111, projection='3d')
    
    def config_plot(self, axis, title):
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        plt.title(title)
        plt.xlim(self.x_axis - self.max_value, self.x_axis + self.max_value)
        plt.ylim(self.y_axis - self.max_value, self.y_axis + self.max_value)   

    
    def build_box(self, axis):
        lines = [
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5],
            [4, 6], [5, 7], [6, 7]
        ]
        for line in lines:
            axis.plot(
                [self.vertex[line[0], 0], self.vertex[line[1], 0]],
                [self.vertex[line[0], 1], self.vertex[line[1], 1]],
                [self.vertex[line[0], 2], self.vertex[line[1], 2]],
                color='black'
            )

    def limits(self):
        min_limits = np.min(self.M, axis=0)
        max_limits = np.max(self.M, axis=0)

        for i in range(3):
            self.max_value = max(self.max_value, abs(max_limits[i] - min_limits[i]))
        self.max_value/=2
        self.x_axis = np.min(self.M[:, 0]) + self.max_value/2
        self.y_axis = np.min(self.M[:, 1]) + self.max_value/2  

    def load_matrix(self):
        """
        Calcula la matriz A (matriz de covarianza) de una matriz M y
        los autovalores y autovectores
        """
        self.A = np.dot(self.M.T, self.M)
        self.mp = np.mean(self.M, axis=0)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)

    def get_box(self):
        """
        Calcula los vértices de la caja que encierra la fractura (matriz m)
        """

        points = self.M - self.mp
        _eigenvalues, _eigenvectors = np.linalg.eig(np.cov(points, rowvar=False))
        points = np.dot(points, _eigenvectors)

        # Calcula los límites de la caja en el nuevo sistema de coordenadas
        min_limits = np.min(points, axis=0)
        max_limits = np.max(points, axis=0)

        # Crear puntos de los límites de la caja en el sistema transformado
        self.vertex = np.array(np.meshgrid([min_limits[0], max_limits[0]], [min_limits[1], max_limits[1]],
                                            [min_limits[2], max_limits[2]])).T.reshape(-1, 3)

        # Transformar los vértices de la caja de vuelta al sistema original
        self.vertex = np.dot(self.vertex, _eigenvectors.T) + self.mp

    def color_points(self):
        self.size = int(self.M_size[0] / 20)
        self.color_m = np.empty((0, 3))
        
        for _ in range(self.size):
            color = np.random.rand(1, 3)
            self.color_m = np.concatenate((self.color_m, np.full((20, 3), color)))

    def plot_points(self):
        axis = Fracture.new_plot()
        self.build_box(axis)

        axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], color=self.color_m, s=2)

        self.config_plot(axis, "Points")

    def triangularization(self):
        """
            Realizar la triangulación de los puntos, teniento en cuenta que cada 20 puntos se apueden realizar
            triangulaiones con los siguientes 20 puntos.

            la variable tri alamacena la dirección de los puntos que forman un triangulo
        """
        self.color_t = []
        self.tri = []
        for i in range(0, self.size - 1):
            for j in range(20):
                if j == 19:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), 0 + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), 0 + (20 * (i + 1)), 0 + (20 * i)])
                    
                    self.color_t.append((self.color_m[j + (20 * i)] + 
                                         self.color_m[j + (20 * (i + 1))] + 
                                         self.color_m[ 0 + (20 * (i + 1))]) / 3)
                    
                    self.color_t.append((self.color_m[j + (20 * i)] + 
                                        self.color_m[j + (20 * (i + 1))] + 
                                        self.color_m[ 0 + (20 * i)]) / 3)
                else:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), (j + 1) + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), (j + 1) + (20 * (i + 1)), (j + 1) + (20 * i)])

                    self.color_t.append((self.color_m[j + (20 * i)] + 
                                         self.color_m[j + (20 * (i + 1))] + 
                                         self.color_m[ (j + 1) + (20 * (i + 1))]) / 3)
                    
                    self.color_t.append((self.color_m[j + (20 * i)] + 
                                        self.color_m[(j + 1) + (20 * (i + 1))] + 
                                        self.color_m[(j + 1) + (20 * i)]) / 3)
                    

        self.color_t = np.array(self.color_t)
        self.tri = np.array(self.tri)


    def plot_triangles(self):
        axis = Fracture.new_plot()
        self.build_box(axis)

        for s in range(self.tri.shape[0]):
            triangle = np.array([self.M[self.tri[s][0]], self.M[self.tri[s][1]], self.M[self.tri[s][2]], self.M[self.tri[s][0]]])
            axis.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=self.color_t[s])

        self.config_plot(axis, "Triangles")

    def norm(self):
        self.isdegenerate = []
        self.normal = np.zeros(((self.size - 1)* 40, 3))
        aux = 1
        for i, triangle in enumerate(self.tri):
            p1, p2, p3 = self.M[triangle]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            self.normal[i] = n / np.linalg.norm(n)
            if np.isnan(self.normal[i]).any():
                self.isdegenerate.append(i)
                
                random_vector = np.random.rand(3)

                random_vector /= np.linalg.norm(random_vector)

                self.normal[i] = aux * self.normal[i - 1] / np.abs(np.dot(random_vector, self.normal[i - 1]))

                self.normal[i] /= np.linalg.norm(self.normal[i])

            if i > 0:
                aux = np.dot(self.normal[i], self.normal[i - 1])

    def plot_normals(self):
        axis = Fracture.new_plot()
        self.build_box(axis)

        for i in range(self.normal.shape[0]):
            mp = np.mean(self.M[self.tri[i]], axis=0)
            v = 0.3 * self.normal[i]
            axis.quiver(mp[0], mp[1], mp[2], v[0], v[1], v[2], color='b')

        self.config_plot(axis, "Normals")

    def plane_from_points(self):
        """
            Calcula los planos a partir de puntos dados y los guarda en la lista.
        """
        for p in self.points:
            p1, p2, p3 = self.vertex[p[0]], self.vertex[p[1]], self.vertex[p[2]]
            self.face.append(Plane.from_points(p1,p2,p3))

    def get_normal_vertex(self):
        self.normal_vertex = [[] for _ in range(self.M.shape[0])]
        for i, n in enumerate(self.normal):
            for j in self.tri[i]:
                self.normal_vertex[j].append(n)

        for i in range(len(self.normal_vertex)):
            self.normal_vertex[i] = np.mean(self.normal_vertex[i], axis=0)

        self.normal_vertex = np.array(self.normal_vertex)

    @staticmethod
    def make_line(point, vector):
        """
            Crea una línea a partir de un punto y un vector.

            Args:
            - point: Punto inicial de la línea.
            - vector: Vector director de la línea.

            Returns:
            - Líne
        """
        return Line(point=point, direction=vector)
    

    @staticmethod
    def is_direction(pi, pf, vd):
        """
            Verifica si un punto final está en la dirección de un vector dado desde un punto inicial.
        """
        vr = pf - pi
        return np.dot(vr, vd) > 0
    
    @staticmethod
    def close_intersection(pi, intersections):
        """
            Encuentra la intersección más cercana a un punto inicial desde una lista de intersecciones.
        """
        intersect = None
        min_dist = float('inf')
        point = Point(pi)
        for intersection in intersections:
            dist = point.distance_point(intersection)
            if dist < min_dist:
                min_dist = dist
                intersect = intersection
            
        return intersect
    
    def intersection_plane_line(self, line, pi, vd):
        """
            Encuentra la intersección más cercana de una línea con varias caras (planos).
        """
        intersections = []
        for plane in self.face:
            point = plane.intersect_line(line)
            if point is not None and Fracture.is_direction(pi, point, vd):
                intersections.append(point)
        
        return Fracture.close_intersection(pi, intersections)
    
    def get_intersections(self):
        """
            Calcula las intersecciones más cercanas de líneas definidas por puntos medios de triángulos y sus normales con las caras de la caja.

            Utiliza las normales de los triángulos y los puntos medios de los triángulos para definir las líneas.
            Luego, encuentra la intersección más cercana de cada línea con las caras de la caja.
            Las intersecciones se almacenan en una variable global `inter`.
        """

        self.inter = []
        for i in range(self.normal_vertex.shape[0]):
            mp = self.M[i] # np.mean(self.M[self.tri[i]], axis=0)
            v = self.normal_vertex[i]
            if np.any(np.isnan(v)):
                self.inter.append(v)
                continue
            line = Fracture.make_line(mp, v)
            self.inter.append(self.intersection_plane_line(line, mp, v))
        
        self.inter = np.array(self.inter)

    def plot_intersections(self):
        """
            Añade los puntos de intersección al gráfico.
        """
        axis = Fracture.new_plot()
        self.build_box(axis)
        
        for i in range(self.size):
            m = self.inter[i * 20:i * 20 + 20]
            axis.scatter(m[:, 0], m[:, 1], m[:, 2], color=self.color_m[i], s=2)

        self.config_plot(axis, "Intersections")

    def plot_surface_triangles(self):
        axis = Fracture.new_plot()
        self.build_box(axis)
        
        for s in range(len(self.tri)):
            triangle = np.array([self.inter[self.tri[s][0]], self.inter[self.tri[s][1]], self.inter[self.tri[s][2]], self.inter[self.tri[s][0]]])
            axis.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=self.color_t[s])

        self.config_plot(axis, "surface Triangles")

    def build_fracture(self):
        self.limits()
        self.load_matrix()
        self.get_box()
        self.color_points()
        self.triangularization()
        self.norm()
        self.get_normal_vertex()
        self.plane_from_points()
        self.get_intersections()

        # plt.show()}

    def print_fracture(self, axis):
        self.print_squares(axis)
        axis.scatter(self.mp_s[:, 0], self.mp_s[:, 1], self.mp_s[:, 2],s=1, color=self.color_s)
        self.config_plot(axis, "Squares")
