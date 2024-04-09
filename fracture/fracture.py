import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
from sympy import symbols, Eq, Point3D, Plane, solve

use('Qt5Agg')


class Fracture:
    """
        Clase para cargar datos de fracturas, calcular propiedades y visualizar en 3D utilizando Plotly       """

    def __init__(self):
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
        # self.M = self.M[:60]
        self.A = np.dot(self.M.T, self.M)

        self.mp = np.mean(self.M, axis=0)

        self.autovalues, self.autovectors = np.linalg.eig(self.A)

    def get_traces(self):
        """
        Calcula los vértices de la caja que encierra la fractura (matriz m)
        """
        max_p = np.amax(np.dot((self.M - self.mp), self.autovectors), axis=0)
        min_p = np.amin(np.dot((self.M - self.mp), self.autovectors), axis=0)

        # Crear vertices y dirección de la caja
        box = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    points = np.array([max_p[0] if i else min_p[0],
                                       max_p[1] if j else min_p[1],
                                       max_p[2] if k else min_p[2]])
                    box.append(self.mp + np.dot(points, self.autovectors.T))

        self.vertex = [
            [box[0], box[1], box[3], box[2], box[0]],
            [box[4], box[5], box[7], box[6], box[4]],
            [box[0], box[1], box[5], box[4], box[0]],
            [box[2], box[3], box[7], box[6], box[2]],
            [box[1], box[3], box[7], box[5], box[1]],
            [box[0], box[2], box[6], box[4], box[0]]
        ]

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

    def print_matplotlib(self):
        """
        Representación de los datos en 3D utilizando Matplotlib
        """

        # Dibujar caja
        for vert in self.vertex:
            v = np.array(vert)
            x, y, z = v[:, 0], v[:, 1], v[:, 2]
            self.axis.plot(x, y, z, color='g')

        # Agregar puntos a la figura
        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='b', s=2, label='points')

        # Graficar autovectores
        for i in range(len(self.autovalues)):
            autovector = self.autovectors[:, i]
            ip = self.mp
            ep = self.mp + autovector
            self.axis.plot([ip[0], ep[0]], [ip[1], ep[1]], [ip[2], ep[2]], color='red', linewidth=3,
                           label=f'Autovector {i + 1}')

    def print_point(self):
        """
        Representación de los datos en 3D utilizando Matplotlib
        """
        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='brown', s=2, label='points')
        # self.axis.scatter(self.mp[0], self.mp[1], self.mp[2], c='b', s=2, label='middle point')

        # Graficar autovectores
        # self.axis.quiver(*self.mp, *self.autovectors, color='pink')
        # self.axis.scatter(self.vertex[:, 0], self.vertex[:, 1], self.vertex[:, 2], color='blue', s=10)

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
        self.tri = []
        size = int(len(self.M) / 20)
        for i in range(0, size - 1):
            for j in range(20):
                if j == 19:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), 0 + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), 0 + (20 * (i + 1)), 0 + (20 * i)])
                else:
                    self.tri.append([j + (20 * i), j + (20 * (i + 1)), (j + 1) + (20 * (i + 1))])
                    self.tri.append([j + (20 * i), (j + 1) + (20 * (i + 1)), (j + 1) + (20 * i)])

        self.tri = np.array(self.tri)

    def norm(self):
        """
        sacar la norma de cada triangulo
        """
        self.normal = []
        for triangle in self.tri:
            p1, p2, p3 = self.M[triangle]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            self.normal.append(n / np.linalg.norm(n))

    def print_triangles(self):
        """
        graficar triangularizacion
        :return:
        """
        for s in self.tri:
            triangle = np.array([self.M[s[0]], self.M[s[1]], self.M[s[2]], self.M[s[0]]])
            self.axis.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2])

    def surface(self):
        self.axis.plot_trisurf(self.M[:, 0], self.M[:, 1], self.M[:, 2],
                               triangles=self.tri, color='lightpink')

    def print_norm(self):
        """
        graficar la norma de los triangulos
        """
        for i in range(len(self.normal)):
            mp = np.mean(self.M[self.tri[i]], axis=0)
            v = 0.1 * self.normal[i]
            self.axis.quiver(mp[0], mp[1], mp[2], v[0], v[1], v[2], color='b')

    def ecuacion_plano(self, p1, p2, p3):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)

        normal = np.cross(v1, v2)

        A, B, C = normal

        D = -np.dot(normal, np.array(p1))

        return f"{A}x + {B}y + {C}z + {D} = 0"

    def eq_box(self, vertices):
        x, y, z = symbols('x y z')

        x_min = min(v[0] for v in vertices)
        x_max = max(v[0] for v in vertices)
        y_min = min(v[1] for v in vertices)
        y_max = max(v[1] for v in vertices)
        z_min = min(v[2] for v in vertices)
        z_max = max(v[2] for v in vertices)

        caras = [
            Eq(x, x_min),
            Eq(x, x_max),
            Eq(y, y_min),
            Eq(y, y_max),
            Eq(z, z_min),
            Eq(z, z_max)
        ]

        return caras

    def points_in_box(self, punto, vertices_caja):
        x, y, z = punto
        x_coords, y_coords, z_coords = zip(*vertices_caja)

        dentro_x = min(x_coords) <= x <= max(x_coords)
        dentro_y = min(y_coords) <= y <= max(y_coords)
        dentro_z = min(z_coords) <= z <= max(z_coords)

        return dentro_x and dentro_y and dentro_z

    def box_intersection(self, punto_inicio, direccion):
        x, y, z, t = symbols('x y z t')

        # Representa el vector como una línea paramétrica
        linea = [punto_inicio[i] + t * direccion[i] for i in range(3)]

        # Encuentra los puntos de intersección con cada cara
        intersecciones = []
        for cara in self.caras:
            interseccion = solve(cara.subs({x: linea[0], y: linea[1], z: linea[2]}), t)
            intersecciones.extend(
                [linea[0].subs(t, t_val), linea[1].subs(t, t_val), linea[2].subs(t, t_val)] for t_val in interseccion if
                t_val.is_real)


        # puntos_interseccion = [punto for punto in intersecciones if self.punto_dentro_de_caja(punto, vertices_caja)]
        puntos_interseccion = intersecciones

        return puntos_interseccion

    def points_in_middle(self, punto, punto_inicio, punto_fin):
        dentro_x = (punto_inicio[0] <= punto[0] <= punto_fin[0]) or (punto_fin[0] <= punto[0] <= punto_inicio[0])
        dentro_y = (punto_inicio[1] <= punto[1] <= punto_fin[1]) or (punto_fin[1] <= punto[1] <= punto_inicio[1])
        dentro_z = (punto_inicio[2] <= punto[2] <= punto_fin[2]) or (punto_fin[2] <= punto[2] <= punto_inicio[2])
        return dentro_x and dentro_y and dentro_z

    def direction_points(self, punto_inicial, vector_director, distancia):
        norma_vector = np.linalg.norm(vector_director)
        direccion = vector_director / norma_vector
        punto_final = punto_inicial + distancia * direccion
        return punto_final

    def box_planes(self):

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

        normal = self.normal
        tri = self.tri

        self.caras = self.eq_box(self.vertex)

        inter = []
        for i in range(len(normal)):
            mp = np.mean(self.M[tri[i]], axis=0)
            v = normal[i]
            p = self.box_intersection(mp, v)
            op = self.direction_points(mp, v, 10)
            for pp in p:
                if self.points_in_middle(pp, mp, op):
                    inter.append(pp)
                    self.axis.scatter(pp[0], pp[1], pp[2], c='brown', s=2)
            #print(p)
            #inter.append(p)
            #print(p)

            # self.axis.scatter(p[0][0], p[0][1], p[0][2], c='brown', s=2)


    def mainloop(self):
        """
        Método principal para cargar datos, calcular propiedades y visualizar datos.
        """
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')
        self.load("FRAC0003_nrIter4.txt")
        self.load_matrix()
        self.get_box()
        # self.print_point()
        self.triangularization()
        self.norm()
        # self.print_triangles()
        # self.surface()
        # self.print_norm()
        self.box_planes()
        # self.part3()
        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')

        self.axis.legend()
        # plt.xlim(5, 15)
        # plt.ylim(-5, 5)
        plt.show()
