import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt

use('Qt5Agg')


class Fracture:
    """
        Clase para cargar datos de fracturas, calcular propiedades y visualizar en 3D utilizando Plotly       """

    def __init__(self):
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

        self.tri alamacena la dirección de los puntos que fforman un triangulo
        """

        points = [p for p in range(20)]
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
        """
        points = []
        for i in range(20, len(self.M) + 1, 20):
            points.append(np.array(self.M[i - 20:i]))

        points = np.array(points)

        self.tri = []
        for i in range(0, len(points) - 1):
            for j in range(20):
                if j == 19:
                    self.tri.append([points[i][j], points[i + 1][j], points[i + 1][0]])
                    self.tri.append([points[i][j], points[i][0], points[i + 1][j]])
                    # aux.append([points[i][j], points[i][0],
                    #            points[i + 1][j], points[i + 1][j]])
                else:
                    self.tri.append([points[i][j], points[i + 1][j], points[i + 1][j + 1]])
                    self.tri.append([points[i][j], points[i][j + 1], points[i + 1][j]])
                    # aux.append([points[i][j], points[i][j + 1],
                    #            points[i + 1][j], points[i + 1][j + 1]])

        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='brown', s=8, label='points')

        self.tri = np.array(self.tri)
        """

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
            x = [mp[0], v[0]]
            y = [mp[1], v[1]]
            z = [mp[2], v[2]]
            self.axis.quiver(mp[0], mp[1], mp[2], v[0], v[1], v[2], color='b')

    def mainloop(self):
        """
        Método principal para cargar datos, calcular propiedades y visualizar datos.
        """
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')
        self.load("FRAC0003_nrIter4.txt")
        self.load_matrix()
        self.get_box()
        self.print_point()
        self.triangularization()
        self.norm()
        self.print_triangles()
        # self.surface()
        self.print_norm()

        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')

        self.axis.legend()
        plt.xlim(5, 15)
        plt.ylim(-5, 5)
        plt.show()
