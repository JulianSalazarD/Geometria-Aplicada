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

        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')

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

        self.axis.legend()
        plt.show()

    def print_point(self):
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')

        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='brown', s=2, label='points')
        self.axis.scatter(self.mp[0], self.mp[1], self.mp[2], c='b', s=2, label='middle point')

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
        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')

        # Mostrar el gráfico
        self.axis.legend()
        plt.show()

    def triangularization(self):
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')

        self.M = self.M[:40]

        points = []
        for i in range(20, len(self.M) + 1, 20):
            points.append(np.array(self.M[i - 20:i]))

        points = np.array(points)

        square = []
        for i in range(0, len(points) - 1):
            aux = []
            for j in range(20):
                if j == 19:
                    aux.append([points[i][j], points[i + 1][j], points[i + 1][0]])
                    aux.append([points[i][j], points[i][0], points[i + 1][j]])
                    # aux.append([points[i][j], points[i][0],
                    #            points[i + 1][j], points[i + 1][j]])
                else:
                    aux.append([points[i][j], points[i + 1][j], points[i + 1][j + 1]])
                    aux.append([points[i][j], points[i][j + 1], points[i + 1][j]])
                    # aux.append([points[i][j], points[i][j + 1],
                    #            points[i + 1][j], points[i + 1][j + 1]])
                square.append(aux)

        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='brown', s=8, label='points')

        square = np.array(square)

        for a in square:
            for s in a:
                self.axis.plot(s[:, 0], s[:, 1], s[:, 2])
                print(s)

        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')

        # Mostrar el gráfico
        self.axis.legend()
        plt.show()

    def print_triangles(self):
        # Crear la figura
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')

        # Graficar los puntos
        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], c='b', marker='o')

        # Graficar las líneas que unen los vértices de los triángulos
        #
        # for simplex in self.tri.simplices:
        #     x = [self.M[simplex[0], 0], self.M[simplex[1], 0], self.M[simplex[2], 0], self.M[simplex[0], 0]]
        #     y = [self.M[simplex[0], 1], self.M[simplex[1], 1], self.M[simplex[2], 1], self.M[simplex[0], 1]]
        #     z = [self.M[simplex[0], 2], self.M[simplex[1], 2], self.M[simplex[2], 2], self.M[simplex[0], 2]]
        #     self.axis.plot(x, y, z, color='r')

        # color = (1, 1, 0, 0.3)
        # Graficar los triángulos
        # self.axis.plot_trisurf(self.M[:, 0], self.M[:, 1], self.M[:, 2], triangles=self.tri.simplices, color=color)

        # Mostrar la figura
        plt.show()

    def norm(self):
        self.normal = []
        for simplex in self.tri.simplices:
            p0, p1, p2, p3 = self.M[simplex]
            normal_012 = np.cross(p1 - p0, p2 - p0)
            normal_013 = np.cross(p1 - p0, p3 - p0)
            normal_023 = np.cross(p2 - p0, p3 - p0)
            normal_123 = np.cross(p2 - p1, p3 - p1)

            n = (normal_012 + normal_013 + normal_023 + normal_123) / 4.0
            n = n / np.linalg.norm(n)
            self.normal.append(n)

    def mainloop(self):
        """
        Método principal para cargar datos, calcular propiedades y visualizar datos.
        """
        self.load("FRAC0003_nrIter4.txt")
        self.load_matrix()
        # self.get_traces()
        # self.print_matplotlib()
        # self.get_box()
        # self.print_point()
        self.triangularization()
