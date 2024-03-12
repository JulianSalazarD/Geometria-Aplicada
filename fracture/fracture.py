import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Fracture:

    def __init__(self):
        self.M = None
        self.A = None
        self.mp = None
        self.autovalues = None
        self.autovectors = None
        self.axis = None
        self.fig = None
        self.traces = None

    def load(self, filename):

        dataset = []
        with open(filename, 'r') as file:
            for line in file:
                values = [float(value) for value in line.split()]
                dataset.append(values)
        self.M = np.array(dataset)

    def load_matrix(self):

        self.A = np.dot(self.M.T, self.M)

        self.mp = np.mean(self.M, axis=0)

        self.autovalues, self.autovectors = np.linalg.eig(self.A)

    def get_traces(self):
        max_p = np.amax(np.dot((self.M - self.mp), self.autovectors), axis=0)
        min_p = np.amin(np.dot((self.M - self.mp), self.autovectors), axis=0)

        # crear vertices y drireccion de la caja
        box = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    points = np.array([max_p[0] if i else min_p[0],
                                       max_p[1] if j else min_p[1],
                                       max_p[2] if k else min_p[2]])
                    # Transformar el punto de vuelta al espacio original
                    box.append(self.mp + np.dot(points, self.autovectors.T))

        vertex = [
            [box[0], box[1], box[3], box[2], box[0]],
            [box[4], box[5], box[7], box[6], box[4]],
            [box[0], box[1], box[5], box[4], box[0]],
            [box[2], box[3], box[7], box[6], box[2]],
            [box[1], box[3], box[7], box[5], box[1]],
            [box[0], box[2], box[6], box[4], box[0]]
        ]

        # Crear trazas para cada cara de la caja
        self.traces = []
        for vertice in vertex:
            x, y, z = zip(*vertice)
            trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(width=2))
            self.traces.append(trace)

    def print_plotly(self):

        self.fig = go.Figure(data=self.traces)

        # Agregar puntos a la figura
        self.fig.add_trace(go.Scatter3d(x=self.M[:, 0], y=self.M[:, 1], z=self.M[:, 2], mode='markers',
                                        marker=dict(color='blue', size=2), name='points'))

        # graficar autovectores
        for i in range(len(self.autovalues)):
            autovector = self.autovectors[:, i]
            ip = self.mp
            ep = self.mp + autovector
            self.fig.add_trace(go.Scatter3d(
                x=[ip[0], ep[0]],
                y=[ip[1], ep[1]],
                z=[ip[2], ep[2]],
                mode='lines',
                line=dict(color='red', width=3),
                name=f'Autovector {i + 1}'
            ))

        # Mostrar figura
        self.fig.show()

    def mainloop(self):
        self.load("FRAC0006_nrIter27.txt")
        self.load_matrix()
        self.get_traces()
        self.print_plotly()

    def print_matplotlib(self):

        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')
        # Graficar los puntos
        self.axis.scatter(self.M[:, 0], self.M[:, 1], self.M[:, 2], s=1, label='Points')

        # Graficar los autovectores centrados en el punto promedio

        for i in range(len(self.autovalues)):
            autovector = self.autovectors[:, i]
            plt.quiver(self.mp[0], self.mp[1], self.mp[2], autovector[0], autovector[1], autovector[2],
                       color='r', label=f'Autovector {i + 1}')

        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')
        plt.axis('equal')
        self.axis.legend()
        plt.show()
