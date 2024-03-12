import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import vtk


class Fracture:

    def __init__(self):
        self.M = None
        self.A = None
        self.mp = None
        self.autovalues = None
        self.autovectors = None
        self.axis = None
        self.fig = None

    def load(self, filename):

        dataset = []
        with open(filename, 'r') as file:
            for line in file:
                values = [float(value) for value in line.split()]
                dataset.append(values)
        return np.array(dataset)

    def load_matrix(self):

        self.M = self.load("FRAC0006_nrIter27.txt")
        self.A = np.dot(self.M.T, self.M)

        self.mp = np.mean(self.M, axis=0)

        self.autovalues, self.autovectors = np.linalg.eig(self.A)

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

    def print_plotly(self):
        self.fig = go.Figure()
        # Agregar los autovectores a la figura
        ''' 
        for i in range(len(self.autovalues)):
            autovector = self.autovectors[:, i]
            x = autovector[0] / np.linalg.norm(autovector[0])
            y = autovector[1] / np.linalg.norm(autovector[1])
            z = autovector[2] / np.linalg.norm(autovector[2])

            self.fig.add_trace(go.Scatter3d(
                x=[self.mp[0], self.mp[0] + x],
                y=[self.mp[1], self.mp[1] + y],
                z=[self.mp[2], self.mp[2] + z],
                mode='lines',
                line=dict(color='red'),
                name=f'Autovector {i + 1}'
            ))
            '''

        max_extremos = np.amax((self.M - self.mp) @ self.autovectors, axis=0)
        min_extremos = np.amin((self.M - self.mp) @ self.autovectors, axis=0)

        puntos_caja = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    punto = np.array([max_extremos[0] if i else min_extremos[0],
                                      max_extremos[1] if j else min_extremos[1],
                                      max_extremos[2] if k else min_extremos[2]])
                    # Transformar el punto de vuelta al espacio original
                    puntos_caja.append(self.mp + punto @ self.autovectors.T)

        vertices = [
            [puntos_caja[0], puntos_caja[1], puntos_caja[3], puntos_caja[2], puntos_caja[0]],
            [puntos_caja[4], puntos_caja[5], puntos_caja[7], puntos_caja[6], puntos_caja[4]],
            [puntos_caja[0], puntos_caja[1], puntos_caja[5], puntos_caja[4], puntos_caja[0]],
            [puntos_caja[2], puntos_caja[3], puntos_caja[7], puntos_caja[6], puntos_caja[2]],
            [puntos_caja[1], puntos_caja[3], puntos_caja[7], puntos_caja[5], puntos_caja[1]],
            [puntos_caja[0], puntos_caja[2], puntos_caja[6], puntos_caja[4], puntos_caja[0]]
        ]

        # Crear trazas para cada cara de la caja
        trazas = []
        for vertice in vertices:
            x, y, z = zip(*vertice)
            traza = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(width=2)
            )
            trazas.append(traza)

        # Crear figura y agregar trazas
        self.fig = go.Figure(data=trazas)

        # Agregar puntos a la figura
        self.fig.add_trace(go.Scatter3d(x=self.M[:, 0], y=self.M[:, 1], z=self.M[:, 2], mode='markers',
                                        marker=dict(color='blue', size=5)))

        for i in range(len(self.autovalues)):
            autovector = self.autovectors[:, i]
            punto_inicio = self.mp - autovector
            punto_fin = self.mp + autovector
            self.fig.add_trace(go.Scatter3d(
                x=[punto_inicio[0], punto_fin[0]],
                y=[punto_inicio[1], punto_fin[1]],
                z=[punto_inicio[2], punto_fin[2]],
                mode='lines',
                line=dict(color='red', width=3),
                name=f'Autovector {i + 1}'
            ))


        # Mostrar figura
        self.fig.show()
        '''
         x_min, y_min, z_min = np.min(self.M, axis=0)
        x_max, y_max, z_max = np.max(self.M, axis=0)
        box = [
            # Cara frontal
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)],
            # Cara trasera
            [(x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)],
            # Conectar las caras
            [(x_min, y_min, z_min), (x_min, y_min, z_max)],
            [(x_max, y_min, z_min), (x_max, y_min, z_max)],
            [(x_max, y_max, z_min), (x_max, y_max, z_max)],
            [(x_min, y_max, z_min), (x_min, y_max, z_max)]
        ]

        for face in box:
            x_face = [p[0] for p in face]
            y_face = [p[1] for p in face]
            z_face = [p[2] for p in face]
            self.fig.add_trace(
                go.Scatter3d(x=x_face + [x_face[0]], y=y_face + [y_face[0]], z=z_face + [z_face[0]], mode='lines'))

        # Configurar el diseño de la figura
        # self.fig.update_layout(scene=dict(aspectmode='cube'))

        # Mostrar figura
        self.fig.show()
        '''
