from fracture.fracture import Fracture

if __name__ == '__main__':
    f = Fracture().mainloop()

    ''' 
    from matplotlib import use
    use('Qt5Agg')
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import Delaunay

    # Generar una nube de puntos aleatorios en 3D
    np.random.seed(0)
    n_puntos = 100
    puntos = np.random.randn(n_puntos, 3)

    # Triangulación de Delaunay
    triangulacion = Delaunay(puntos)

    # Crear la figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c='b', marker='o')

    # Graficar las líneas que unen los vértices de los triángulos
    for simplex in triangulacion.simplices:
        x = [puntos[simplex[0], 0], puntos[simplex[1], 0], puntos[simplex[2], 0], puntos[simplex[0], 0]]
        y = [puntos[simplex[0], 1], puntos[simplex[1], 1], puntos[simplex[2], 1], puntos[simplex[0], 1]]
        z = [puntos[simplex[0], 2], puntos[simplex[1], 2], puntos[simplex[2], 2], puntos[simplex[0], 2]]
        ax.plot(x, y, z, color='r')

    # Mostrar la figura
    plt.show()
    '''

    ''' 
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import Delaunay

    # Generar una nube de puntos aleatorios en 3D
    np.random.seed(0)
    n_puntos = 100
    puntos = np.random.randn(n_puntos, 3)

    # Triangulación de Delaunay
    triangulacion = Delaunay(puntos)

    # Crear la figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c='b', marker='o')

    # Graficar los triángulos
    ax.plot_trisurf(puntos[:, 0], puntos[:, 1], puntos[:, 2], triangles=triangulacion.simplices, color='lightpink')

    # Mostrar la figura
    plt.show()
    '''

    ''' 
    import numpy as np
    import plotly.graph_objects as go
    from scipy.spatial import Delaunay

    # Generar una nube de puntos aleatorios en 3D
    np.random.seed(0)
    n_puntos = 6
    puntos = np.random.randn(n_puntos, 3)

    # Triangulación de Delaunay
    triangulacion = Delaunay(puntos)

    # Crear figura
    fig = go.Figure()

    # Añadir puntos de datos
    fig.add_trace(go.Scatter3d(x=puntos[:, 0], y=puntos[:, 1], z=puntos[:, 2], mode='markers', marker=dict(size=3)))
    

    # Añadir líneas que unen los vértices de los triángulos
    for simplex in triangulacion.simplices:
        x = [puntos[simplex[0], 0], puntos[simplex[1], 0], puntos[simplex[2], 0], puntos[simplex[0], 0]]
        y = [puntos[simplex[0], 1], puntos[simplex[1], 1], puntos[simplex[2], 1], puntos[simplex[0], 1]]
        z = [puntos[simplex[0], 2], puntos[simplex[1], 2], puntos[simplex[2], 2], puntos[simplex[0], 2]]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=2)))

    # Mostrar figura
    fig.show()
    '''

    ''' 
    import numpy as np
    from scipy.spatial import Delaunay
    import plotly.graph_objects as go

    # Generar una nube de puntos aleatorios en 3D
    np.random.seed(0)
    n_puntos = 100
    puntos = np.random.randn(n_puntos, 3)

    # Triangulación de Delaunay
    triangulacion = Delaunay(puntos)

    # Crear figura
    fig = go.Figure()

    # Añadir puntos de datos
    fig.add_trace(go.Scatter3d(x=puntos[:, 0], y=puntos[:, 1], z=puntos[:, 2], mode='markers', marker=dict(size=3)))

    # Añadir triángulos
    for simplex in triangulacion.simplices:
        x = [puntos[simplex[0], 0], puntos[simplex[1], 0], puntos[simplex[2], 0], puntos[simplex[0], 0]]
        y = [puntos[simplex[0], 1], puntos[simplex[1], 1], puntos[simplex[2], 1], puntos[simplex[0], 1]]
        z = [puntos[simplex[0], 2], puntos[simplex[1], 2], puntos[simplex[2], 2], puntos[simplex[0], 2]]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color='lightpink'))

    # Mostrar figura
    fig.show()
    '''
