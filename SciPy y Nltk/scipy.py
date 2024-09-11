import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Datos de ejemplo
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# Crear una función de interpolación
f_interpolacion = interp1d(x, y, kind='cubic')

# Generar nuevos puntos
x_nuevos = np.linspace(0, 4, 100)
y_nuevos = f_interpolacion(x_nuevos)

# Graficar
plt.plot(x, y, 'o', label='Datos originales')
plt.plot(x_nuevos, y_nuevos, '-', label='Interpolación cúbica')
plt.legend()
plt.show()
