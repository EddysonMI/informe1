import pandas as pd
import numpy as np


serie = pd.Series([10, 20, 30, 40, 50], name='Valores')
print("Serie:")
print(serie)


datos = {
    'Categorías': ['A', 'B', 'C', 'D', 'E'],
    'Valores': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(datos)
print("\nDataFrame:")
print(df)








array = np.array([10, 20, 30, 40, 50])
print("\nArray numpy:")
print(array)


print("\nSuma del array:")
print(np.sum(array))

print("\nMedia del array:")
print(np.mean(array))

print("\nDesviación estándar del array:")
print(np.std(array))


array2 = np.array([1, 2, 3, 4, 5])
print("\nSuma de arrays:")
print(array + array2)

print("\nMultiplicación de arrays:")
print(array * array2)






