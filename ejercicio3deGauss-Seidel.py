import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    errors = []
    iterations_data = []
    
    for iteration in range(max_iter):
        x_old = np.copy(x)
        
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum1) / A[i][i]
        
        abs_error = np.linalg.norm(x - x_old, ord=np.inf)
        rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)
        sq_error = np.linalg.norm(x - x_old) ** 2
        errors.append((abs_error, rel_error, sq_error))
        iterations_data.append([iteration + 1] + list(x))
        
        if abs_error < tol:
            break
    
    return x, errors, iterations_data

def plot_errors(errors, title):
    errors = np.array(errors)
    plt.figure()
    plt.plot(errors[:, 0], label='Error Absoluto')
    plt.plot(errors[:, 1], label='Error Relativo')
    plt.plot(errors[:, 2], label='Error Cuadrático')
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def solve_and_plot(A, b, title, var_names):
    solution, errors, iterations_data = gauss_seidel(A, b)
    
    # Mostrar tabla con soluciones iterativas
    columns = ["Iteración"] + var_names
    df = pd.DataFrame(iterations_data, columns=columns)
    print(f"\n{title} - Tabla de soluciones iterativas:")
    print(df.to_string(index=False))
    
    # Graficar errores
    plot_errors(errors, title)
    
    # Análisis de convergencia
    print(f"\n{title} - Convergencia:")
    if len(errors) < 100:
        print(f"El método convergió en {len(errors)} iteraciones.")
    else:
        print("El método no convergió en el número máximo de iteraciones permitido.")
    
    print(f"Solución final: {solution}\n")

# Ejercicio 3: Modelo de economía lineal
A3 = np.array([[15, -4, -1, -2, 0, 0, 0, 0, 0, 0],
               [-3, 18, -2, 0, -1, 0, 0, 0, 0, 0],
               [-1, -2, 20, 0, 0, -5, 0, 0, 0, 0],
               [-2, -1, -4, 22, 0, 0, -1, 0, 0, 0],
               [0, -1, -3, -1, 25, 0, 0, -2, 0, 0],
               [0, 0, -2, 0, -1, 28, 0, 0, -1, 0],
               [0, 0, 0, -4, 0, -2, 30, 0, 0, -3],
               [0, 0, 0, 0, -1, 0, -1, 35, -2, 0],
               [0, 0, 0, 0, 0, -2, 0, -3, 40, -1],
               [0, 0, 0, 0, 0, 0, -3, 0, -1, 45]])
b3 = np.array([200, 250, 180, 300, 270, 310, 320, 400, 450, 500])
solve_and_plot(A3, b3, "Modelo de Economía Lineal", ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"])