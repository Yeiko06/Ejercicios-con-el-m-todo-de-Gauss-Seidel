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

# Ejercicio 2: Transferencia de calor en una placa metálica
A2 = np.array([[20, -5, -3, 0, 0],
               [-4, 18, -2, -1, 0],
               [-3, -1, 22, -5, 0],
               [0, -2, -4, 25, -1],
               [0, 0, 0, -1, 30]])
b2 = np.array([100, 120, 130, 150, 180])
solve_and_plot(A2, b2, "Transferencia de Calor", ["T1", "T2", "T3", "T4", "T5"])