import numpy as np

# Parameter
L, C = 0.5, 10e-6  # Henry dan Farad
target_frequency, tolerance = 1000, 0.1  # Hz dan toleransi
a, b, R0 = 0, 100, 50  # Batas untuk Biseksi dan tebakan awal untuk Newton-Raphson

# Fungsi frekuensi resonansi f(R) dan turunannya f'(R)
def f(R):
    return (1 / (2 * np.pi)) * np.sqrt(1 / (L * C) - (R ** 2) / (4 * L ** 2))

def f_prime(R):
    return (-R / (4 * np.pi * L)) * (1 / np.sqrt(1 / (L * C) - (R ** 2) / (4 * L ** 2)))

# Metode Biseksi
def bisection_method(a, b, tol):
    while (b - a) / 2 > tol:
        mid = (a + b) / 2
        if f(mid) == target_frequency or (b - a) / 2 < tol:
            return mid
        elif f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Metode Newton-Raphson
def newton_raphson(R, tol):
    while abs(f(R) - target_frequency) > tol:
        R -= (f(R) - target_frequency) / f_prime(R)
    return R

# Menjalankan metode Biseksi
R_bisection = bisection_method(a, b, tolerance)
print(f"Hasil Akhir R (Bisection): {R_bisection:.4f} Ohm")

# Menjalankan metode Newton-Raphson
R_newton = newton_raphson(R0, tolerance)
print(f"Hasil Akhir R (Newton-Raphson): {R_newton:.4f} Ohm")

# Perbandingan hasil akhir
print("\nPerbandingan Hasil Akhir:")
print(f"R (Bisection): {R_bisection:.4f} Ohm")
print(f"R (Newton-Raphson): {R_newton:.4f} Ohm")

import numpy as np

# Sistem persamaan
A = np.array([[4, -1, -1],
              [-1, 3, -1],
              [-1, -1, 5]], dtype=float)
b = np.array([5, 3, 4], dtype=float)

# a) Fungsi eliminasi Gauss
def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError("Sistem tidak memiliki solusi unik.")
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]
        
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]
    return x

# b) Fungsi menghitung determinan dengan ekspansi kofaktor
def cofactor_determinant(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix[0, 0]
    elif n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    det = 0
    for c in range(n):
        minor = np.delete(np.delete(matrix, 0, axis=0), c, axis=1)
        det += ((-1) ** c) * matrix[0, c] * cofactor_determinant(minor)
    return det

# c) Menyelesaikan sistem persamaan menggunakan eliminasi Gauss dan menghitung determinan
x_gauss = gauss_elimination(A.copy(), b.copy())
det_A = cofactor_determinant(A)
print("Solusi dengan Metode Gauss:", x_gauss)
print("Determinan matriks A:", det_A)

# d) Fungsi metode Gauss-Jordan
def gauss_jordan(A, b):
    n = len(b)
    augmented = np.hstack((A, b.reshape(-1, 1)))
    
    for i in range(n):
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        augmented[[i, max_row]] = augmented[[max_row, i]]
        
        augmented[i] = augmented[i] / augmented[i, i]
        
        for j in range(n):
            if i != j:
                factor = augmented[j, i]
                augmented[j] -= factor * augmented[i]
    
    return augmented[:, -1]

# Menyelesaikan sistem persamaan menggunakan Gauss-Jordan
x_gauss_jordan = gauss_jordan(A.copy(), b.copy())
print("Solusi dengan Metode Gauss-Jordan:", x_gauss_jordan)

# e) Fungsi menghitung kofaktor matriks
def matrix_cofactor(matrix, row, col):
    minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
    return ((-1) ** (row + col)) * cofactor_determinant(minor)

# Fungsi mencari matriks adjoint
def adjoint(matrix):
    n = matrix.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            adj[j, i] = matrix_cofactor(matrix, i, j)
    return adj

# Fungsi mencari invers menggunakan metode adjoin
def inverse(matrix):
    det = cofactor_determinant(matrix)
    if det == 0:
        raise ValueError("Matriks tidak memiliki invers karena determinannya nol.")
    adj = adjoint(matrix)
    return adj / det

# Menghitung invers matriks A menggunakan metode adjoin
A_inv = inverse(A)
print("Invers matriks A menggunakan metode adjoin:\n", A_inv)

import numpy as np

# Fungsi resistansi termistor
def R(T):
    return 5000 * (1 - (3500 / (T + 298)))

# a) Fungsi untuk menghitung selisih maju, mundur, dan tengah
def forward_difference(func, T, dT):
    return (func(T + dT) - func(T)) / dT

def backward_difference(func, T, dT):
    return (func(T) - func(T - dT)) / dT

def central_difference(func, T, dT):
    return (func(T + dT) - func(T - dT)) / (2 * dT)

# b) Fungsi untuk menghitung nilai eksak
def exact_derivative(T):
    # Derivatif eksak dari R(T) = 5000 * (1 - (3500 / (T + 298)))
    return (5000 * (3500) / ((T + 298)**2))

# c) Hitung pada rentang temperatur 250K sampai 350K dengan interval 10K
T_values = np.arange(250, 360, 10)
dT = 10  # interval temperatur
results = {
    "T": T_values,
    "Forward": [forward_difference(R, T, dT) for T in T_values],
    "Backward": [backward_difference(R, T, dT) for T in T_values],
    "Central": [central_difference(R, T, dT) for T in T_values],
    "Exact": [exact_derivative(T) for T in T_values],
}

# d) Hitung error relatif
errors = {
    "Forward": np.abs((np.array(results["Forward"]) - np.array(results["Exact"])) / results["Exact"]),
    "Backward": np.abs((np.array(results["Backward"]) - np.array(results["Exact"])) / results["Exact"]),
    "Central": np.abs((np.array(results["Central"]) - np.array(results["Exact"])) / results["Exact"]),
}

# Cetak hasil
print("Temperature (K)\tForward Error\tBackward Error\tCentral Error")
for i in range(len(T_values)):
    print(f"{T_values[i]}\t\t{errors['Forward'][i]:.6e}\t{errors['Backward'][i]:.6e}\t{errors['Central'][i]:.6e}")

# e) Metode Extrapolasi Richardson
def richardson_extrapolation(f, T, dT):
    forward = forward_difference(f, T, dT)
    backward = backward_difference(f, T, dT)
    return (4 * forward - backward) / 3  # Richardson's formula for accuracy improvement

# Hitung hasil dengan extrapolasi Richardson
richardson_results = {
    "T": T_values,
    "Richardson": [richardson_extrapolation(R, T, dT) for T in T_values],
}

# Bandingkan hasilnya dengan metode sebelumnya
errors["Richardson"] = np.abs((np.array(richardson_results["Richardson"]) - np.array(results["Exact"])) / results["Exact"])

# Cetak hasil untuk error Richardson
print("\nTemperature (K)\tRichardson Error")
for i in range(len(T_values)):
    print(f"{T_values[i]}\t\t{errors['Richardson'][i]:.6e}")