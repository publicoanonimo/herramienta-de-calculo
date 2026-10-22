
"""
Eficiencia técnica DEA (CCR y BCC, orientación a insumos) de las
universidades públicas autónomas (UPA) de Bolivia, 2023.

Fuente de datos: Aguirre y Quintanilla (2025), Tablas 1, 3 y 4.
Verificación: forma envolvente (primal) frente a forma de multiplicadores (dual).

Requisitos: Python 3.8+, numpy, scipy (>= 1.9)
Ejecución:  python dea_upa.py
"""
import numpy as np
from scipy.optimize import linprog


# ---------- 1. Datos ----------
# Cada posición de las listas corresponde a una universidad.
units = ["UABJB", "UAGRM", "UAJMS", "UAP", "UATF", "UMRPSFXCH",
         "UMSA", "UMSS", "UNSXX", "UPEA", "UTO"]

# Insumos: gasto ejecutado (millones de Bs), docentes y administrativos.
gasto = [241.452209, 1005.408070, 296.512161, 121.549255, 297.490761, 434.327528,
         1064.217282, 810.660334, 106.983676, 482.059598, 287.961975]
docentes = [943, 1651, 987, 616, 656, 1249, 2468, 1824, 365, 3437, 995]
admin    = [1039, 1740, 883, 753, 420, 1244, 1411, 1339, 261, 835, 610]

# Producto: número de titulados en 2023.
titulados = [1008, 3155, 1996, 185, 2411, 5055, 4504, 5479, 1232, 3588, 2279]

# Matrices del modelo: X (n x m) para insumos y Y (n x s) para productos.
# n = 11 universidades, m = 3 insumos, s = 1 producto.
X = np.column_stack([gasto, docentes, admin]).astype(float)
Y = np.column_stack([titulados]).astype(float)


# ---------- 2. Forma envolvente (primal) ----------
# Para cada universidad o, resuelve el problema:
#   min θ   s.a.  Σ λ_j x_ij ≤ θ x_io   (insumos)
#                 Σ λ_j y_j  ≥ y_o      (producto)
#                 Σ λ_j = 1             (solo BCC: rendimientos variables)
#                 λ ≥ 0
# θ = 1 indica que la universidad está en la frontera de eficiencia.
def dea_envolvente(X, Y, rts="CCR"):
    n, m = X.shape
    s = Y.shape[1]
    theta = []
    for o in range(n):
        # Variables del problema: [θ, λ_1, ..., λ_n]. Solo se minimiza θ.
        c = np.r_[1.0, np.zeros(n)]

        # Restricciones de desigualdad A_ub · z ≤ b_ub.
        # Insumos: Σ λ_j x_ij − θ x_io ≤ 0.
        A_ub = [np.r_[-X[o, i], X[:, i]] for i in range(m)]
        # Producto: −Σ λ_j y_j ≤ −y_o, equivalente a Σ λ_j y_j ≥ y_o.
        A_ub += [np.r_[0.0, -Y[:, r]] for r in range(s)]
        b_ub = np.r_[np.zeros(m), -Y[o, :]]

        # Restricción de igualdad Σ λ_j = 1, solo en el modelo BCC.
        A_eq = np.array([np.r_[0.0, np.ones(n)]]) if rts == "BCC" else None
        b_eq = [1] if rts == "BCC" else None

        # Resolución con programación lineal (método HiGHS). Todas las
        # variables son no negativas.
        r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0, None)] * (n + 1), method="highs")
        theta.append(r.x[0])   # valor óptimo de θ para la universidad o
    return np.array(theta)


# ---------- 3. Forma de multiplicadores (dual) ----------
# Para cada universidad o, maximiza la producción ponderada sujeta a que
# los insumos ponderados de o valgan 1. Debe dar el mismo resultado que la
# forma envolvente; por eso sirve como verificación independiente.
def dea_multiplicadores(X, Y, rts="CCR"):
    n, m = X.shape
    s = Y.shape[1]
    eff = []
    for o in range(n):
        # Variables: [v (pesos de insumos), u (pesos de productos), u0].
        # Se usa −c porque linprog minimiza y aquí se busca un máximo.
        c = -np.r_[np.zeros(m), Y[o], 1.0 if rts == "BCC" else 0.0]

        # Restricciones u·y_j − v·x_j + u0 ≤ 0, una por universidad j.
        # En CCR, u0 se fija en 0; en BCC queda libre.
        A_ub = np.column_stack([-X, Y, np.ones(n) if rts == "BCC" else np.zeros(n)])
        b_ub = np.zeros(n)

        # Normalización: v · x_o = 1 (insumos ponderados de la universidad o).
        A_eq = np.r_[X[o], np.zeros(s), 0.0].reshape(1, -1)

        # Pesos v y u no negativos. u0 es libre en BCC y nulo en CCR.
        bnd_u0 = (None, None) if rts == "BCC" else (0, 0)
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1],
                    bounds=[(0, None)] * (m + s) + [bnd_u0], method="highs")
        eff.append(-r.fun)     # se invierte el signo para obtener el máximo
    return np.array(eff)


# ---------- 4. Cálculo y verificación ----------
ccr_env = dea_envolvente(X, Y, "CCR")
ccr_mul = dea_multiplicadores(X, Y, "CCR")
bcc_env = dea_envolvente(X, Y, "BCC")
bcc_mul = dea_multiplicadores(X, Y, "BCC")

# Tabla de resultados. Eficiencia de escala = CCR / BCC (fórmula 3 del artículo).
print(f"{'UPA':12s} {'CCR':>8s} {'BCC':>8s} {'Escala':>8s}")
for i, u in enumerate(units):
    escala = ccr_env[i] / bcc_env[i]
    print(f"{u:12s} {ccr_env[i]*100:8.2f} {bcc_env[i]*100:8.2f} {escala*100:8.2f}")

# Verificación: las dos formulaciones deben dar diferencias cercanas a cero.
print()
print("Verificación (envolvente vs. multiplicadores):")
print(f"  Diferencia máxima CCR: {np.max(np.abs(ccr_env - ccr_mul)):.2e}")
print(f"  Diferencia máxima BCC: {np.max(np.abs(bcc_env - bcc_mul)):.2e}")
