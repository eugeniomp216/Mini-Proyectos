from __future__ import annotations
from numpy import *
from numpy.linalg import *
import numpy as np
from numpy.typing import ArrayLike,NDArray
from typing import Any, Dict, List, Literal, Tuple, Union


"""
np.asarray(A): No hace copia si A ya es un array de NumPy. Simplemente devuelve una referencia al array original. 
Solo crea un array nuevo si A es otro tipo de dato (como una lista o una tupla).



"""

def norma_vec(X, p):
    
    X = np.asarray(X)
    if p < 1:
        raise ValueError(f"El orden 'p' debe ser >= 1 para definir una norma (recibido: {p}).")

    if X.ndim == 1:
        pass  
    
    elif X.ndim == 2 and X.shape[1] == 1:
        X = X.flatten()  # Resultado: Un array 1D
    
    else:
        raise ValueError("El argumento debe ser un vector o matriz columna")
    

    X = X.astype(np.float64)

    if p == np.inf:
        return np.max(np.abs(X))
    else:
        return np.sum(np.abs(X)**p)**(1/p)
    



def norma_vec(x: ArrayLike, p: Union[int, float] = 2) -> float:
    """Calcula la p-norma vectorial con estabilidad numérica en todo el rango de p.

    Combina escalado por la norma infinito (evita overflow) con evaluación
    logarítmica adaptativa (evita underflow cuando p es muy grande).

    Args:
        x: Vector 1D o matriz columna (n, 1) de valores reales o complejos.
        p: Orden de la norma (p >= 1 o np.inf). Por defecto es 2.

    Returns:
        Valor escalar de la p-norma en coma flotante de doble precisión.

    Raises:
        ValueError: Si 'x' no es un vector válido, está vacío, o si p < 1.
    """
    if p < 1:
        raise ValueError(f"El orden 'p' debe ser >= 1 (recibido: {p}).")

    arr = np.asarray(x)

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError(f"Se esperaba un vector 1D o columna (n, 1), recibido shape {arr.shape}.")

    if arr.size == 0:
        raise ValueError("No se puede calcular la norma de un vector vacío.")

    abs_x = np.abs(arr).astype(np.float64)
    m = float(np.max(abs_x))

    if p == np.inf or m == 0.0:
        return m

    if p == 1:
        return float(np.sum(abs_x))

    # Escalar por m in (0, 1] previene overflow (entradas enormes)
    # y underflow catastrófico (entradas diminutas) al elevar a p
    y = abs_x / m
    return float(m * np.sum(y ** p) ** (1.0 / p))
    

def verificar_convergencia_norma_inf(x: ArrayLike,tol: float = 1e-10,max_p: int = 200,verbose: bool = False,) -> Dict[str, Any]:
    """Evalúa la convergencia numérica de ||x||_p hacia ||x||_inf cuando p -> inf.

    Args:
        x: Vector de entrada (1D) o matriz columna (n, 1).
        tol: Tolerancia del error relativo para detener la iteración.
        max_p: Orden máximo 'p' a evaluar.
        verbose: Si es True, imprime el progreso por consola.

    Returns:
        Diccionario con:
            - 'converged' (bool): Si se alcanzó la tolerancia antes de max_p.
            - 'p_final' (int): Último orden 'p' evaluado.
            - 'norm_inf' (float): Valor de la norma infinito.
            - 'history' (list[dict]): Registro por iteración con 'p', 'norm_p' y 'rel_error'.
    """
    if tol <= 0:
        raise ValueError(f"La tolerancia debe ser estrictamente positiva (recibida: {tol}).")
    if max_p < 1:
        raise ValueError(f"max_p debe ser >= 1 (recibido: {max_p}).")

    norm_inf = norma_vec(x, np.inf)

    if norm_inf == 0.0:
        return {
            "converged": True,
            "p_final": 1,
            "norm_inf": 0.0,
            "history": [{"p": 1, "norm_p": 0.0, "rel_error": 0.0}],
        }

    history: List[Dict[str, float]] = []
    converged = False

    for p in range(1, max_p + 1):
        norm_p = norma_vec(x, p)
        rel_error = abs(norm_p - norm_inf) / norm_inf
        history.append({"p": p, "norm_p": norm_p, "rel_error": rel_error})

        if verbose:
            print(f"p = {p:3d} | ||x||_p = {norm_p:.12f} | Error relativo = {rel_error:.2e}")

        if rel_error < tol:
            converged = True
            break

    return {
        "converged": converged,
        "p_final": history[-1]["p"],
        "norm_inf": norm_inf,
        "history": history,
    }


def norma_mat(a: ArrayLike, p: MatrixNormOrder = "fro") -> float:
    """Calcula la norma matricial de una matriz real o compleja (m x n).

    Soporta las normas inducidas p=1 (máxima suma por columnas), p=np.inf
    (máxima suma por filas), p=2 (norma espectral vía SVD) y la norma
    de Frobenius ('fro').

    Args:
        a: Matriz bidimensional (m, n) no vacía.
        p: Tipo de norma: 1, 2, np.inf o 'fro'. Por defecto es 'fro'.

    Returns:
        Valor escalar de la norma matricial como float estándar de Python.

    Raises:
        ValueError: Si 'a' no es una matriz 2D no vacía o 'p' no es válido.
    """
    arr = np.asarray(a)

    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una matriz 2D, pero se recibió shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("No se puede calcular la norma de una matriz vacía.")

    # Preservar precisión real (float64) salvo que la entrada ya sea compleja
    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    arr = arr.astype(dtype, copy=False) #copy=False Si el array ya es del tipo de dato que te pedí, no hagas una copia en memoria, simplemente devuélveme el mismo array".

    if p == np.inf:
        return float(np.max(np.sum(np.abs(arr), axis=1)))
    if p == 1:
        return float(np.max(np.sum(np.abs(arr), axis=0)))
    if p == 2:
        # Valor singular máximo (evita formar A^H A y elevar al cuadrado cond(A))
        return float(np.linalg.svd(arr, compute_uv=False)[0])
    if p == "fro":
        # Reutiliza norma_vec para mantener estabilidad frente a overflow/underflow
        return norma_vec(arr.ravel(), 2)

    raise ValueError(f"Orden de norma matricial no soportado: {p!r}. Usa 1, 2, np.inf o 'fro'.")



def descenso(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,tol: float = 1e-15,) -> NDArray[np.inexact]:
    """Resuelve el sistema triangular inferior A X = B por sustitución hacia delante.

    Soporta tanto un único término independiente (vector 1D) como múltiples
    sistemas simultáneos (matriz 2D), con entradas reales o complejas.
    Solo accede a la parte triangular inferior de A.

    Args:
        a: Matriz cuadrada triangular inferior de dimensiones (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume diagonal unitaria en A (evita divisiones
            y permite operar sobre factores L almacenados en matrices LU compactas).
        tol: Umbral mínimo para el módulo de los elementos diagonales al verificar
            singularidad (se ignora si unit_diagonal es True).

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).

    Raises:
        ValueError: Si las dimensiones de A o B son inválidas o incompatibles.
        np.linalg.LinAlgError: Si A es singular (algún elemento diagonal <= tol).
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[0] != arr_a.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr_a.shape}.")
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A es ({n}, {n}) pero B tiene {arr_b.shape[0]} filas."
        )

    if not unit_diagonal and np.any(np.abs(np.diag(arr_a)) <= tol):
        raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en la diagonal.")

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    mat_a = arr_a.astype(dtype, copy=False)
    x = arr_b.astype(dtype, copy=True)

    for i in range(n):
        if i > 0:
            x[i] -= mat_a[i, :i] @ x[:i]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x


def remonte(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,tol: float = 1e-15,) -> np.ndarray:
    """Resuelve el sistema triangular superior A X = B por sustitución hacia atrás.

    Soporta un único término independiente (vector 1D) o múltiples sistemas
    simultáneos (matriz 2D), con entradas reales o complejas.

    Args:
        a: Matriz cuadrada triangular superior de dimensiones (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal de A (útil en
            factorizaciones tipo Crout o LDL^T compactas).
        tol: Umbral mínimo para los elementos diagonales al detectar singularidad.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).

    Raises:
        ValueError: Si las dimensiones de A o B son inválidas o incompatibles.
        np.linalg.LinAlgError: Si A es singular (algún elemento diagonal <= tol).
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[0] != arr_a.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr_a.shape}.")
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A es ({n}, {n}) pero B tiene {arr_b.shape[0]} filas."
        )

    if not unit_diagonal and np.any(np.abs(np.diag(arr_a)) <= tol):
        raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en la diagonal.")

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    mat_a = arr_a.astype(dtype, copy=False)
    x = arr_b.astype(dtype, copy=True)

    for i in range(n - 1, -1, -1):
        if i < n - 1:
            x[i] -= mat_a[i, i + 1:] @ x[i + 1:]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x



def descenso1(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Resuelve A X = B para una matriz triangular inferior con diagonal unitaria."""
    return descenso(a, b, unit_diagonal=True)



def remonte1(A, B):
   """Resuelve A X = B para una matriz triangular inferior con diagonal unitaria."""
   return remonte(a, b, unit_diagonal=True)

def _validar_sistema_triangular(a: ArrayLike,b: ArrayLike,unit_diagonal: bool,tol: float,) -> Tuple[np.ndarray, np.ndarray]:
    """Valida dimensiones, singularidad y prepara los tipos de A y B."""
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[0] != arr_a.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr_a.shape}.")
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A es ({n}, {n}) pero B tiene {arr_b.shape[0]} filas."
        )

    if not unit_diagonal and np.any(np.abs(np.diag(arr_a)) <= tol):
        raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en la diagonal.")

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    mat_a = arr_a.astype(dtype, copy=False)
    x = arr_b.astype(dtype, copy=True)

    return mat_a, x


def descenso_bidiagonal(
    a: ArrayLike,
    b: ArrayLike,
    unit_diagonal: bool = False,
    tol: float = 1e-15,
) -> np.ndarray:
    """Resuelve A X = B en tiempo O(n) para una matriz bidiagonal inferior.

    Utiliza únicamente la diagonal principal y la primera subdiagonal de A.
    Soporta vectores 1D o múltiples términos independientes (matriz 2D),
    con entradas reales o complejas.

    Args:
        a: Matriz cuadrada bidiagonal inferior de dimensiones (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal de A (útil para
            el factor L de una factorización LU tridiagonal).
        tol: Umbral mínimo para los elementos diagonales al verificar singularidad.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).

    Raises:
        ValueError: Si las dimensiones de A o B son inválidas o incompatibles.
        np.linalg.LinAlgError: Si A es singular (algún elemento diagonal menor o igual a tol).
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[0] != arr_a.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr_a.shape}.")
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A es ({n}, {n}) pero B tiene {arr_b.shape[0]} filas."
        )

    if not unit_diagonal and np.any(np.abs(np.diag(arr_a)) <= tol):
        raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en la diagonal.")

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    mat_a = arr_a.astype(dtype, copy=False)
    x = arr_b.astype(dtype, copy=True)

    for i in range(n):
        if i > 0:
            x[i] -= mat_a[i, i - 1] * x[i - 1]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x



def remonte_bidiagonal(
    a: ArrayLike,
    b: ArrayLike,
    unit_diagonal: bool = False,
    tol: float = 1e-15,
) -> np.ndarray:
    """Resuelve A X = B en tiempo O(n) para una matriz bidiagonal superior.

    Utiliza únicamente la diagonal principal y la primera supradiagonal de A.
    Soporta vectores 1D o matrices 2D con entradas reales o complejas.

    Args:
        a: Matriz cuadrada bidiagonal superior de dimensiones (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal de A.
        tol: Umbral mínimo para los elementos diagonales al detectar singularidad.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).
    """
    mat_a, x = _validar_sistema_triangular(a, b, unit_diagonal, tol)
    n = mat_a.shape[0]

    for i in range(n - 1, -1, -1):
        if i < n - 1:
            x[i] -= mat_a[i, i + 1] * x[i + 1]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x


def _validar_bandwidth(bandwidth: int | None, n: int) -> int:
    """Valida el ancho de banda r. Si es None, usa la matriz triangular completa."""
    if bandwidth is None:
        return n - 1
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)) or bandwidth < 0:
        raise ValueError(f"El ancho de banda 'bandwidth' debe ser un entero >= 0, recibido: {bandwidth!r}.")
    return int(bandwidth)


def descenso_r(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,bandwidth: int | None = None,tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B por sustitución hacia delante (completa o en banda).

    Args:
        a: Matriz cuadrada triangular inferior (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal de A.
        bandwidth: Número de subdiagonales no nulas (r >= 0). Si es None,
            usa toda la parte triangular inferior. Con bandwidth=1 resuelve
            sistemas bidiagonales en O(n).
        tol: Umbral mínimo para los elementos diagonales al detectar singularidad.

    Returns:
        Solución X con la misma forma que B.
    """
    mat_a, x = _validar_sistema_triangular(a, b, unit_diagonal, tol)
    n = mat_a.shape[0]
    r = _validar_bandwidth(bandwidth, n)

    for i in range(n):
        if i > 0 and r > 0:
            j_ini = max(0, i - r)
            x[i] -= mat_a[i, j_ini:i] @ x[j_ini:i]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x


def remonte_r(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,bandwidth: int | None = None,tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B por sustitución hacia atrás (completa o en banda).

    Args:
        a: Matriz cuadrada triangular superior (n, n).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal de A.
        bandwidth: Número de supradiagonales no nulas (r >= 0). Si es None,
            usa toda la parte triangular superior.
        tol: Umbral mínimo para los elementos diagonales al detectar singularidad.

    Returns:
        Solución X con la misma forma que B.
    """
    mat_a, x = _validar_sistema_triangular(a, b, unit_diagonal, tol)
    n = mat_a.shape[0]
    r = _validar_bandwidth(bandwidth, n)

    for i in range(n - 1, -1, -1):
        if i < n - 1 and r > 0:
            j_fin = min(n, i + r + 1)
            x[i] -= mat_a[i, i + 1:j_fin] @ x[i + 1:j_fin]
        if not unit_diagonal:
            x[i] /= mat_a[i, i]

    return x



def _validar_bidiagonal_compacta(a: ArrayLike,b: ArrayLike,diag_col: int,unit_diagonal: bool,tol: float,) -> tuple[np.ndarray, np.ndarray]:
    """Valida una matriz bidiagonal en formato compacto (n, 2) y el término B."""
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[1] != 2:
        raise ValueError(
            f"A debe estar en formato compacto (n, 2), pero se recibió shape {arr_a.shape}."
        )
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A tiene {n} filas pero B tiene {arr_b.shape[0]}."
        )

    if not unit_diagonal and np.any(np.abs(arr_a[:, diag_col]) <= tol):
        raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en la diagonal.")

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    mat_a = arr_a.astype(dtype, copy=False)
    x = arr_b.astype(dtype, copy=True)

    return mat_a, x


def descenso_bidiagonal_compacto(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B en O(n) tiempo y O(n) memoria para una bidiagonal inferior.

    Args:
        a: Matriz compacta de forma (n, 2), donde:
            - a[:, 0] es la subdiagonal (el elemento a[0, 0] se ignora).
            - a[:, 1] es la diagonal principal.
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal e ignora a[:, 1].
        tol: Umbral mínimo para los elementos diagonales al verificar singularidad.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).
    """
    mat_a, x = _validar_bidiagonal_compacta(a, b, diag_col=1, unit_diagonal=unit_diagonal, tol=tol)
    n = mat_a.shape[0]

    for i in range(n):
        if i > 0:
            x[i] -= mat_a[i, 0] * x[i - 1]
        if not unit_diagonal:
            x[i] /= mat_a[i, 1]

    return x


def remonte_bidiagonal_compacto(a: ArrayLike,b: ArrayLike,unit_diagonal: bool = False,tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B en O(n) tiempo y O(n) memoria para una bidiagonal superior.

    Args:
        a: Matriz compacta de forma (n, 2), donde:
            - a[:, 0] es la diagonal principal.
            - a[:, 1] es la supradiagonal (el elemento a[n-1, 1] se ignora).
        b: Vector (n,) o matriz de términos independientes (n, q).
        unit_diagonal: Si es True, asume unos en la diagonal e ignora a[:, 0].
        tol: Umbral mínimo para los elementos diagonales al verificar singularidad.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).
    """
    mat_a, x = _validar_bidiagonal_compacta(a, b, diag_col=0, unit_diagonal=unit_diagonal, tol=tol)
    n = mat_a.shape[0]

    for i in range(n - 1, -1, -1):
        if i < n - 1:
            x[i] -= mat_a[i, 1] * x[i + 1]
        if not unit_diagonal:
            x[i] /= mat_a[i, 0]

    return x



def _validar_matriz_cuadrada(a: ArrayLike) -> np.ndarray:
    """Valida que la entrada sea una matriz cuadrada 2D no vacía y conserva su cuerpo."""
    arr = np.asarray(a)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("La matriz A no puede estar vacía.")

    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    return arr.astype(dtype, copy=False)


def determinantes_principales(a: ArrayLike) -> np.ndarray:
    """Calcula los menores principales líderes det(A[:k, :k]) para k = 1, ..., n.

    Nota numérica:
        Esta implementación directa tiene complejidad O(n^4) y está pensada para
        análisis teórico o matrices de tamaño moderado.

    Args:
        a: Matriz cuadrada (n, n) real o compleja.

    Returns:
        Vector 1D de longitud n con los determinantes principales líderes.

    Raises:
        ValueError: Si 'a' no es una matriz cuadrada 2D no vacía.
    """
    mat = _validar_matriz_cuadrada(a)
    n = mat.shape[0]
    dets = np.empty(n, dtype=mat.dtype)

    for k in range(1, n + 1):
        dets[k - 1] = np.linalg.det(mat[:k, :k])

    return dets


def admite_lu(a: ArrayLike,tol: float = 1e-10,exigir_invertible: bool = False,) -> bool:
    """Comprueba el criterio de los menores principales para la factorización LU sin pivoteo.

    Verifica si det(A[:k, :k]) es no nulo para k = 1, ..., n - 1 (o hasta n si
    se exige además que A sea invertible). Se detiene en cuanto detecta un menor
    menor o igual que 'tol'.

    Nota numérica:
        El criterio basado en determinantes es sensible al escalado de la matriz
        (p. ej., c * I con |c| pequeño puede dar falsos negativos) y cuesta O(n^4).
        En producción es preferible intentar la factorización LU y comprobar los pivotes.

    Args:
        a: Matriz cuadrada (n, n) real o compleja.
        tol: Umbral mínimo para considerar no nulo el módulo de un determinante.
        exigir_invertible: Si es True, exige también det(A) != 0 (k = n).

    Returns:
        True si satisface el criterio de menores principales; False en caso contrario.

    Raises:
        ValueError: Si 'a' no es una matriz cuadrada 2D no vacía o si tol < 0.
    """
    if tol < 0:
        raise ValueError(f"La tolerancia 'tol' debe ser >= 0, recibido: {tol}.")

    mat = _validar_matriz_cuadrada(a)
    n = mat.shape[0]
    limite = n if exigir_invertible else n - 1

    for k in range(1, limite + 1):
        if abs(np.linalg.det(mat[:k, :k])) <= tol:
            return False

    return True



def facto_lu(a: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Calcula la factorización LU sin pivoteo en formato compacto.

    Almacena los multiplicadores de L (asumiendo diagonal unitaria) debajo de
    la diagonal principal y los elementos de U en la diagonal y por encima.

    Args:
        a: Matriz cuadrada (n, n) real o compleja.
        tol: Umbral mínimo para los pivotes u_kk.

    Returns:
        Matriz compacta LU de dimensiones (n, n).

    Raises:
        ValueError: Si 'a' no es una matriz cuadrada 2D no vacía.
        np.linalg.LinAlgError: Si algún pivote es nulo o menor o igual que 'tol'.
    """
    lu = _validar_matriz_cuadrada(a).copy()
    n = lu.shape[0]

    for k in range(n - 1):
        if abs(lu[k, k]) <= tol:
            raise np.linalg.LinAlgError(
                f"Pivote nulo o casi nulo en k={k} (|u_kk|={abs(lu[k, k]):.2e}). "
                "La matriz no admite factorización LU sin pivoteo."
            )
        # Actualización vectorizada de multiplicadores y complemento de Schur
        lu[k + 1:, k] /= lu[k, k]
        lu[k + 1:, k + 1:] -= np.outer(lu[k + 1:, k], lu[k, k + 1:])

    return lu


def metodo_lu(a: ArrayLike, b: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Resuelve A X = B mediante factorización LU sin pivoteo.

    Args:
        a: Matriz cuadrada (n, n) real o compleja.
        b: Vector (n,) o matriz de términos independientes (n, q).
        tol: Umbral mínimo para pivotes y elementos diagonales.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).
    """
    arr_a = _validar_matriz_cuadrada(a)
    arr_b = np.asarray(b)
    if arr_b.ndim not in (1, 2) or arr_b.size == 0 or arr_b.shape[0] != arr_a.shape[0]:
        raise ValueError(f"Dimensiones incompatibles entre A {arr_a.shape} y B {arr_b.shape}.")

    lu = facto_lu(arr_a, tol=tol)
    y = descenso(lu, arr_b, unit_diagonal=True, tol=tol)
    return remonte(lu, y, tol=tol)


def facto_cholesky(a: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Calcula la factorización de Cholesky A = L L^T para matrices reales SPD.

    Args:
        a: Matriz cuadrada (n, n) real, simétrica y definida positiva.
        tol: Umbral mínimo para los radicandos diagonales.

    Returns:
        Matriz triangular inferior L de dimensiones (n, n) tal que L @ L.T == A.

    Raises:
        ValueError: Si 'a' no es cuadrada, es compleja o no es simétrica.
        np.linalg.LinAlgError: Si 'a' no es definida positiva.
    """
    mat = _validar_matriz_cuadrada(a)

    if np.iscomplexobj(mat):
        raise ValueError("Esta implementación de Cholesky solo admite matrices reales.")
    if not np.allclose(mat, mat.T, rtol=1e-12, atol=1e-12):
        raise ValueError("La matriz A debe ser simétrica para aplicar Cholesky.")

    n = mat.shape[0]
    l_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        diag_val = mat[i, i] - np.dot(l_mat[i, :i], l_mat[i, :i])
        if diag_val <= tol:
            raise np.linalg.LinAlgError(
                f"La matriz no es definida positiva (radicando <= {tol} en i={i})."
            )
        l_mat[i, i] = np.sqrt(diag_val)

        if i < n - 1:
            l_mat[i + 1:, i] = (mat[i + 1:, i] - l_mat[i + 1:, :i] @ l_mat[i, :i]) / l_mat[i, i]

    return l_mat


def metodo_cholesky(a: ArrayLike, b: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Resuelve A X = B mediante factorización de Cholesky (A real SPD).

    Args:
        a: Matriz cuadrada (n, n) real, simétrica y definida positiva.
        b: Vector (n,) o matriz de términos independientes (n, q).
        tol: Umbral numérico para la factorización y las sustituciones.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).
    """
    arr_a = _validar_matriz_cuadrada(a)
    arr_b = np.asarray(b)
    if arr_b.ndim not in (1, 2) or arr_b.size == 0 or arr_b.shape[0] != arr_a.shape[0]:
        raise ValueError(f"Dimensiones incompatibles entre A {arr_a.shape} y B {arr_b.shape}.")

    l_mat = facto_cholesky(arr_a, tol=tol)
    y = descenso(l_mat, arr_b, tol=tol)
    return remonte(l_mat.T, y, tol=tol)




def _validar_sistema_general(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Valida dimensiones y devuelve copias listas para operar de A (n, n) y B (n,) o (n, q)."""
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    if arr_a.ndim != 2 or arr_a.shape[0] != arr_a.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada 2D, recibido shape {arr_a.shape}.")
    if arr_b.ndim not in (1, 2):
        raise ValueError(f"B debe ser un vector 1D o matriz 2D, recibido shape {arr_b.shape}.")

    n = arr_a.shape[0]
    if n == 0 or arr_b.size == 0:
        raise ValueError("A y B no pueden estar vacíos.")
    if arr_b.shape[0] != n:
        raise ValueError(
            f"Dimensiones incompatibles: A es ({n}, {n}) pero B tiene {arr_b.shape[0]} filas."
        )

    dtype = np.complex128 if (np.iscomplexobj(arr_a) or np.iscomplexobj(arr_b)) else np.float64
    return arr_a.astype(dtype, copy=True), arr_b.astype(dtype, copy=True)


def gauss_pp(a: ArrayLike, b: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Resuelve A X = B mediante eliminación gaussiana con pivoteo parcial.

    Reduce el sistema a una forma triangular superior usando intercambios de
    filas para elegir el pivote de mayor módulo en cada columna y resuelve
    por sustitución hacia atrás (remonte).

    Args:
        a: Matriz cuadrada invertible (n, n) real o compleja.
        b: Vector (n,) o matriz de términos independientes (n, q).
        tol: Umbral mínimo para considerar un pivote no nulo.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).

    Raises:
        ValueError: Si las dimensiones de A o B son inválidas o incompatibles.
        np.linalg.LinAlgError: Si la matriz A es singular o numéricamente singular.
    """
    gauss_a, gauss_b = _validar_sistema_general(a, b)
    n = gauss_a.shape[0]

    for k in range(n - 1):
        # Pivoteo parcial: buscar el mayor elemento en valor absoluto de la columna k
        ik = int(np.argmax(np.abs(gauss_a[k:, k]))) + k
        if abs(gauss_a[ik, k]) <= tol:
            raise np.linalg.LinAlgError(f"Matriz singular o casi singular en la columna k={k}.")

        if ik != k:
            gauss_a[[k, ik]] = gauss_a[[ik, k]]
            gauss_b[[k, ik]] = gauss_b[[ik, k]]

        # Eliminación vectorizada bajo el pivote
        factors = gauss_a[k + 1:, k] / gauss_a[k, k]
        gauss_a[k + 1:, k + 1:] -= np.outer(factors, gauss_a[k, k + 1:])

        if gauss_b.ndim == 1:
            gauss_b[k + 1:] -= factors * gauss_b[k]
        else:
            gauss_b[k + 1:] -= np.outer(factors, gauss_b[k])

    return remonte(gauss_a, gauss_b, tol=tol)


def gaussjordan_pp(a: ArrayLike, b: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Resuelve A X = B mediante eliminación de Gauss-Jordan con pivoteo parcial.

    Diagonaliza implícitamente A hasta la identidad aplicando las mismas
    transformaciones elementales sobre B.

    Args:
        a: Matriz cuadrada invertible (n, n) real o compleja.
        b: Vector (n,) o matriz de términos independientes (n, q).
        tol: Umbral mínimo para considerar un pivote no nulo.

    Returns:
        Solución X con la misma forma que B: (n,) o (n, q).

    Raises:
        ValueError: Si las dimensiones de A o B son inválidas o incompatibles.
        np.linalg.LinAlgError: Si la matriz A es singular o numéricamente singular.
    """
    gauss_a, gauss_b = _validar_sistema_general(a, b)
    n = gauss_a.shape[0]

    for k in range(n):
        ik = int(np.argmax(np.abs(gauss_a[k:, k]))) + k
        if abs(gauss_a[ik, k]) <= tol:
            raise np.linalg.LinAlgError(f"Matriz singular o casi singular en la columna k={k}.")

        if ik != k:
            gauss_a[[k, ik]] = gauss_a[[ik, k]]
            gauss_b[[k, ik]] = gauss_b[[ik, k]]

        # Normalizar solo la parte activa de la fila pivote (columnas k+1 en adelante)
        piv = gauss_a[k, k]
        gauss_a[k, k + 1:] /= piv
        gauss_b[k] /= piv

        # Eliminar el resto de filas de golpe (poniendo factor 0 en la propia fila k)
        factors = gauss_a[:, k].copy()
        factors[k] = 0.0

        gauss_a[:, k + 1:] -= np.outer(factors, gauss_a[k, k + 1:])
        if gauss_b.ndim == 1:
            gauss_b -= factors * gauss_b[k]
        else:
            gauss_b -= np.outer(factors, gauss_b[k])

    return gauss_b

def _buscar_pivote(col: np.ndarray, pivoteo: str, tol: float) -> int:
    """Devuelve el índice relativo del pivote según la estrategia elegida."""
    abs_col = np.abs(col)
    if pivoteo == "parcial":
        idx = int(np.argmax(abs_col))
        if abs_col[idx] <= tol:
            raise np.linalg.LinAlgError("Matriz singular o numéricamente singular.")
        return idx
    if pivoteo == "primero":
        # np.flatnonzero devuelve los índices donde se cumple la condición
        validos = np.flatnonzero(abs_col > tol)
        if validos.size == 0:
            raise np.linalg.LinAlgError("Matriz singular o numéricamente singular.")
        return int(validos[0])
    raise ValueError(f"Estrategia de pivoteo no válida: {pivoteo!r}. Use 'parcial' o 'primero'.")


def gauss(a: ArrayLike,b: ArrayLike,pivoteo: str = "parcial",tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B por eliminación gaussiana ('parcial' o 'primero')."""
    if pivoteo not in ("parcial", "primero"):
        raise ValueError(f"Estrategia de pivoteo no válida: {pivoteo!r}. Use 'parcial' o 'primero'.")

    gauss_a, gauss_b = _validar_sistema_general(a, b)
    n = gauss_a.shape[0]

    for k in range(n - 1):
        ik = k + _buscar_pivote(gauss_a[k:, k], pivoteo, tol)
        if ik != k:
            gauss_a[[k, ik]] = gauss_a[[ik, k]]
            gauss_b[[k, ik]] = gauss_b[[ik, k]]

        factors = gauss_a[k + 1:, k] / gauss_a[k, k]
        gauss_a[k + 1:, k + 1:] -= np.outer(factors, gauss_a[k, k + 1:])
        if gauss_b.ndim == 1:
            gauss_b[k + 1:] -= factors * gauss_b[k]
        else:
            gauss_b[k + 1:] -= np.outer(factors, gauss_b[k])

    return remonte(gauss_a, gauss_b, tol=tol)


def gauss_jordan(a: ArrayLike,b: ArrayLike,pivoteo: str = "parcial",tol: float = 1e-15,) -> np.ndarray:
    """Resuelve A X = B por Gauss-Jordan ('parcial' o 'primero')."""
    if pivoteo not in ("parcial", "primero"):
        raise ValueError(f"Estrategia de pivoteo no válida: {pivoteo!r}. Use 'parcial' o 'primero'.")

    gauss_a, gauss_b = _validar_sistema_general(a, b)
    n = gauss_a.shape[0]

    for k in range(n):
        ik = k + _buscar_pivote(gauss_a[k:, k], pivoteo, tol)
        if ik != k:
            gauss_a[[k, ik]] = gauss_a[[ik, k]]
            gauss_b[[k, ik]] = gauss_b[[ik, k]]

        piv = gauss_a[k, k]
        gauss_a[k, k + 1:] /= piv
        gauss_b[k] /= piv

        factors = gauss_a[:, k].copy()
        factors[k] = 0.0
        gauss_a[:, k + 1:] -= np.outer(factors, gauss_a[k, k + 1:])
        if gauss_b.ndim == 1:
            gauss_b -= factors * gauss_b[k]
        else:
            gauss_b -= np.outer(factors, gauss_b[k])

    return gauss_b

def inversa(a: ArrayLike, tol: float = 1e-15) -> np.ndarray:
    """Calcula la matriz inversa A^{-1} resolviendo A X = I con pivoteo parcial."""
    mat = _validar_matriz_cuadrada(a)
    identidad = np.eye(mat.shape[0], dtype=mat.dtype)
    return gauss(mat, identidad, pivoteo="parcial", tol=tol)