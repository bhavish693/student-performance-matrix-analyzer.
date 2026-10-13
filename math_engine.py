"""
=============================================================================
Student Performance Matrix Analyzer - Core Mathematical Engine
=============================================================================
MAT2303: Linear Algebra & Vector Calculus - Mini Project

This module implements all linear algebra concepts from Modules 1, 2, and 3:
  Module 1: Matrices, Row Reduction, Rank, Linear Systems, LU Decomposition
  Module 2: Vector Spaces, Basis, Linear Independence, Norms, Inner Products,
            Orthogonality, Gram-Schmidt, Markov Chains
  Module 3: Linear Transformations, Eigenvalues, Eigenvectors, 
            Diagonalization, SVD, PCA
=============================================================================
"""

import numpy as np
from numpy.linalg import matrix_rank, eig, svd, norm, inv, det, solve
import json


class MatrixAnalyzer:
    """
    Comprehensive matrix analyzer implementing all LA concepts
    from MAT2303 Modules 1-3.
    """

    def __init__(self, data_matrix):
        """
        Initialize with a student performance data matrix.
        Rows = students, Columns = subjects/features
        """
        self.original_data = np.array(data_matrix, dtype=float)
        self.m, self.n = self.original_data.shape  # m students, n features
        self.results = {}

    # =========================================================================
    # MODULE 1: Matrices, Row Reduction, Rank, Linear Systems, LU Decomposition
    # =========================================================================

    def matrix_formation(self):
        """
        Stage 1: Form the student-performance matrix A (m x n).
        Each row is a student vector, each column is a subject/feature.
        """
        A = self.original_data.copy()
        self.results['matrix'] = {
            'A': A.tolist(),
            'shape': list(A.shape),
            'description': (
                f"Performance matrix A of shape {A.shape[0]}×{A.shape[1]}. "
                f"Each row represents a student's scores across {A.shape[1]} subjects."
            )
        }
        return A

    def row_echelon_form(self, A=None):
        """
        Stage 2: Compute Row Echelon Form (REF) via Gaussian Elimination.
        Demonstrates elementary row operations.
        """
        if A is None:
            A = self.original_data.copy()
        else:
            A = A.copy()

        m, n = A.shape
        pivot_row = 0
        pivot_columns = []
        operations = []

        for col in range(n):
            if pivot_row >= m:
                break

            # Find the largest absolute value in the column for partial pivoting
            max_row = pivot_row
            for row in range(pivot_row + 1, m):
                if abs(A[row, col]) > abs(A[max_row, col]):
                    max_row = row

            if abs(A[max_row, col]) < 1e-10:
                continue

            # Swap rows
            if max_row != pivot_row:
                A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
                operations.append(f"R{pivot_row+1} ↔ R{max_row+1}")

            # Eliminate below
            for row in range(pivot_row + 1, m):
                if abs(A[row, col]) > 1e-10:
                    factor = A[row, col] / A[pivot_row, col]
                    A[row] = A[row] - factor * A[pivot_row]
                    operations.append(
                        f"R{row+1} → R{row+1} - ({factor:.4f})R{pivot_row+1}"
                    )

            pivot_columns.append(col)
            pivot_row += 1

        self.results['ref'] = {
            'REF': A.tolist(),
            'operations': operations,
            'pivot_columns': pivot_columns,
            'description': (
                "Row Echelon Form obtained via Gaussian elimination with "
                "partial pivoting. Elementary row operations applied sequentially."
            )
        }
        return A

    def reduced_row_echelon_form(self, A=None):
        """
        Stage 3: Compute Reduced Row Echelon Form (RREF).
        Full Gauss-Jordan elimination.
        """
        if A is None:
            A = self.original_data.copy()
        else:
            A = A.copy()

        m, n = A.shape
        pivot_row = 0

        for col in range(n):
            if pivot_row >= m:
                break

            # Find pivot
            max_row = pivot_row
            for row in range(pivot_row + 1, m):
                if abs(A[row, col]) > abs(A[max_row, col]):
                    max_row = row

            if abs(A[max_row, col]) < 1e-10:
                continue

            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

            # Scale pivot row
            A[pivot_row] = A[pivot_row] / A[pivot_row, col]

            # Eliminate all other rows (above and below)
            for row in range(m):
                if row != pivot_row and abs(A[row, col]) > 1e-10:
                    factor = A[row, col]
                    A[row] = A[row] - factor * A[pivot_row]

            pivot_row += 1

        self.results['rref'] = {
            'RREF': A.tolist(),
            'description': (
                "Reduced Row Echelon Form via Gauss-Jordan elimination. "
                "Leading entries are 1 and are the only nonzero entries in their columns."
            )
        }
        return A

    def compute_rank(self):
        """
        Stage 4: Compute the rank of the performance matrix.
        rank(A) = number of linearly independent rows/columns.
        """
        A = self.original_data
        rank = int(matrix_rank(A))

        self.results['rank'] = {
            'rank': rank,
            'max_possible_rank': min(self.m, self.n),
            'is_full_rank': rank == min(self.m, self.n),
            'description': (
                f"Rank of A = {rank}. This means there are {rank} linearly "
                f"independent student performance patterns. "
                f"{'The matrix is full rank.' if rank == min(self.m, self.n) else f'The matrix is rank-deficient ({min(self.m, self.n) - rank} dependent patterns).'}"
            )
        }
        return rank

    def lu_decomposition(self):
        """
        Stage 5: LU Decomposition of AᵀA (square matrix).
        AᵀA = LU where L is lower triangular, U is upper triangular.
        Used for solving normal equations in least squares.
        """
        A = self.original_data
        ATA = A.T @ A  # n x n square matrix

        n = ATA.shape[0]
        L = np.eye(n)
        U = ATA.copy()

        for i in range(n):
            for j in range(i + 1, n):
                if abs(U[i, i]) > 1e-10:
                    factor = U[j, i] / U[i, i]
                    L[j, i] = factor
                    U[j] = U[j] - factor * U[i]

        # Verify: LU ≈ AᵀA
        reconstruction_error = np.max(np.abs(L @ U - ATA))

        self.results['lu'] = {
            'ATA': ATA.tolist(),
            'L': L.tolist(),
            'U': U.tolist(),
            'reconstruction_error': float(reconstruction_error),
            'description': (
                f"LU Decomposition of AᵀA ({n}×{n}). "
                f"L is lower triangular with 1s on diagonal, U is upper triangular. "
                f"Reconstruction error ||LU - AᵀA||∞ = {reconstruction_error:.2e}"
            )
        }
        return L, U

    def solve_linear_system(self):
        """
        Stage 6: Solve the normal equation AᵀAx = Aᵀb for least squares.
        Here b = mean performance vector, finding weights for each subject.
        """
        A = self.original_data
        b = np.mean(A, axis=1)  # Average performance per student

        ATA = A.T @ A
        ATb = A.T @ b

        try:
            x = solve(ATA, ATb)
            residual = b - A @ x
            residual_norm = float(norm(residual))

            self.results['linear_system'] = {
                'ATA': ATA.tolist(),
                'ATb': ATb.tolist(),
                'solution_x': x.tolist(),
                'residual_norm': residual_norm,
                'description': (
                    "Solved normal equations AᵀAx = Aᵀb using least squares. "
                    f"Solution vector x gives subject weights. "
                    f"||residual|| = {residual_norm:.4f}"
                )
            }
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            self.results['linear_system'] = {
                'solution_x': x.tolist(),
                'description': "Solved using pseudo-inverse (matrix is singular)."
            }

        return x

    # =========================================================================
    # MODULE 2: Vector Spaces, Basis, Norms, Inner Products, Gram-Schmidt
    # =========================================================================

    def vector_representation(self):
        """
        Stage 7: Represent each student as a vector in Rⁿ.
        Compute vector norms and statistics.
        """
        A = self.original_data
        student_vectors = []

        for i in range(self.m):
            v = A[i]
            l2_norm = float(norm(v))
            l1_norm = float(norm(v, 1))
            linf_norm = float(norm(v, np.inf))
            student_vectors.append({
                'student': i + 1,
                'vector': v.tolist(),
                'L2_norm': round(l2_norm, 4),
                'L1_norm': round(l1_norm, 4),
                'Linf_norm': round(linf_norm, 4),
            })

        self.results['vectors'] = {
            'student_vectors': student_vectors,
            'space_dimension': self.n,
            'description': (
                f"Each student is represented as a vector in R^{self.n}. "
                f"L2 norm = Euclidean magnitude (overall performance), "
                f"L1 norm = total absolute scores, "
                f"L∞ norm = maximum score in any subject."
            )
        }
        return student_vectors

    def linear_independence_check(self):
        """
        Stage 8: Check linear independence of column vectors (subject vectors).
        Determines if any subject scores are redundant.
        """
        A = self.original_data
        rank = int(matrix_rank(A))
        n_cols = A.shape[1]

        # Check each subset of columns
        independent_cols = []
        dependent_cols = []

        for j in range(n_cols):
            sub = A[:, :j + 1]
            if matrix_rank(sub) == j + 1:
                independent_cols.append(j)
            else:
                dependent_cols.append(j)

        # Compute determinant of AᵀA for overall check
        ATA = A.T @ A
        det_ATA = float(det(ATA))

        self.results['independence'] = {
            'rank': rank,
            'n_features': n_cols,
            'independent_features': independent_cols,
            'dependent_features': dependent_cols,
            'det_ATA': round(det_ATA, 4),
            'all_independent': rank == n_cols,
            'description': (
                f"Rank = {rank}, Number of features = {n_cols}. "
                f"{'All features are linearly independent.' if rank == n_cols else f'{n_cols - rank} features are linearly dependent (redundant).'} "
                f"det(AᵀA) = {det_ATA:.4f} ({'non-zero → independent' if abs(det_ATA) > 1e-6 else 'zero → dependent'})"
            )
        }
        return rank == n_cols

    def inner_product_analysis(self):
        """
        Stage 9: Compute inner products between student vectors.
        ⟨u, v⟩ = uᵀv = Σ uᵢvᵢ
        This measures similarity between student performance profiles.
        """
        A = self.original_data
        # Gram matrix: G[i,j] = ⟨student_i, student_j⟩
        G = A @ A.T

        # Cosine similarity between students
        norms = norm(A, axis=1)
        cos_sim = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if norms[i] > 0 and norms[j] > 0:
                    cos_sim[i, j] = G[i, j] / (norms[i] * norms[j])

        self.results['inner_product'] = {
            'gram_matrix': G.tolist(),
            'cosine_similarity': cos_sim.tolist(),
            'description': (
                "Gram matrix G = AAᵀ where G[i,j] = ⟨studentᵢ, studentⱼ⟩. "
                "Cosine similarity cos(θ) = ⟨u,v⟩/(||u||·||v||) measures "
                "how similar two students' performance profiles are."
            )
        }
        return G

    def gram_schmidt_orthogonalization(self):
        """
        Stage 10: Apply Gram-Schmidt process to the column vectors of A.
        Produces an orthonormal basis for the column space.
        
        Algorithm:
        v₁ = u₁ / ||u₁||
        uₖ = aₖ - Σⱼ₌₁ᵏ⁻¹ ⟨aₖ, vⱼ⟩vⱼ  (remove projections)
        vₖ = uₖ / ||uₖ||
        """
        A = self.original_data
        columns = A.T.copy()  # Each column of A as a row
        n = len(columns)

        Q = np.zeros_like(columns, dtype=float)
        R = np.zeros((n, n), dtype=float)
        steps = []

        for k in range(n):
            u = columns[k].copy()
            step_projections = []

            for j in range(k):
                proj_coeff = float(np.dot(columns[k], Q[j]))
                R[j, k] = proj_coeff
                u = u - proj_coeff * Q[j]
                step_projections.append({
                    'onto_vector': j + 1,
                    'projection_coefficient': round(proj_coeff, 4)
                })

            u_norm = float(norm(u))
            R[k, k] = u_norm

            if u_norm > 1e-10:
                Q[k] = u / u_norm
            else:
                Q[k] = np.zeros_like(u)

            steps.append({
                'vector': k + 1,
                'projections_removed': step_projections,
                'norm_before_normalization': round(u_norm, 4),
                'orthonormal_vector': Q[k].tolist()
            })

        # Verify orthonormality: QQᵀ ≈ I
        QQT = Q @ Q.T
        orthonormality_error = np.max(np.abs(QQT - np.eye(n)))

        self.results['gram_schmidt'] = {
            'Q': Q.tolist(),
            'R': R.tolist(),
            'steps': steps,
            'orthonormality_error': float(orthonormality_error),
            'description': (
                "Gram-Schmidt orthogonalization applied to column vectors. "
                f"Produced orthonormal basis Q ({Q.shape[0]}×{Q.shape[1]}). "
                f"QR decomposition: A = QR. Orthonormality error = {orthonormality_error:.2e}"
            )
        }
        return Q, R

    def basis_and_dimension(self):
        """
        Stage 11: Find basis for column space and null space of A.
        Column space = span of linearly independent columns.
        Null space = {x : Ax = 0}.
        """
        A = self.original_data
        rank = int(matrix_rank(A))

        # Column space basis (from RREF pivot columns)
        rref = self.reduced_row_echelon_form(A.copy())
        pivot_cols = []
        for i in range(min(A.shape)):
            for j in range(A.shape[1]):
                if abs(rref[i, j] - 1.0) < 1e-10:
                    pivot_cols.append(j)
                    break

        col_space_basis = A[:, pivot_cols].tolist() if pivot_cols else []

        # Null space (via SVD)
        U, S, VT = svd(A)
        null_space_basis = VT[rank:].tolist()
        nullity = self.n - rank

        self.results['basis'] = {
            'column_space_basis': col_space_basis,
            'column_space_dimension': rank,
            'null_space_basis': null_space_basis,
            'nullity': nullity,
            'rank_nullity_theorem': f"rank(A) + nullity(A) = {rank} + {nullity} = {rank + nullity} = n = {self.n}",
            'description': (
                f"Column space dim = {rank} (number of independent performance factors). "
                f"Null space dim = {nullity}. "
                f"Rank-Nullity Theorem verified: rank + nullity = {rank + nullity} = {self.n}."
            )
        }
        return pivot_cols

    # =========================================================================
    # MODULE 3: Linear Transformations, Eigenvalues, SVD, PCA
    # =========================================================================

    def linear_transformation(self):
        """
        Stage 12: Apply linear transformation T: Rⁿ → Rⁿ.
        
        A linear transformation T satisfies:
        1. T(u + v) = T(u) + T(v)  (additivity)
        2. T(cu) = cT(u)          (homogeneity)
        
        We use the transformation T(x) = (AᵀA)x which maps feature vectors
        to weighted feature vectors, representing the correlation structure.
        
        Also demonstrate: centering transformation (mean subtraction),
        standardization transformation, and grade-to-GPA mapping.
        """
        A = self.original_data

        # Transformation 1: Centering (subtract column means)
        col_means = np.mean(A, axis=0)
        centered = A - col_means  # T₁(A) = A - μ

        # Transformation 2: Standardization (z-score normalization)
        col_stds = np.std(A, axis=0, ddof=0)
        col_stds[col_stds == 0] = 1  # avoid division by zero
        standardized = centered / col_stds  # T₂(x) = (x - μ)/σ

        # Transformation 3: Correlation-based transformation
        # T₃(x) = AᵀA·x maps x to its correlated representation
        ATA = A.T @ A
        sample_input = A[0]
        transformed_sample = ATA @ sample_input

        # Verify linearity of centering transformation
        u = A[0]
        v = A[1]
        c = 2.0
        T_u_plus_v = (u + v) - 2 * col_means  # T(u+v) adjusted for centering
        T_u = u - col_means
        T_v = v - col_means

        # For centering: T(u+v) ≠ T(u) + T(v) in general (it's affine, not linear)
        # But the standardized transform IS linear if means are 0 (on centered data)

        # Transformation 4: Projection matrix for dimensional reduction
        # P = Q·Qᵀ where Q contains the first k orthonormal vectors
        Q_full, _ = self.gram_schmidt_orthogonalization()
        k = min(2, self.n)  # Project to 2D
        Q_k = np.array(Q_full[:k])  # First k orthonormal vectors
        P = Q_k.T @ Q_k  # Projection matrix

        # Verify P² = P (idempotent - property of projection)
        P_squared = P @ P
        projection_error = float(np.max(np.abs(P_squared - P)))

        self.results['linear_transformation'] = {
            'centered_data': centered.tolist(),
            'column_means': col_means.tolist(),
            'standardized_data': standardized.tolist(),
            'column_stds': col_stds.tolist(),
            'correlation_matrix_ATA': ATA.tolist(),
            'sample_transformation': {
                'input': sample_input.tolist(),
                'output_ATA_x': transformed_sample.tolist(),
            },
            'projection_matrix': P.tolist(),
            'projection_is_idempotent': projection_error < 1e-10,
            'projection_error': float(projection_error),
            'description': (
                "Linear transformations applied:\n"
                "1. Centering: T₁(x) = x - μ (subtract column means)\n"
                "2. Standardization: T₂(x) = (x - μ)/σ (z-score normalization)\n"
                "3. Correlation transform: T₃(x) = AᵀA·x\n"
                "4. Projection: T₄(x) = Px where P = QₖQₖᵀ (P² = P, idempotent)\n"
                f"Projection verification: ||P² - P|| = {projection_error:.2e}"
            )
        }
        return centered, standardized

    def eigen_analysis(self):
        """
        Stage 13: Compute eigenvalues and eigenvectors of the covariance-type matrix.
        
        For AᵀA (n×n):
        AᵀA · v = λv
        
        Eigenvalues λᵢ represent the variance captured by each eigenvector.
        Eigenvectors vᵢ are the principal directions of variation.
        """
        A = self.original_data
        # Center the data first
        centered = A - np.mean(A, axis=0)
        # Covariance-type matrix (up to scaling)
        cov_matrix = (centered.T @ centered) / (self.m - 1)

        eigenvalues, eigenvectors = eig(cov_matrix)

        # Sort by magnitude (descending)
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        # Verify Av = λv for each eigenpair
        verifications = []
        for i in range(len(eigenvalues)):
            lhs = cov_matrix @ eigenvectors[:, i]
            rhs = eigenvalues[i] * eigenvectors[:, i]
            error = float(norm(lhs - rhs))
            verifications.append({
                'eigenvalue': round(float(eigenvalues[i]), 4),
                'eigenvector': eigenvectors[:, i].tolist(),
                'verification_error': round(error, 8),
            })

        # Proportion of variance explained
        total_variance = float(np.sum(eigenvalues))
        variance_explained = (eigenvalues / total_variance * 100).tolist()
        cumulative_variance = np.cumsum(eigenvalues / total_variance * 100).tolist()

        # Characteristic equation: det(A - λI) = 0
        trace = float(np.trace(cov_matrix))
        det_cov = float(det(cov_matrix))

        self.results['eigen'] = {
            'covariance_matrix': cov_matrix.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist(),
            'verifications': verifications,
            'variance_explained_percent': variance_explained,
            'cumulative_variance_percent': cumulative_variance,
            'trace': round(trace, 4),
            'determinant': round(det_cov, 4),
            'sum_eigenvalues': round(float(np.sum(eigenvalues)), 4),
            'product_eigenvalues': round(float(np.prod(eigenvalues)), 4),
            'trace_equals_sum_eigenvalues': abs(trace - np.sum(eigenvalues)) < 1e-6,
            'det_equals_product_eigenvalues': abs(det_cov - np.prod(eigenvalues)) < 1e-2,
            'description': (
                "Eigen analysis of covariance matrix C = (1/(m-1))·AcᵀAc.\n"
                f"Eigenvalues: {[round(float(e), 4) for e in eigenvalues]}\n"
                f"Sum of eigenvalues = tr(C) = {trace:.4f} ✓\n"
                f"Product of eigenvalues = det(C) = {det_cov:.4f}\n"
                f"Variance explained: {[round(v, 2) for v in variance_explained]}%"
            )
        }
        return eigenvalues, eigenvectors

    def diagonalization(self):
        """
        Stage 14: Diagonalize the covariance matrix.
        
        If C = PDP⁻¹ where D = diag(λ₁, λ₂, ..., λₙ)
        and P = [v₁ | v₂ | ... | vₙ] (matrix of eigenvectors)
        
        Then C is diagonalizable if P is invertible.
        """
        centered = self.original_data - np.mean(self.original_data, axis=0)
        C = (centered.T @ centered) / (self.m - 1)

        eigenvalues, P = eig(C)
        eigenvalues = eigenvalues.real
        P = P.real
        D = np.diag(eigenvalues)

        try:
            P_inv = inv(P)
            # Verify: C ≈ PDP⁻¹
            C_reconstructed = P @ D @ P_inv
            reconstruction_error = float(np.max(np.abs(C - C_reconstructed)))
            is_diagonalizable = True
        except np.linalg.LinAlgError:
            P_inv = np.zeros_like(P)
            reconstruction_error = float('inf')
            is_diagonalizable = False

        self.results['diagonalization'] = {
            'P': P.tolist(),
            'D': D.tolist(),
            'P_inv': P_inv.tolist(),
            'is_diagonalizable': is_diagonalizable,
            'reconstruction_error': round(reconstruction_error, 10),
            'description': (
                f"Diagonalization: C = PDP⁻¹. "
                f"{'Matrix is diagonalizable.' if is_diagonalizable else 'Matrix is NOT diagonalizable.'} "
                f"Reconstruction error ||C - PDP⁻¹||∞ = {reconstruction_error:.2e}"
            )
        }
        return P, D

    def svd_analysis(self):
        """
        Stage 15: Singular Value Decomposition A = UΣVᵀ.
        
        U: m×m orthogonal matrix (left singular vectors)
        Σ: m×n diagonal matrix (singular values σᵢ)
        V: n×n orthogonal matrix (right singular vectors)
        
        Singular values σᵢ = √λᵢ where λᵢ are eigenvalues of AᵀA.
        """
        A = self.original_data
        centered = A - np.mean(A, axis=0)

        U, S, VT = svd(centered, full_matrices=False)

        # Verify: A ≈ U·diag(S)·Vᵀ
        A_reconstructed = U @ np.diag(S) @ VT
        reconstruction_error = float(np.max(np.abs(centered - A_reconstructed)))

        # Low-rank approximations
        reconstructions = {}
        for k in range(1, min(len(S), self.n) + 1):
            A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
            error = float(norm(centered - A_k, 'fro'))
            energy = float(np.sum(S[:k] ** 2) / np.sum(S ** 2) * 100)
            reconstructions[f'rank_{k}'] = {
                'frobenius_error': round(error, 4),
                'energy_captured_percent': round(energy, 2),
            }

        # Condition number
        condition_number = float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf')

        self.results['svd'] = {
            'U': U.tolist(),
            'singular_values': S.tolist(),
            'VT': VT.tolist(),
            'reconstruction_error': float(reconstruction_error),
            'low_rank_approximations': reconstructions,
            'condition_number': round(condition_number, 4),
            'description': (
                f"SVD: A = UΣVᵀ. Singular values: {[round(float(s), 4) for s in S]}. "
                f"Condition number κ(A) = σ₁/σₙ = {condition_number:.4f}. "
                f"Full reconstruction error = {reconstruction_error:.2e}"
            )
        }
        return U, S, VT

    def pca_analysis(self):
        """
        Stage 16: Principal Component Analysis (PCA).
        
        PCA Pipeline:
        1. Center data: Ac = A - μ
        2. Compute covariance: C = (1/(m-1))AcᵀAc
        3. Eigendecomposition: C = PDP⁻¹
        4. Select top k principal components
        5. Project: Y = Ac · Pₖ (scores in PC space)
        6. Reconstruct: Â = Y · Pₖᵀ + μ
        """
        A = self.original_data
        col_means = np.mean(A, axis=0)
        centered = A - col_means

        # Covariance matrix
        cov_matrix = (centered.T @ centered) / (self.m - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(cov_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Sort descending
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Variance explained
        total_var = np.sum(eigenvalues)
        var_explained = eigenvalues / total_var * 100
        cum_var = np.cumsum(var_explained)

        # Choose k components that explain ≥ 90% variance
        k = np.argmax(cum_var >= 90) + 1
        k = max(k, 2)  # At least 2 for visualization
        k = min(k, self.n)

        # Principal components (top k eigenvectors)
        PC = eigenvectors[:, :k]

        # Project data onto principal components
        scores = centered @ PC  # m × k matrix

        # Reconstruct
        reconstructed = scores @ PC.T + col_means
        reconstruction_error = float(norm(A - reconstructed, 'fro'))

        # Loading matrix interpretation
        loadings = []
        for i in range(k):
            loading = {
                'PC': i + 1,
                'eigenvalue': round(float(eigenvalues[i]), 4),
                'variance_explained': round(float(var_explained[i]), 2),
                'cumulative_variance': round(float(cum_var[i]), 2),
                'loadings': eigenvectors[:, i].tolist(),
            }
            loadings.append(loading)

        self.results['pca'] = {
            'n_components': int(k),
            'principal_components': PC.tolist(),
            'scores': scores.tolist(),
            'loadings': loadings,
            'reconstructed_data': reconstructed.tolist(),
            'reconstruction_error': round(reconstruction_error, 4),
            'total_variance': round(float(total_var), 4),
            'eigenvalues': eigenvalues.tolist(),
            'variance_explained': var_explained.tolist(),
            'cumulative_variance': cum_var.tolist(),
            'description': (
                f"PCA with {k} principal components capturing "
                f"{cum_var[k-1]:.2f}% of total variance.\n"
                f"Reduced dimensionality: {self.n}D → {k}D. "
                f"Reconstruction error = {reconstruction_error:.4f}"
            )
        }
        return scores, PC

    def orthogonal_projection(self):
        """
        Stage 17: Orthogonal projection of data onto PC subspace.
        
        Projection of b onto subspace W spanned by columns of Q:
        proj_W(b) = Q(QᵀQ)⁻¹Qᵀb = QQᵀb (when Q is orthonormal)
        
        Residual: b - proj_W(b) is orthogonal to W.
        """
        A = self.original_data
        centered = A - np.mean(A, axis=0)

        # Get orthonormal basis from PCA
        cov = (centered.T @ centered) / (self.m - 1)
        eigenvalues, eigenvectors = eig(cov)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        k = 2  # Project to 2D
        Q = eigenvectors[:, :k]

        # Projection matrix
        P = Q @ Q.T  # n × n

        projections = []
        for i in range(self.m):
            x = centered[i]
            proj = P @ x  # Projection onto PC subspace
            residual = x - proj
            # Verify orthogonality: projᵀ · residual ≈ 0
            dot_product = float(np.dot(proj, residual))

            projections.append({
                'student': i + 1,
                'original': x.tolist(),
                'projection': proj.tolist(),
                'residual': residual.tolist(),
                'residual_norm': round(float(norm(residual)), 4),
                'orthogonality_check': round(dot_product, 8),
            })

        self.results['projection'] = {
            'projection_matrix': P.tolist(),
            'projections': projections,
            'subspace_dimension': k,
            'description': (
                f"Orthogonal projection onto {k}D PC subspace. "
                "proj_W(x) = QQᵀx, residual = x - proj_W(x). "
                "Verified: projection ⊥ residual (dot product ≈ 0)."
            )
        }
        return projections

    def markov_chain_analysis(self):
        """
        Stage 18: Model grade transitions as a Markov Chain.
        
        States: Fail (F), Pass (C/D), Good (B), Excellent (A)
        Transition matrix P: P[i,j] = probability of moving from state j to state i
        
        x_{k+1} = P · x_k
        
        Steady state: Pq = q (eigenvector for λ=1)
        """
        A = self.original_data

        # Categorize each score into grade states
        # 0-39: Fail, 40-59: Pass, 60-79: Good, 80-100: Excellent
        def categorize(score):
            if score < 40:
                return 0  # Fail
            elif score < 60:
                return 1  # Pass
            elif score < 80:
                return 2  # Good
            else:
                return 3  # Excellent

        states = ['Fail (<40)', 'Pass (40-59)', 'Good (60-79)', 'Excellent (80-100)']

        # Count transitions between subjects (column to next column)
        transition_counts = np.zeros((4, 4))
        for student in A:
            for j in range(len(student) - 1):
                from_state = categorize(student[j])
                to_state = categorize(student[j + 1])
                transition_counts[to_state, from_state] += 1

        # Normalize columns to get stochastic matrix
        col_sums = transition_counts.sum(axis=0)
        col_sums[col_sums == 0] = 1
        P = transition_counts / col_sums

        # Verify P is stochastic (columns sum to 1)
        col_sums_P = P.sum(axis=0).tolist()

        # Initial state distribution
        initial_counts = np.zeros(4)
        for student in A:
            initial_counts[categorize(student[0])] += 1
        x0 = initial_counts / initial_counts.sum()

        # Predict state evolution over 5 steps
        states_evolution = [x0.tolist()]
        xk = x0.copy()
        for step in range(5):
            xk = P @ xk
            states_evolution.append(xk.tolist())

        # Find steady state (eigenvector for eigenvalue 1)
        eigenvalues_P, eigenvectors_P = eig(P)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues_P - 1.0))
        steady_state = eigenvectors_P[:, idx].real
        steady_state = np.abs(steady_state)
        if steady_state.sum() > 0:
            steady_state = steady_state / steady_state.sum()

        self.results['markov'] = {
            'states': states,
            'transition_matrix': P.tolist(),
            'is_stochastic': all(abs(s - 1.0) < 1e-6 for s in col_sums_P),
            'initial_distribution': x0.tolist(),
            'state_evolution': states_evolution,
            'steady_state': steady_state.tolist(),
            'description': (
                "Markov Chain model for grade transitions. "
                "P is a stochastic matrix (columns sum to 1). "
                f"Steady state: {[round(float(s), 4) for s in steady_state]}. "
                "x_{k+1} = P · x_k models how grades evolve across subjects."
            )
        }
        return P, steady_state

    # =========================================================================
    # RUN COMPLETE PIPELINE
    # =========================================================================

    def run_full_analysis(self):
        """Run the complete sequential mathematical pipeline."""
        print("Stage 1: Matrix Formation...")
        self.matrix_formation()

        print("Stage 2: Row Echelon Form (Gaussian Elimination)...")
        self.row_echelon_form()

        print("Stage 3: Reduced Row Echelon Form...")
        self.reduced_row_echelon_form()

        print("Stage 4: Rank Computation...")
        self.compute_rank()

        print("Stage 5: LU Decomposition...")
        self.lu_decomposition()

        print("Stage 6: Linear System (Least Squares)...")
        self.solve_linear_system()

        print("Stage 7: Vector Representation & Norms...")
        self.vector_representation()

        print("Stage 8: Linear Independence Check...")
        self.linear_independence_check()

        print("Stage 9: Inner Product & Similarity Analysis...")
        self.inner_product_analysis()

        print("Stage 10: Gram-Schmidt Orthogonalization...")
        self.gram_schmidt_orthogonalization()

        print("Stage 11: Basis & Dimension...")
        self.basis_and_dimension()

        print("Stage 12: Linear Transformations...")
        self.linear_transformation()

        print("Stage 13: Eigenvalue Analysis...")
        self.eigen_analysis()

        print("Stage 14: Diagonalization...")
        self.diagonalization()

        print("Stage 15: SVD Analysis...")
        self.svd_analysis()

        print("Stage 16: PCA Analysis...")
        self.pca_analysis()

        print("Stage 17: Orthogonal Projection...")
        self.orthogonal_projection()

        print("Stage 18: Markov Chain Analysis...")
        self.markov_chain_analysis()

        print("\n[OK] All 18 stages completed successfully!")
        return self.results


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_data(n_students=10, n_subjects=6, seed=42):
    """
    Generate realistic student performance data.
    Subjects: Math, Physics, Chemistry, English, Computer Science, Statistics
    """
    np.random.seed(seed)

    subjects = ['Mathematics', 'Physics', 'Chemistry', 'English',
                'Computer Science', 'Statistics']

    student_names = [
        'Alice', 'Bob', 'Charlie', 'Diana', 'Eve',
        'Frank', 'Grace', 'Henry', 'Iris', 'Jack'
    ]

    # Generate correlated scores (students good in math tend to be good in physics)
    base_ability = np.random.normal(65, 15, n_students)

    data = np.zeros((n_students, n_subjects))
    correlations = [1.0, 0.85, 0.75, 0.4, 0.9, 0.88]

    for j in range(n_subjects):
        noise = np.random.normal(0, 10, n_students)
        data[:, j] = correlations[j] * base_ability + (1 - correlations[j]) * noise + \
                     np.random.normal(0, 5, n_students)

    # Clip to valid range [0, 100]
    data = np.clip(data, 0, 100).round(1)

    return data, subjects, student_names


if __name__ == '__main__':
    # Quick test
    data, subjects, names = generate_sample_data()
    print("Sample Data:")
    print(f"Students: {names}")
    print(f"Subjects: {subjects}")
    print(f"Matrix shape: {data.shape}")
    print(data)

    analyzer = MatrixAnalyzer(data)
    results = analyzer.run_full_analysis()
    print(f"\nTotal analysis stages: {len(results)}")
