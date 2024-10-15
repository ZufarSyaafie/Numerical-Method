import sys
import numpy as np

class GEPP():
    """
    Gaussian elimination with partial pivoting.

    input: A is an n x n numpy matrix
           b is an n x 1 numpy array
    output: x is the solution of Ax=b
            with the entries permuted in
            accordance with the pivoting
            done by the algorithm
    post-condition: A and b have been modified.
    :return
    """

    def __init__(self, A, b, doPricing=True):
        #super(GEPP, self).__init__()

        self.A = A                      # input: A is an n x n numpy matrix
        self.b = b                      # b is an n x 1 numpy array
        self.doPricing = doPricing

        self.n = None                   # n is the length of A
        self.x = None                   # x is the solution of Ax=b

        self._validate_input()          # method that validates input
        self._elimination()             # method that conducts elimination
        self._backsub()                 # method that conducts back-substitution

    def _validate_input(self):
        self.n = len(self.A)
        if self.b.size != self.n:
            raise ValueError("Invalid argument: incompatible sizes between" +
                             "A & b.", self.b.size, self.n)

    def _elimination(self):
        """
        k represents the current pivot row. Since GE traverses the matrix in the
        upper right triangle, we also use k for indicating the k-th diagonal
        column index.
        :return
        """

        # Elimination
        for k in range(self.n - 1):
            if self.doPricing:
                # Pivot
                maxindex = abs(self.A[k:, k]).argmax() + k
                if self.A[maxindex, k] == 0:
                    raise ValueError("Matrix is singular.")
                # Swap
                if maxindex != k:
                    self.A[[k, maxindex]] = self.A[[maxindex, k]]
                    self.b[[k, maxindex]] = self.b[[maxindex, k]]
            else:
                if abs(self.A[k, k]) <= sys.float_info.epsilon:
                    raise ValueError("Pivot element is zero. Try setting doPricing to True.")
            # Eliminate
            for row in range(k + 1, self.n):
                multiplier = self.A[row, k] / self.A[k, k]
                self.A[row, k:] = self.A[row, k:] - multiplier * self.A[k, k:]
                self.b[row] = self.b[row] - multiplier * self.b[k]

    def _backsub(self):
        # Back Substitution

        self.x = np.zeros(self.n)
        for k in range(self.n - 1, -1, -1):
            self.x[k] = (self.b[k] - np.dot(self.A[k, k + 1:], self.x[k + 1:])) / self.A[k, k]
