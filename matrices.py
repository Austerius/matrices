import decimal
import errors
import itertools  # required for zip_longest function
import numbers  # required to check number type
from math import factorial
decimal.getcontext().prec = 5  # precision of the matrix decimal elements equal to 5 numbers after '.'


class Matrix(object):
    """Basic matrix class
        Has numeric decimal type of values
    """
    def __init__(self, list_of_lists, columns=None, rows=None,):
        """matrix initialization method
            "list_of_lists" parameter should look like this: [[1, 2, 3] [4, 5, 6]]
            which represent matrix with 2 rows and 3 columns
            (number of lists - its "m" matrix's parameter which represent rows
            number of elements in each list - its "n" matrix's parameter and represent columns)
            P.S.: for some reason I mistook columns and rows in class initialization, so
            parameter 'columns' actually represent rows and 'rows' - columns
        """

        if columns is None:
            try:
                self.m = len(list_of_lists)
            except TypeError:
                raise errors.WrongInputType("Should be list of lists type: [[],[]]")
        else:
            self.m = columns
        if rows is None:
            for arg in list_of_lists:  # this is redundant check
                try:
                    self.n = len(arg)
                except TypeError:
                    raise errors.WrongInputType("Should be list of lists type: [[],[]]")
        else:
            self.n = rows
        self.A = []
        self._matrix_input_checker(list_of_lists)

        # Here we initialize matrix A with parameter list_of_lists
        for column in list_of_lists:
            temp_arg = []
            for arg in column:
                if not isinstance(arg, decimal.Decimal):
                    temp_arg.append(decimal.Decimal(repr(arg)))  # transformation values in list_of_lists to Decimal
                else:
                    temp_arg.append(arg)
            self.A.append(temp_arg)

    def __iter__(self):
        """Here we make our matrix iterable by implementing __iter__ method"""
        for row in self.A:
            yield row

    def _matrix_input_checker(self, input_matrix):
        """Method for checking inputted data to be relevant type and in same dimension
        """
        # checker for list in list type: [[]]
        try:
            if not (all(isinstance(column, list) for column in input_matrix)):
                raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        except TypeError:
            raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        # checking if inputted matrix has valid dimensions (if columns == m)
        # print("Number of columns: {}".format(len(input_matrix)))
        try:
            if self.m != len(input_matrix):
                raise errors.WrongDimension("Wrong number of columns for initialization")
        except TypeError:
            raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        # Checking values in inputted matrix for numeric type
        for row in input_matrix:
            # Checking for relevant number of rows
            # print("Number of rows: {}: ".format(len(row)))
            if self.n != len(row):
                raise errors.WrongDimension("Wrong number of rows for initialization")
            for num in row:
                if not isinstance(num, numbers.Number):
                    raise errors.WrongElementType("Should be a numeric type!")

    def is_zero_matrix(self):
        """Checking if matrix is a Null matrix
        return True if so
        """
        for column in self.A:
            for arg in column:
                if arg != 0:
                    return False
        return True

    def is_square_matrix(self):
        """Checking if matrix is square matrix: m=n"""
        if self.m == self.n:
            return True
        else:
            return False

    def is_diagonal_matrix(self):
        """Checking if matrix is a diagonal matrix: has all elements equal zero except of main diagonal
           WE presume that diagonal matrix has to be also square matrix
        """
        if self.is_square_matrix():
            j = 0
            i = 0
            for columns in self.A:
                for row in columns:
                    if row == 0 and i == j:
                        return False
                    elif row != 0 and i != j:
                        return False
                    j += 1
                i += 1
                j = 0
            return True
        else:
            #  Here we raised an error because non square matrices does have main diagonal, so we explicitly warned
            #  users about needs of square type matrix, but we rewrote it to return just False
            return False
            # raise errors.WrongInputType("Need a square matrix here")

    def is_identity_matrix(self):
        """Checking if matrix is identity matrix: diagonal square matrix, where all elements of the main
            diagonal equals to one. Return True if so
        """
        if self.is_diagonal_matrix():
            i = 0
            j = 0
            for columns in self.A:
                for arg in columns:
                    if arg != 1 and i == j:
                        return False
                    j += 1
                i += 1
                j = 0
            return True
        else:
            return False  # Just returning False here but could be an error like in commented code below
            #  raise errors.WrongInputType("Not diagonal matrix cant be identity matrix")

    def matrix_is_equal(self, matrix_for_compare):
        """ Method for compering two matrices. If they equal its return True, otherwise - False
            Parameter 'matrix_for_compare)' is Matrix() class instance.
            One matrix is equal to another if all of their components are equal
        """
        if not isinstance(matrix_for_compare, Matrix):
            raise errors.WrongInputType("Input parameter should be a Matrix type")
        if self.m != matrix_for_compare.m:
            return False
        if self.n != matrix_for_compare.n:
            return False
        for column1, column2 in zip(self.A, matrix_for_compare):
            for arg1, arg2 in zip(column1, column2):
                if arg1 != arg2:
                    return False

        return True

    def matrix_transposition(self):
        """THis method will Transpose matrix self.A and return new transposed matrix """
        #  here we creating zero matrix with n columns and m rows
        transpose_list = [[0 for x in range(self.m)] for y in range(self.n)]
        # this is how transposition works: a[i][j] == at[j][i], where at - transpose matrix
        for columns, i in zip(self.A, range(0, self.m)):
            for arg, j in zip(columns, range(0, self.n)):
                transpose_list[j].pop(i)  # need to remove 0 from position before inserting real value
                transpose_list[j].insert(i, arg)  # if 0 not removed - it will shifted position and expand matrix
        # print(transpose_list)
        return Matrix(transpose_list)

    def matrix_is_transpose(self, matrix_for_compare):
        """Here we checking if matrix self.A is transpose to matrix_for_compare
            Returning True if so"""
        if not isinstance(matrix_for_compare, Matrix):
            raise errors.WrongInputType("Input parameter should be a Matrix type")
        if self.m != matrix_for_compare.n:
            return False
        if self.n != matrix_for_compare.m:
            return False
        if self.matrix_is_equal(matrix_for_compare.matrix_transposition()):
            return True
        return False

    def matrix_addition(self, matrix_for_add):
        """ This method will add 'matrix_for_add' to the self.A matrix and return results in form of new Matrix.
            Note, that 'matrix_for_add' need to be a Matrix() type. Also, to add one matrix to another,
            they both need to have same dimensions
        """
        if not isinstance(matrix_for_add, Matrix):
            raise errors.WrongInputType("matrix_for_add parameter should be a Matrix() type")
        if self.m != matrix_for_add.m:
            raise errors.WrongDimension("Matrices doesnt have same number of rows")
        if self.n != matrix_for_add.n:
            raise errors.WrongDimension("Matrices should have same number of columns")
        list_matrix_add = []
        for rows1, rows2 in zip(self.A, matrix_for_add):
            temp_list = []
            for arg1, arg2 in zip(rows1, rows2):
                temp_list.append(arg1 + arg2)
            list_matrix_add.append(temp_list)
        return Matrix(list_matrix_add)

    def multiply_by_number(self, number):
        """ Method for scalar multiplication matrix A by parameter 'number'(should be a numeric type)
            For multiply matrix by some number, we actually need to multiply all her elements by this number
            After we done with multiplication, a new matrix will be returned"""
        if not isinstance(number, numbers.Number):
            raise errors.WrongInputType("Parameter 'number' should be a numeric type!")
        # converting number to decimal format for consistence
        if not isinstance(number, decimal.Decimal):
            number_dec = decimal.Decimal(repr(number))
        else:
            number_dec = number
        return_list = []
        for rows in self.A:
            temp_list = []
            for arg in rows:
                temp_list.append(arg*number_dec)
            return_list.append(temp_list)
        return Matrix(return_list)

    def matrix_subtraction(self, matrix_for_subtract):
        """ THis method will subtract incoming matrix(parameter 'matrix_for_subtract' which is Matrix type)
            from self.A matrix. Code for this class is similar(actually the same) as for matrix_addition method,
            except when we need add arg2 to agg1 - we subtract it. Method could be implemented as call of
            matrix_addition with 'matrix_for add' multiplied by '-1'(check the commented code)"""
        # return self.matrix_addition(matrix_for_subtract.multiply_by_number(-1))
        if not isinstance(matrix_for_subtract, Matrix):
            raise errors.WrongInputType("matrix_for_subtract parameter should be a Matrix() type")
        if self.m != matrix_for_subtract.m:
            raise errors.WrongDimension("Matrices doesnt have same number of rows")
        if self.n != matrix_for_subtract.n:
            raise errors.WrongDimension("Matrices should have same number of columns")
        list_matrix_subtract = []
        for rows1, rows2 in zip(self.A, matrix_for_subtract):
            temp_list = []
            for arg1, arg2 in zip(rows1, rows2):
                temp_list.append(arg1 - arg2)
            list_matrix_subtract.append(temp_list)
        return Matrix(list_matrix_subtract)

    def divide_by_number(self, number):
        """ Dividing matrix self.A by 'number' and returning new matrix
            Basically, we need to call multiply_by_number method with income parameter: 1/number """
        return self.multiply_by_number(1/number)

    def matrix_multiplication(self, matrix_for_multiplication):
        """ Method for multiplying two matrices. Input parameter 'matrix_for_multiplication' should be Matrix()
            type. Also, input matrix should have same number of rows as number of columns in self.A matrix.
            In another words: self.n == matrix_for_multiplication.m """
        if not isinstance(matrix_for_multiplication, Matrix):
            raise errors.WrongInputType("matrix_for_multiplication should be a Matrix() type")
        if self.n != matrix_for_multiplication.m:
            raise errors.WrongDimension("Matrices dimensions not compatible")
        multiplication_list = []
        # c[i][k] = sum(a[i][j]*b[j][k])
        for rows in self.A:
            temp_list = []
            for i in range(0, matrix_for_multiplication.n):
                temp = 0
                for arg1, rows2 in itertools.zip_longest(rows, matrix_for_multiplication):
                    temp += arg1*rows2[i]
                temp_list.append(temp)
            multiplication_list.append(temp_list)
        return Matrix(multiplication_list)

    # TODO need to write method for calculating determinant, based on elementary rows manipulation
    def matrix_determinant(self):
        """ Here we will compute a determinant for square matrix self.A recursively
            Better to use this for 3 and less dimensions matrix (or up to 6 dimensions; for calculating 8 dimensional
            matrix this method will need several minutes)"""
        if not self.is_square_matrix():
            raise errors.WrongDimension("Determinant can be computed only for square matrix")
        det = 0
        if self.m == 1:
            det = self.A[0][0]
        elif self.m == 2:
            det = self.A[0][0]*self.A[1][1] - self.A[0][1]*self.A[1][0]
        elif self.m == 3:
            det = self.A[0][0]*self.A[1][1]*self.A[2][2] + self.A[0][1]*self.A[1][2]*self.A[2][0] +\
                    self.A[1][0]*self.A[2][1]*self.A[0][2] - self.A[2][0]*self.A[1][1]*self.A[0][2] -\
                    self.A[1][0]*self.A[0][1]*self.A[2][2] - self.A[2][1]*self.A[1][2]*self.A[0][0]
        else:
            det += self._matrix_determinant_n(self.A)
            return det/factorial(self.m - 3)  # for some reason we have deviation = (n-3)! where n - matrix dimension
        return det

    def _matrix_determinant_n(self, matrix_n):
        """ Private method, that supposed to calculate determinant for n-dimensional matrix recursively
            Its used in matrix_determinant method, when dimension of matrix greater than 3"""
        det = 0
        dimension = len(matrix_n)
        if dimension == 3:
            det = Matrix(matrix_n).matrix_determinant()
        while dimension > 3:
            for i in range(0, len(matrix_n)):
                temp_matrix_list = []
                for rows, k in zip(matrix_n, range(0, len(matrix_n))):
                    temp = []
                    if k == 0:
                        continue
                    for arg, l in zip(rows, range(0, len(matrix_n))):
                        if l == i:
                            continue
                        temp.append(arg)
                    temp_matrix_list.append(temp)
                new_matrix_n = temp_matrix_list
                det += (matrix_n[0][i])*self._matrix_determinant_n(new_matrix_n)*(-1)**i
            dimension -= 1
        return det
    # def __getitem__(self, item):
    #     return self.A

    def matrix_show(self):
            """ Simple method to print a matrix
                (Will be updated soon)
            """
            for row in self.A:
                print(row)


class ZeroMatrix(Matrix):
    """class for initialization zero matrix"""
    def __init__(self, columns, rows):
        self.zero_matrix = []
        try:
            for i in range(0, columns):
                temp = []
                for j in range(0, rows):
                    temp.append(0)
                self.zero_matrix.append(temp)
        except TypeError:
            raise errors.WrongInputType("Columns and rows should be an integer type")
        super().__init__(columns=columns, rows=rows, list_of_lists=self.zero_matrix)


class IdentityMatrix(Matrix):
    """ class for initialization an identity matrix: square, diagonal matrix where main diagonal consist of '1'
        'dimension' parameter represent dimension of a matrix or(in another words) number of columns/rows
    """
    def __init__(self, dimension):
        self.identity_matrix = []
        try:
            for i in range(0, dimension):
                temp = []
                for j in range(0, dimension):
                    if i == j:
                        temp.append(1)
                    else:
                        temp.append(0)
                self.identity_matrix.append(temp)
        except TypeError:
            raise errors.WrongInputType("Dimension should be an integer type")
        super(IdentityMatrix, self).__init__(columns=dimension, rows=dimension, list_of_lists=self.identity_matrix)

if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6.789]]
    A = Matrix(matrix, 2, 3)
    A.matrix_show()
    print(A)
    B = Matrix([[0], [0]])
    print(A.is_zero_matrix())
    print(B.is_zero_matrix())
    print(Matrix.is_zero_matrix(B))
    print(decimal.Decimal(repr(24324.2)))
    O = ZeroMatrix(2, 3)
    O.matrix_show()
    print("=" * 45)
    I = IdentityMatrix(3)
    I.matrix_show()
    state = A.matrix_is_equal(O)
    print(state)
    AT = A.matrix_transposition()
    AT.matrix_show()
    print(A.matrix_is_transpose(AT))
    D = Matrix([[0, 2, 3], [3, 5, -7]])
    E = Matrix([[-1, -2, 4], [-4, 5, 8]])
    DE = D.matrix_addition(E)
    DE.matrix_show()
    F = D.multiply_by_number(2)
    F.matrix_show()
    print("="*50)
    G = Matrix([[3]])
    print(G.matrix_determinant())
    J = Matrix([[1, 4],
                [3, 6]])
    print(J.matrix_determinant())
    K = Matrix([[5, -2, 1],
                [3, 1, -4],
                [6, 0, -3]])
    print(K.matrix_determinant())
    L = Matrix([[3, 5, 7, 8],
                [-1, 7, 0, 1],
                [0, 5, 3, 2],
                [1, -1, 7, 4]])
    print(L.matrix_determinant())
    M = Matrix([[1, 2, 0, 0, 0],
                [3, 2, 3, 0, 0],
                [0, 4, 3, 4, 0],
                [0, 0, 5, 4, 5],
                [0, 0, 0, 6, 5]])
    N = Matrix([[0, 6, -2, -1, 5],
                [0, 0, 0, -9, -7],
                [0, 15, 35, 0, 0],
                [0, -1, -11, -2, 1],
                [-2, -2, 3, 0, -2]])
    print(M.matrix_determinant())
    print(N.matrix_determinant())
    P = Matrix([[1, 2, 3, 0, 0, 0],
                [4, 3, 0, 0, -1, -2],
                [1, 2, 1, 1, 1, 0],
                [3, -2, -2, 0, 0, 0],
                [4, 1, -1, -5, 5, 0],
                [0, 0, 6, 5, 4, 0]])
    print(P.matrix_determinant())
    R = Matrix([[2, 0, 4, 0, 5, 0, 1],
                [0, 3, -5, -1, 2, 0, 3],
                [4, 5, 1, 1, 2, 3, 0],
                [2, -4, 4, 4, 4, 2, 2],
                [2, 0, 3, 4, 1, 3, 0],
                [0, -5, -5, 6, 1, 4, 0],
                [3, 0, 4, -1, 2, 0, 7]])
    print(R.matrix_determinant())






