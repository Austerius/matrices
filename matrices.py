import decimal
import errors  # our custom class for handling errors
import itertools  # required for zip_longest function
import numbers  # required to check number type
from math import factorial


class Matrix(object):
    """ Basic matrix class
        Has numeric decimal type of values
    """
    def __init__(self, list_of_lists, rows=None, columns=None,):
        """ Matrix initialization method
            "list_of_lists" parameter should look like this: [[1, 2, 3], [4, 5, 6]] or in another form:
            [[1, 2, 3],
            [4, 5, 6]]  (in this particular case matrix with 2 rows and 3 columns will be initialize)
            number of lists - it's "m" matrix's parameter (which represent rows)
            number of elements in each list - it's "n" matrix's parameter and represent columns
        """
        #  if we didn't initialize number of rows and columns explicitly - calculate them using len() function
        if rows is None:
            try:
                self.m = len(list_of_lists)
            except TypeError:
                raise errors.WrongInputType("Should be list of lists type: [[],[]]")
        else:
            self.m = rows
        if columns is None:
            for arg in list_of_lists:  # this is redundant check
                try:
                    self.n = len(arg)
                except TypeError:
                    raise errors.WrongInputType("Should be list of lists type: [[],[]]")
        else:
            self.n = columns
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
        """ Here we make our matrix iterable by implementing __iter__ method"""
        for row in self.A:
            yield row

    def _matrix_input_checker(self, input_matrix):
        """ Private method for checking inputted data to be relevant type and in same dimension
        """
        # checker for list of lists type: [[]]
        try:
            if not (all(isinstance(column, list) for column in input_matrix)):
                raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        except TypeError:
            raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        # checking if inputted matrix has valid dimensions (if rows == m)
        # print("Number of rows: {}".format(len(input_matrix)))
        try:
            if self.m != len(input_matrix):
                raise errors.WrongDimension("Wrong number of columns for initialization")
        except TypeError:
            raise errors.WrongInputType("You should pass a list of lists like this: [[arg]]")
        # Checking values in inputted matrix for numeric type
        for row in input_matrix:
            # Checking for relevant number of columns
            # print("Number of columns: {}: ".format(len(row)))
            if self.n != len(row):
                raise errors.WrongDimension("Wrong number of rows for initialization")
            for num in row:
                if not isinstance(num, numbers.Number):
                    raise errors.WrongElementType("Should be a numeric type!")

    def is_zero_matrix(self):
        """ Checking if matrix is a Null matrix (consisting of zeros)
            return True if so
        """
        for column in self.A:
            for arg in column:
                if arg != 0:
                    return False
        return True

    def is_square_matrix(self):
        """ Checking if matrix is square matrix: m=n"""
        if self.m == self.n:
            return True
        else:
            return False

    def is_diagonal_matrix(self):
        """ Checking if matrix is a diagonal matrix: has all elements equal to zero (except of main diagonal)
           We presume, that diagonal matrix has to be also a square matrix
        """
        if self.is_square_matrix():
            for i in range(0, self.m):
                for j in range(0, self.n):
                    if self.A[i][j] == 0 and i == j:
                        return False
                    if self.A[i][j] != 0 and i != j:
                        return False
            return True
        else:
            #  Here we raised an error because non square matrices doesnt have main diagonal, so we explicitly warned
            #  users about needs of square type matrix, but we rewrote it to return just False
            return False
            # raise errors.WrongInputType("Need a square matrix here")

    def is_identity_matrix(self):
        """ Checking if matrix is identity matrix: diagonal square matrix, where all elements of the main
            diagonal equals to one. Return True if so
        """
        # it's one way to implement this method, but better example will be 'is_diagonal_matrix' implementation
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
        """ Method for comparing two matrices. If they equal - it will return True, otherwise - False
            Parameter 'matrix_for_compare' is an instance of Matrix() class.
            One matrix is equal to another if all of their components(elements) are equal
        """
        if not isinstance(matrix_for_compare, Matrix):
            raise errors.WrongInputType("Input parameter should be a Matrix type")
        #  both matrices should have same dimensions
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
        """ This method will Transpose matrix self.A and return new transposed matrix """
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
        """ Here we checking if matrix self.A is transpose to matrix_for_compare
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

    def multiply_by_number(self, number, precision=2):
        """ Method for scalar multiplication matrix A by parameter 'number'(should be a numeric type)
            For multiply matrix by some number, we actually need to multiply all her elements by this number
            After we done with multiplication, a new matrix will be returned
            'precision' parameter is optional and will point to a number or positions after '.' symbol
            precision should be a positive, integer type number"""
        if not isinstance(number, numbers.Number):
            raise errors.WrongInputType("Parameter 'number' should be a numeric type!")
        # converting number to decimal format for consistence
        if not isinstance(number, decimal.Decimal):
            number_dec = decimal.Decimal(repr(number))
        else:
            number_dec = number
        precision = self._precision_check(precision)
        return_list = []
        for rows in self.A:
            temp_list = []
            for arg in rows:
                try:
                    temp_list.append((arg*number_dec).quantize(decimal.Decimal(str((10**-precision)))))
                except decimal.InvalidOperation:
                    raise errors.WrongDimension("This is to much. Try to low down precision value")
            return_list.append(temp_list)
        return Matrix(return_list)

    def _precision_check(self, pres):
        """ Private method for checking input for precision value"""
        # method get a general check in test_14 of TestMatrix() class
        if not isinstance(pres, int):
            raise errors.WrongInputType("Precision should be an integer number")
        return abs(pres)

    def matrix_subtraction(self, matrix_for_subtract):
        """ This method will subtract incoming matrix(parameter 'matrix_for_subtract' which is Matrix type)
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

    def divide_by_number(self, number, precision=2):
        """ Dividing matrix self.A by 'number' and returning new matrix
            Basically, we need to call multiply_by_number method with income parameter: 1/number """
        precision = self._precision_check(precision)
        number_for_division = decimal.Decimal(1/number).quantize(decimal.Decimal(str(10**-precision)))
        return self.multiply_by_number(number_for_division, precision=precision)

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

    def matrix_determinant(self, precision=2):
        """ This method calculate determinant of the matrix, using elementary rows manipulation:
            first - we'll transform our matrix to upper triangular form and then - multiplied her main diagonal elements
            between themselves to get a determinant
            'precision' parameter it's number of positions after '.' symbol. It use default roundup from decimal
            method getcontext().rounding. (recommending rounding=ROUND_HALF_EVEN, which is a default)"""
        if not self.is_square_matrix():  # checking if matrix instance is a square matrix
            raise errors.WrongDimension("Determinant can be computed only for square matrix")
        precision = self._precision_check(precision)
        det = decimal.Decimal(1)  # initializing variable for determinant
        if self.m == 1:
            det = self.A[0][0]  # for square matrix with one row determinant is equal to her single element
        elif self.m == 2:  # for rank 2 matrix we will use common formula
            det = self.A[0][0]*self.A[1][1] - self.A[0][1]*self.A[1][0]
        else:
            transform_matrix = []  # creating temporary matrix for transformation operations
            for rows in self.A:
                transform_matrix.append(rows)
            for i in range(0, self.m):
                for j in range(0, self.m):
                    # need to transform all transform_matrix[j][i] elements to 0(except j =< i element)
                    if j == i and transform_matrix[i][j] == 0:
                        if i == (self.m - 1):
                            return 0
                        else:
                            for k in range(i+1, self.m):
                                if transform_matrix[k][i] != 0:
                                    # swapping rows of transform_matrix,
                                    # so we position non zero element on main diagonal
                                    temp = transform_matrix.pop(i)
                                    temp2 = transform_matrix.pop(k-1)
                                    transform_matrix.insert(i, temp2)
                                    transform_matrix.insert(k, temp)
                                    det *= (-1)  # switching determinant sight
                                    break
                                if transform_matrix[k][i] == 0 and k == (self.m - 1) and transform_matrix[i][j] == 0:
                                    return 0  # we have 0 on main diagonal, so determinant equal to 0
                    if i == j and transform_matrix[i][j] != 0:
                        det *= transform_matrix[i][i]
                    if j < i:
                        continue
                    if j > i and transform_matrix[j][i] != 0:
                        # we need to subtract a multiplied by scalar transform_matrix i-row from j-row
                        # so element [j][i] will be equal to zero
                        temp_scalar = transform_matrix[j][i]/transform_matrix[i][i]
                        temp_list = []
                        for arg1, arg2 in zip(transform_matrix[i], transform_matrix[j]):
                            temp_list.append(arg2 - arg1*temp_scalar)
                        transform_matrix.pop(j)  # deleting old j-row
                        transform_matrix.insert(j, temp_list)  # inserting new, transformed j-row
        # 10^-2 will be equal to 0.01(to positions after '.') and so on
        try:
            det = det.quantize(decimal.Decimal(str(10**-precision)))
        except decimal.InvalidOperation:
            raise errors.WrongDimension("This is to much. Try to low down precision value")
        return det

    def matrix_inverse(self, precision=3):
        """ Here we will try to find an inverse matrix to self.A using Gauss-Jordan method(elementary row operation)
            The return type of 'matrix_inverse' is a Matrix()"""
        if not self.is_square_matrix():  # checking if matrix instance is a square matrix
            raise errors.WrongDimension("Matrix should be a square to compute her inverse")
        if self.matrix_determinant() == 0:
            raise errors.WrongInputType("Determinant is Zero. This matrix doesnt have inverse matrix")
        precision = self._precision_check(precision)
        # we will need an instance of identity matrix for this method
        identity_matrix = IdentityMatrix(self.m)
        # unpacking into list of lists:
        inverse_matrix = []
        matrix_for_manipulations = []
        for rows1, rows2 in zip(identity_matrix, self.A):
            temp1 = []
            temp2 = []
            for arg1, arg2 in zip(rows1, rows2):
                temp1.append(arg1)
                temp2.append(arg2)
            inverse_matrix.append(temp1)
            matrix_for_manipulations.append(temp2)
        # first, we will transform 'matrix_for_manipulations' to upper triangular form
        # also, we will duplicate all elementary row operations on 'inverse_matrix'
        for i in range(0, self.m):
            for j in range(0, self.m):
                # here we 'eliminate' zeroes on a main diagonal
                if i == j and matrix_for_manipulations[i][j] == 0:
                    # we are not checking for special occasions with whole row or column equal to zero
                    # since in that case determinant would be a zero, and we already know that it's not true
                    for k in range(i+1, self.m):
                        if matrix_for_manipulations[k][i] != 0:
                            # switching rows in matrices
                            first_temp1 = matrix_for_manipulations.pop(i)
                            first_temp2 = matrix_for_manipulations.pop(k-1)
                            matrix_for_manipulations.insert(i, first_temp2)
                            matrix_for_manipulations.insert(k, first_temp1)
                            # duplicated operations for identity matrix
                            first_temp3 = inverse_matrix.pop(i)
                            first_temp4 = inverse_matrix.pop(k-1)
                            inverse_matrix.insert(i, first_temp4)
                            inverse_matrix.insert(k, first_temp3)
                if j < i:
                    continue
                if j > i and matrix_for_manipulations[j][i] != 0:
                    # here we need to subtract multiplied by scalar 'i' element from 'j' element,
                    # so as a result we will get '0' on a position beneath main diagonal([i][i])
                    # the 'scalar' can be found by dividing [j][i] element by [i][i] element
                    temp_scalar1 = matrix_for_manipulations[j][i]/matrix_for_manipulations[i][i]
                    # now we use our 'precision' for scalars:
                    try:
                        temp_scalar1 = temp_scalar1.quantize(decimal.Decimal(str(10**-precision)))
                        temp_list1 = []
                        temp_list2 = []
                        for arg1, arg2, arg3, arg4 in zip(matrix_for_manipulations[i], matrix_for_manipulations[j],
                                                          inverse_matrix[i], inverse_matrix[j]):
                            temp_list1.append((arg2 - (arg1*temp_scalar1)).quantize(decimal.Decimal(str(10**-precision))))
                            temp_list2.append((arg4 - arg3*temp_scalar1).quantize(decimal.Decimal(str(10**-precision))))
                    except decimal.InvalidOperation:
                        raise errors.WrongDimension("This is to much. Try to low down precision value")
                    # here we replacing row [j] with new, modified row
                    matrix_for_manipulations.pop(j)
                    matrix_for_manipulations.insert(j, temp_list1)
                    # and same thing for future inverse matrix
                    inverse_matrix.pop(j)
                    inverse_matrix.insert(j, temp_list2)
        # so, yeah, now we have upper triangular matrix
        # print('*'*45)
        # for row in matrix_for_manipulations:
        #     print(row)
        # print('*'*45)
        # next step it's to start from lower right conner([self.m][self.m]) and move up, multiplying each row by
        # appropriate number(inverse one in this situation), so we would get a '1' on a diagonal
        # Also, we need to 'eliminate' all others number, which are not on main diagonal
        for i in range(self.m - 1, -1, -1):
            # making '1' on diagonal
            temp_scalar2 = decimal.Decimal(1)/matrix_for_manipulations[i][i]
            try:
                temp_scalar2 = temp_scalar2.quantize(decimal.Decimal(str(10**-precision)))
                temp_list1 = []
                temp_list2 = []
                for arg1, arg2 in zip(matrix_for_manipulations[i], inverse_matrix[i]):
                    temp_list1.append((arg1*temp_scalar2).quantize(decimal.Decimal(str(10**-precision))))
                    temp_list2.append((arg2*temp_scalar2).quantize(decimal.Decimal(str(10**-precision))))
            except decimal.InvalidOperation:
                raise errors.WrongDimension("This is to much. Try to low down precision value")
            matrix_for_manipulations.pop(i)
            matrix_for_manipulations.insert(i, temp_list1)
            inverse_matrix.pop(i)
            inverse_matrix.insert(i, temp_list2)
            for k in range(i-1, -1, -1):
                if matrix_for_manipulations[k][i] != 0:
                    temp_scalar3 = matrix_for_manipulations[k][i]/matrix_for_manipulations[i][i]
                    try:
                        temp_scalar3 = temp_scalar3.quantize(decimal.Decimal(str(10**-precision)))
                        second_temp_list1 = []
                        second_temp_list2 = []
                        for ar1, ar2, ar3, ar4 in zip(matrix_for_manipulations[i], matrix_for_manipulations[k],
                                                      inverse_matrix[i], inverse_matrix[k]):
                            second_temp_list1.append((ar2 - ar1*temp_scalar3).quantize(decimal.Decimal(str(10**-precision))))
                            second_temp_list2.append((ar4 - ar3*temp_scalar3).quantize(decimal.Decimal(str(10**-precision))))
                    except decimal.InvalidOperation:
                        raise errors.WrongDimension("This is to much. Try to low down precision value")
                    matrix_for_manipulations.pop(k)
                    matrix_for_manipulations.insert(k, second_temp_list1)
                    inverse_matrix.pop(k)
                    inverse_matrix.insert(k, second_temp_list2)
        # print('*'*90)
        # for row in matrix_for_manipulations:
        #     print(row)
        # print('*'*90)
        return Matrix(inverse_matrix)

    def _matrix_determinant(self):
        """ Here we will compute a determinant for square matrix self.A recursively
            It's a good example 'how not to do this'
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
            return det/factorial(self.m - 3)  # for 'some reason' we have deviation = (n-3)! where n - matrix dimension
        return det

    def _matrix_determinant_n(self, matrix_n):
        """ Private method, that supposed to calculate determinant for n-dimensional matrix recursively
            Its used in _matrix_determinant method, when dimension of matrix greater than 3"""
        det = 0
        dimension = len(matrix_n)
        if dimension == 3:
            det = Matrix(matrix_n)._matrix_determinant()
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

    def matrix_unpack(self):
        """ Method for unpacking class instance of Matrix() into list of lists: [[],[]]
            Returning type - list """
        # TODO: need to add option for unpacking into float list and other formats
        matrix_list = []
        for rows in self.A:
            matrix_list.append(rows)
        return matrix_list

    def matrix_show(self):
            """ Simple method for printing a matrix
                (Will be updated soon)
            """
            for row in self.A:
                print(row)


class ZeroMatrix(Matrix):
    """ class for initialization zero matrix"""
    def __init__(self, rows, columns):
        self.zero_matrix = []
        try:
            for i in range(0, rows):
                temp = []
                for j in range(0, columns):
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
    # Just some quick testing, will be deleted soon
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
    I = Matrix([[1, 2, 3],
                [2, 4, 6],
                [7, 8, 9]])
    print(I.matrix_determinant())
    Em = Matrix([[0, 6, -2, -1, 5],
                 [0, 0, 0, -9, -7],
                 [0, 15, 35, 0, 0],
                 [0, -1, -11, -2, 1],
                 [-2, -2, 3, 0, -2]])
    C = Matrix([[1, 0, 3],
                [3, 0, 0],
                [5, 0, 5]])
    print(Em.matrix_determinant())
    print(C.matrix_determinant())
    print('&'*70)
    print(Em.matrix_inverse().matrix_show())
    invEm = Em.matrix_inverse()
    multi = invEm.matrix_multiplication(Em)
    print("+"*60)
    multi.matrix_show()


