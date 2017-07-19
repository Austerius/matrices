import unittest
import matrices
import errors


class TestMatrix(unittest.TestCase):
    """Here we testing matrix class"""
    def test_1(self):
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6]], 2, 3)
        with self.assertRaises(errors.WrongDimension):
            B = matrices.Matrix([[1, 2], [3, 4, 5]])
        with self.assertRaises(errors.WrongDimension):
            C = matrices.Matrix([[1, 2, 3], [4, 5]])

    def test_2(self):
        """Checking for not numeric type"""
        with self.assertRaises(errors.WrongElementType):
            A = matrices.Matrix([["fgg", 2, 3], [4, 5, 6]], 2, 3)
            B = matrices.Matrix([[1, 8, None, 9], [7, 3, 4, 5]])

    def test_3(self):
        """Checking for errors with passing wrong dimension"""
        with self.assertRaises(errors.WrongDimension):
           A= matrices.Matrix([[1, 2, 3], [4, 5, 6]], 2, 2)

    def test_4(self):
        """Checking for errors with wrong input type"""

        with self.assertRaises(errors.WrongInputType):
            A = matrices.Matrix(1, 1, 1)
            A = matrices.Matrix({"1": 1}, 1, 1)
            A = matrices.Matrix((1, ), 1, 1)

    def test_5(self):
        """Checking is_zero_matrix"""
        A = matrices.Matrix([[0, 1], [3, 2]])
        self.assertFalse(A.is_zero_matrix())
        B = matrices.Matrix([[0], [0]])
        self.assertTrue(B.is_zero_matrix())

    def test_6(self):
        """Checking for columns and rows input in __init__"""
        with self.assertRaises(errors.WrongDimension):
            A = matrices.Matrix([[1, 2], [3, 4]], 2, "gg")

    def test_7(self):
        """Checking is_square_matrix method"""
        A = matrices.Matrix([[1, 2], [3, 4]])
        self.assertTrue(A.is_square_matrix())
        B = matrices.Matrix([[1]])
        self.assertTrue(B.is_square_matrix())
        C = matrices.Matrix([[1, 4, 5], [3, 6, 7]])
        self.assertFalse(C.is_square_matrix())

    def test_8(self):
        """ test for is_diagonal_matrix method"""
        A = matrices.Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 9]])
        self.assertTrue(A.is_diagonal_matrix())
        B = matrices.Matrix([[1, 0, 0], [0, 5, 6], [0, 0, 9]])
        self.assertFalse(B.is_diagonal_matrix())
        C = matrices.Matrix([[1, 0, 0, 0], [0, 5, 0, 0], [0, 0, 9, 0]])
        #  Legacy test for square matrix commented below
        # with self.assertRaises(errors.WrongInputType):
        #     self.assertTrue(C.is_diagonal_matrix())
        self.assertFalse(C.is_diagonal_matrix())

    def test_9(self):
        """Testing is_identity_matrix method"""
        A = matrices.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(A.is_identity_matrix())
        B = matrices.Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 9]])
        self.assertFalse(B.is_identity_matrix())
        C = matrices.Matrix([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertFalse(C.is_identity_matrix())
        D = matrices.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        # with self.assertRaises(errors.WrongInputType):
        #     D.is_identity_matrix()
        self.assertFalse(D.is_identity_matrix())

    def test_10(self):
        """Testing is_equal method"""
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(A.matrix_is_equal(B))
        self.assertTrue(B.matrix_is_equal(A))
        C = matrices.Matrix([[1, 2, 3], [4, 5, 0], [7, 8, 9]])
        self.assertFalse(A.matrix_is_equal(C))
        self.assertTrue(A.matrix_is_equal(A))
        I = matrices.IdentityMatrix(2)
        self.assertFalse(A.matrix_is_equal(I))
        V = matrices.Matrix([[1, 2, 3]])
        self.assertFalse(A.matrix_is_equal(V))

    def test_11(self):
        """Testing matrix_transposition method"""
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        AT = matrices.Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertTrue(AT.matrix_is_equal(A.matrix_transposition()))
        B = A.matrix_transposition()
        C = B.matrix_transposition()
        self.assertTrue(A.matrix_is_equal(C))

    def test_12(self):
        """Testing matrix_is_transpose method"""
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        AT = matrices.Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertTrue(A.matrix_is_transpose(AT))
        B = A.matrix_transposition()
        self.assertTrue(B.matrix_is_transpose(A))
        C = matrices.Matrix([[1, 3, 2], [2, 5, 8], [3, 6, 9]])
        self.assertFalse(C.matrix_is_transpose(A))
        D = matrices.IdentityMatrix(5)
        self.assertFalse(D.matrix_is_transpose(A))
        self.assertTrue(D.matrix_is_transpose(D))

    def test_13(self):
        """Testing matrices addition"""
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = matrices.Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
        C = matrices.Matrix([[2, 3, 4], [5, 6, 7], [8, 9, 11]])
        AB = A.matrix_addition(B)
        self.assertTrue(AB.matrix_is_equal(C))
        I = matrices.IdentityMatrix(2)
        with self.assertRaises(errors.WrongDimension):
            AI = A.matrix_addition(I)
        BA = B.matrix_addition(A)
        self.assertTrue(C.matrix_is_equal(BA))
        O = matrices.ZeroMatrix(3, 3)
        AO = A.matrix_addition(O)
        self.assertTrue(AO.matrix_is_equal(A))
        D = matrices.Matrix([[0, 2, 3], [3, 5, -7]])
        E = matrices.Matrix([[-1, -2, 4], [-4, 5, 8]])
        F = matrices.Matrix([[-1, 0, 7], [-1, 10, 1]])
        DE = D.matrix_addition(E)
        self.assertTrue(DE.matrix_is_equal(F))

    def test_14(self):
        """Testing multiplication by number"""
        I = matrices.IdentityMatrix(3)
        Im = I.multiply_by_number(3)
        IM = matrices.Matrix([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        self.assertTrue(IM.matrix_is_equal(Im))
        with self.assertRaises(errors.WrongInputType):
            I.multiply_by_number([3])
        A = matrices.Matrix([[2, 4], [-5, -7]])
        AM = matrices.Matrix([[-2, -4], [5, 7]])
        Am = A.multiply_by_number(-1)
        self.assertTrue(Am.matrix_is_equal(AM))
        # lets try to mess with precision
        with self.assertRaises(errors.WrongDimension):
            I.multiply_by_number(22, -57)
        with self.assertRaises(errors.WrongInputType):
            I.multiply_by_number(-.78, "fg")
        # is Am still equal to AM even with precision = 20?
        Am = A.multiply_by_number(-1, 20)
        self.assertTrue(Am.matrix_is_equal(AM))

    def test_15(self):
        """Testing subtraction method"""
        A = matrices.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = matrices.Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
        C = matrices.Matrix([[0, 1, 2], [3, 4, 5], [6, 7, 7]])
        AB = A.matrix_subtraction(B)
        self.assertTrue(AB.matrix_is_equal(C))
        I = matrices.IdentityMatrix(2)
        with self.assertRaises(errors.WrongDimension):
            AI = A.matrix_subtraction(I)
        O = matrices.ZeroMatrix(3, 3)
        AO = A.matrix_subtraction(O)
        self.assertTrue(AO.matrix_is_equal(A))
        D = matrices.Matrix([[0, 2, 3], [3, 5, -7]])
        E = matrices.Matrix([[-1, -2, 4], [-4, 5, 8]])
        F = matrices.Matrix([[1, 4, -1], [7, 0, -15]])
        DE = D.matrix_subtraction(E)
        self.assertTrue(DE.matrix_is_equal(F))

    def test_16(self):
        """Testing division by number method"""
        A = matrices.Matrix([[0.0, 0.3, 0.6], [-0.6, -0.3, 1.2]])
        AD = matrices.Matrix([[0, 1, 2], [-2, -1, 4]])
        ad = A.divide_by_number(0.3, 3)
        self.assertTrue(ad.matrix_is_equal(AD))
        Am = A.multiply_by_number(1/0.3)
        self.assertTrue(Am.matrix_is_equal(ad))
        # now lets higher up precision to 19 for false result
        # higher value of 'precision' doesn't mean better result, try to stick to reasonable number
        ad = A.divide_by_number(0.3, 19)
        self.assertFalse(ad.matrix_is_equal(AD))

        C = matrices.Matrix([[4, 6, 8], [56, 44, 32]])
        CD = matrices.Matrix([[2, 3, 4], [28, 22, 16]])
        Cd = C.divide_by_number(2)
        self.assertTrue(CD.matrix_is_equal(Cd))

    def test_17(self):
        """testing matrices multiplication"""
        K = matrices.Matrix([[1, 2], [1, 3]])
        L = matrices.Matrix([[3, 4], [1, 4]])
        kl = matrices.Matrix([[5, 12], [6, 16]])
        KL = K.matrix_multiplication(L)
        self.assertTrue(KL.matrix_is_equal(kl))
        M = matrices.Matrix([[1, 2, 7], [3, 4, 7], [5, 6, 7]])
        with self.assertRaises(errors.WrongDimension):
            K.matrix_multiplication(M)
        # checking the transposition multiplication: tran(AB) = tran(B)*tran(A)
        A = matrices.Matrix([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        B = matrices.Matrix([[2, 4],
                             [5, 6],
                             [3, -1]])
        AB = A.matrix_multiplication(B)
        ABt = AB.matrix_transposition()
        Bt = B.matrix_transposition()
        At = A.matrix_transposition()
        BtAt = Bt.matrix_multiplication(At)
        self.assertTrue(ABt.matrix_is_equal(BtAt))
        C = matrices.Matrix([[1, 2],
                             [3, 4],
                             [5, 6]])
        D = matrices.Matrix([[4, 7, 8],
                             [3, 2, 1]])
        CD = C.matrix_multiplication(D)
        DC = D.matrix_multiplication(C)

    def test_18(self):
        """ testing matrix_determinant method"""
        A = matrices.Matrix([[1, 4],
                             [3, 6]])
        B = matrices.Matrix([[5, -2, 1],
                             [3, 1, -4],
                             [6, 0, -3]])
        self.assertEqual(A.matrix_determinant(0), -6)
        self.assertEqual(B.matrix_determinant(), 9)
        C = matrices.Matrix([[3, 5, 7, 8],
                             [-1, 7, 0, 1],
                             [0, 5, 3, 2],
                             [1, -1, 7, 4]])
        self.assertEqual(C.matrix_determinant(5), 122)
        D = matrices.Matrix([[1, 2, 0, 0, 0],
                             [3, 2, 3, 0, 0],
                             [0, 4, 3, 4, 0],
                             [0, 0, 5, 4, 5],
                             [0, 0, 0, 6, 5]])
        self.assertEqual(D.matrix_determinant(), 640)
        E = matrices.Matrix([[0, 6, -2, -1, 5],
                             [0, 0, 0, -9, -7],
                             [0, 15, 35, 0, 0],
                             [0, -1, -11, -2, 1],
                             [-2, -2, 3, 0, -2]])
        self.assertEqual(E.matrix_determinant(), 2480)
        Em = matrices.Matrix([[0, 6, -2, -1, 5],
                              [0, 0, 0, -9, -7],
                              [0, 15, 35, 0, 0],
                              [0, -1, -11, -2, 1],
                              [0, -2, 3, 0, -2]])
        # determinant of matrix with 0 on main diagonal should be a 0
        self.assertEqual(Em.matrix_determinant(), 0)
        F = matrices.Matrix([[1, 2, 3, 0, 0, 0],
                             [4, 3, 0, 0, -1, -2],
                             [1, 2, 1, 1, 1, 0],
                             [3, -2, -2, 0, 0, 0],
                             [4, 1, -1, -5, 5, 0],
                             [0, 0, 6, 5, 4, 0]])
        self.assertEqual(F.matrix_determinant(), -2346)
        G = matrices.Matrix([[2, 0, 4, 0, 5, 0, 1],
                             [0, 3, -5, -1, 2, 0, 3],
                             [4, 5, 1, 1, 2, 3, 0],
                             [2, -4, 4, 4, 4, 2, 2],
                             [2, 0, 3, 4, 1, 3, 0],
                             [0, -5, -5, 6, 1, 4, 0],
                             [3, 0, 4, -1, 2, 0, 7]])
        GT = G.matrix_transposition()
        self.assertEqual(G.matrix_determinant(), 4030)
        # determinant of the transpose matrix should be equal to her original determinant
        self.assertEqual(G.matrix_determinant(), GT.matrix_determinant())
        I = matrices.Matrix([[1, 2, 3],
                             [2, 4, 6],
                             [7, 8, 9]])
        # determinant of matrix with proportional rows should be a zero
        self.assertEqual(I.matrix_determinant(), 0)
        # and some more sets of matrices
        a = matrices.Matrix([[1, -2, 4],
                             [-5, 2, 0],
                             [1, 0, 3]])
        self.assertEqual(a.matrix_determinant(), -32)
        # det(a * B) = det(a) * det(B)
        aB = a.matrix_multiplication(B)
        self.assertEqual(aB.matrix_determinant(), -288)
        # big one incoming (13x13):
        F = matrices.Matrix([[24, -13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-24, 46, -24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, -33, 64, -33, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, -40, 78, -40, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, -45, 88, -45, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, -48, 94, -48, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, -49, 96, -49, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, -48, 94, -48, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, -45, 88, -45, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, -40, 78, -40, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, -33, 64, -33, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -24, 46, -24],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, -13]])
        self.assertEqual(F.matrix_determinant(), 93383110850641920000)
        o = matrices.ZeroMatrix(4, 4)
        # determinant for zero matrix = 0
        self.assertEqual(o.matrix_determinant(), 0)
        i = matrices.IdentityMatrix(4)
        # determinant of identity matrix equal to 1
        self.assertEqual(i.matrix_determinant(), 1)


class TestZeroMatrix(unittest.TestCase):
    """class for testing ZeroMatrix subclass of Matrix"""
    def test_1(self):
        A = matrices.ZeroMatrix(3, 4)
        self.assertTrue(A.is_zero_matrix())
        with self.assertRaises(errors.WrongInputType):
            B = matrices.ZeroMatrix(6, "f")
        C = matrices.ZeroMatrix(2, 2)
        self.assertTrue(C.is_square_matrix())
        self.assertFalse(C.is_diagonal_matrix())
        Z = matrices.Matrix([[0, 0], [0, 0]])
        self.assertTrue(C.matrix_is_equal(Z))


class TestIdentityMatrix(unittest.TestCase):
    """class for testing IdentityMatrix subclass of Matrix"""
    def test_1(self):
        A = matrices.IdentityMatrix(3)
        B = matrices.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(A.is_identity_matrix())
        self.assertFalse(A.is_zero_matrix())
        self.assertTrue(A.is_diagonal_matrix())
        self.assertTrue(A.is_square_matrix())
        self.assertTrue(A.matrix_is_equal(B))
        C = A.matrix_transposition()
        self.assertTrue(C.matrix_is_equal(A))
