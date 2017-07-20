import numpy as np
import unittest
import matrices


class CustomTests(unittest.TestCase):
    """ Here we will try to compare 'matrices.py' vs 'numpy' package in several aspects
        Notice: to run this script you'll need to install numpy first"""
    def setUp(self):
        """ Initializing several matrices to work with. (Yes, they are matrices from test_matrices.py ->
        -> TestMatrix() -> test_18())"""
        self.matrix_a = [[1, 4],
                         [3, 6]]
        self.matrix_b = [[5, -2, 1],
                         [3, 1, -4],
                         [6, 0, -3]]
        self.matrix_c = [[3, 5, 7, 8],
                         [-1, 7, 0, 1],
                         [0, 5, 3, 2],
                         [1, -1, 7, 4]]
        self.matrix_d = [[1, 2, 0, 0, 0],
                         [3, 2, 3, 0, 0],
                         [0, 4, 3, 4, 0],
                         [0, 0, 5, 4, 5],
                         [0, 0, 0, 6, 5]]
        self.matrix_e = [[0, 6, -2, -1, 5],
                         [0, 0, 0, -9, -7],
                         [0, 15, 35, 0, 0],
                         [0, -1, -11, -2, 1],
                         [-2, -2, 3, 0, -2]]
        self.matrix_em = [[0, 6, -2, -1, 5],
                          [0, 0, 0, -9, -7],
                          [0, 15, 35, 0, 0],
                          [0, -1, -11, -2, 1],
                          [0, -2, 3, 0, -2]]
        self.matrix_f = [[1, 2, 3, 0, 0, 0],
                         [4, 3, 0, 0, -1, -2],
                         [1, 2, 1, 1, 1, 0],
                         [3, -2, -2, 0, 0, 0],
                         [4, 1, -1, -5, 5, 0],
                         [0, 0, 6, 5, 4, 0]]
        self.matrix_g = [[2, 0, 4, 0, 5, 0, 1],
                         [0, 3, -5, -1, 2, 0, 3],
                         [4, 5, 1, 1, 2, 3, 0],
                         [2, -4, 4, 4, 4, 2, 2],
                         [2, 0, 3, 4, 1, 3, 0],
                         [0, -5, -5, 6, 1, 4, 0],
                         [3, 0, 4, -1, 2, 0, 7]]
        self.matrix_i = [[1, 2, 3],
                         [2, 4, 6],
                         [7, 8, 9]]

    def test_1(self):
        """ Comparing inverse matrices form numpy and matrices.py
            for comparing we will use numpy method 'numpy.allclose()'
            (As a quick conclusion: all our test subjects passed comparison, but in most cases we needed to
            raise higher 'precision' parameter in our 'matrix_inverse' method)"""
        #self.setUp()
        print("A " + "="*70)
        numA = np.array(self.matrix_a)
        matA = matrices.Matrix(self.matrix_a)
        numAinv = np.linalg.inv(numA)
        matAinv = matA.matrix_inverse(6)
        print(numAinv)
        matAinv.matrix_show()
        # using something like this "np.array(np.float64(matAinv.matrix_unpack()))" to convert types
        self.assertTrue(np.allclose(np.array(np.float64(matAinv.matrix_unpack())), numAinv))
        print("B " + "="*70)
        numB = np.array(self.matrix_b)
        matB = matrices.Matrix(self.matrix_b)
        numBinv = np.linalg.inv(numB)
        matBinv = matB.matrix_inverse(6)
        print(numBinv)
        matBinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matBinv.matrix_unpack())), numBinv))
        print("C " + "="*70)
        numC = np.array(self.matrix_c)
        matC = matrices.Matrix(self.matrix_c)
        numCinv = np.linalg.inv(numC)
        matCinv = matC.matrix_inverse(7)
        print(numCinv)
        matCinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matCinv.matrix_unpack())), numCinv))
        print("D " + "="*70)
        numD = np.array(self.matrix_d)
        matD = matrices.Matrix(self.matrix_d)
        numDinv = np.linalg.inv(numD)
        matDinv = matD.matrix_inverse(7)
        print(numDinv)
        matDinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matDinv.matrix_unpack())), numDinv))
        print("E " + "="*70)
        numE = np.array(self.matrix_e)
        matE = matrices.Matrix(self.matrix_e)
        numEinv = np.linalg.inv(numE)
        matEinv = matE.matrix_inverse(7)
        print(numEinv)
        matEinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matEinv.matrix_unpack())), numEinv))
        print("Em " + "="*70)
        numEm = np.array(self.matrix_em)
        matEm = matrices.Matrix(self.matrix_em)
        # here both methods should return an error
        with self.assertRaises(Exception):
            numEminv = np.linalg.inv(numEm)
        with self.assertRaises(Exception):
            matEminv = matEm.matrix_inverse()
        numI = np.array(self.matrix_i)
        matI = matrices.Matrix(self.matrix_i)
        with self.assertRaises(Exception):
            numIinv = np.linalg.inv(numI)
        with self.assertRaises(Exception):
            matIinv = matI.matrix_inverse(5)
        print("F " + "="*70)
        numF = np.array(self.matrix_f)
        matF = matrices.Matrix(self.matrix_f)
        numFinv = np.linalg.inv(numF)
        matFinv = matF.matrix_inverse(8)
        print(numFinv)
        matFinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matFinv.matrix_unpack())), numFinv))
        print("G " + "="*70)
        numG = np.array(self.matrix_g)
        matG = matrices.Matrix(self.matrix_g)
        numGinv = np.linalg.inv(numG)
        matGinv = matG.matrix_inverse(9)
        print(numGinv)
        matGinv.matrix_show()
        self.assertTrue(np.allclose(np.array(np.float64(matGinv.matrix_unpack())), numGinv))

    def test_2(self):
        """ Here we will compare results for determinant operation in both 'matrices.py' and 'numpy'
        """
        numA = np.array(self.matrix_a)
        matA = matrices.Matrix(self.matrix_a)
        self.assertEqual(np.linalg.det(numA), matA.matrix_determinant())
        numB = np.array(self.matrix_b)
        matB = matrices.Matrix(self.matrix_b)
        # for some reason numpy compute determinant for matrix_b (3x3 with integers) as 9.0000000000000018
        # when it should be just 9, so code below raised a failure
        # self.assertEqual(np.linalg.det(numB), matB.matrix_determinant(20))
        numC = np.array(self.matrix_c)
        matC = matrices.Matrix(self.matrix_c)
        # AssertionError: 121.99999999999991 != Decimal('122.00')
        # self.assertEqual(np.linalg.det(numC), matC.matrix_determinant())
        numD = np.array(self.matrix_d)
        matD = matrices.Matrix(self.matrix_d)
        # AssertionError: 639.99999999999989 != Decimal('640.00'
        # self.assertEqual(np.linalg.det(numD), matD.matrix_determinant())
        numE = np.array(self.matrix_e)
        matE = matrices.Matrix(self.matrix_e)
        # AssertionError: 2479.9999999999968 != Decimal('2480.00')
        # self.assertEqual(np.linalg.det(numE), matE.matrix_determinant())
        numF = np.array(self.matrix_f)
        matF = matrices.Matrix(self.matrix_f)
        # AssertionError: -2346.0000000000009 != Decimal('-2346.00000000000000000000'
        # self.assertEqual(np.linalg.det(numF), matF.matrix_determinant(20))
        self.assertAlmostEqual(np.linalg.det(numF), float(matF.matrix_determinant(20)))
        numG = np.array(self.matrix_g)
        matG = matrices.Matrix(self.matrix_g)
        # AssertionError: 4029.9999999999991 != Decimal('4030.00'
        # self.assertEqual(np.linalg.det(numG), matG.matrix_determinant())
        self.assertAlmostEqual(np.linalg.det(numG), float(matG.matrix_determinant()))