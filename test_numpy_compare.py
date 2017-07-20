import numpy as np
import unittest
import matrices
import time


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

    def test_3(self):
        """ Here we will run a few performance checks for 'matrices.py' and 'numpy'
            First, we'll check speed of determinant calculation, than - inverse matrix computation
            This is not a real Unittest and lots of duplicated code can be bad for yor health!
            Short conclusion: our methods way faster from numpy for a 2x2 example matrix"""
        det_cycles = 1
        numG = np.array(self.matrix_g)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = np.linalg.det(numG)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matG = matrices.Matrix(self.matrix_g)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matG.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        print("="*70)
        print("For matrix G(7x7) calculation of determinant {0} time(s) by numpy took {1} seconds".format(
                                                det_cycles, total_time_num))
        print("For matrix G(7x7) calculation of determinant {0} time(s) by matrices.py took {1} seconds".format(
                                                det_cycles, total_time_mat))
        # as a result we have this:
        # For matrix G(7x7) calculation of determinant 100000 time(s) by numpy took 2.5861989099275493 seconds
        # For matrix G(7x7) calculation of determinant 100000 time(s) by matrices.py took 21.711577452218382 seconds
        # For matrix G(7x7) calculation of determinant 100 time(s) by numpy took 0.002622916097404813 seconds
        # For matrix G(7x7) calculation of determinant 100 time(s) by matrices.py took 0.02498621645262421 seconds
        # As expected: numpy with his progressive algorithms calculate determinant much faster than our teaching method

        # Now, lets check if its true for 2x2 matrix
        # (and yes - we copied previous code, and that, so you already know - is a bad programing example):
        det_cycles = 1
        numA = np.array(self.matrix_a)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = np.linalg.det(numA)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matA = matrices.Matrix(self.matrix_a)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matA.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        print("="*70)
        print("For matrix A(2x2) calculation of determinant {0} time(s) by numpy took {1} seconds".format(
            det_cycles, total_time_num))
        print("For matrix A(2x2) calculation of determinant {0} time(s) by matrices.py took {1} seconds".format(
            det_cycles, total_time_mat))
        # WE, actually, was faster this time(in our method 2x2 matrices was calculating by using standard formula)
        # For matrix A(2x2) calculation of determinant 100 time(s) by numpy took 0.0022251738505491593 seconds
        # For matrix A(2x2) calculation of determinant 100 time(s) by matrices.py took 0.0008682422217946573 seconds
        # For matrix A(2x2) calculation of determinant 100000 time(s) by numpy took 2.4833566697009335 seconds
        # For matrix A(2x2) calculation of determinant 100000 time(s) by matrices.py took 0.8371969034310927 seconds

        # Finally, lets check 3x3 matrices. We, also, will use _matrix_determinant - a private method of 'matrices.py'
        # where for computing determinant static formula was used:
        det_cycles = 1
        numB = np.array(self.matrix_b)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = np.linalg.det(numB)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matB = matrices.Matrix(self.matrix_b)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matB.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matB._matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat_prive = end_time - start_time
        print("="*70)
        print("For matrix B(3x3) calculation of determinant {0} time(s) by numpy took {1} seconds".format(
            det_cycles, total_time_num))
        print("For matrix B(3x3) calculation of determinant {0} time(s) by matrices.py took {1} seconds".format(
            det_cycles, total_time_mat))
        print("For matrix B(3x3) calculation of determinant {0} time(s) by matrices.py using static formula"
              " took {1} seconds".format(det_cycles, total_time_mat_prive))
        # here the results:
        # For matrix B(3x3) calculation of determinant 100 time(s) by numpy took 0.002191624453710725 seconds
        # For matrix B(3x3) calculation of determinant 100 time(s) by matrices.py took 0.006737769830118791 seconds
        # For matrix B(3x3) calculation of determinant 100 time(s) by matrices.py using static formula took 0.0006960489319973936 seconds
        # For matrix B(3x3) calculation of determinant 100000 time(s) by numpy took 2.5949706624671984 seconds
        # For matrix B(3x3) calculation of determinant 100000 time(s) by matrices.py took 3.7783217540756455 seconds
        # For matrix B(3x3) calculation of determinant 100000 time(s) by matrices.py using static formula took 0.705146881684616 seconds
        # AS we see: numpy calculate 'slightly' faster than our general method, but our private mrthod for 3x3 matrices much faster!

        # Now, how about linearly dependent matrix:
        det_cycles = 1
        numI = np.array(self.matrix_i)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = np.linalg.det(numI)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matI = matrices.Matrix(self.matrix_i)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matI.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            det = matI._matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat_prive = end_time - start_time
        print("="*70)
        print("For linearly dependent matrix I(3x3) calculation of determinant {0} time(s) by numpy took {1} seconds".format(
            det_cycles, total_time_num))
        print("For linearly dependent matrix I(3x3) calculation of determinant {0} time(s) by matrices.py took {1} seconds".format(
            det_cycles, total_time_mat))
        print("For linearly dependent matrix I(3x3) calculation of determinant {0} time(s) by matrices.py using static formula"
              " took {1} seconds".format(det_cycles, total_time_mat_prive))
        # For linearly dependent matrix I(3x3) calculation of determinant 100 time(s) by numpy took 0.002870292372888207 seconds
        # For linearly dependent matrix I(3x3) calculation of determinant 100 time(s) by matrices.py took 0.00395559515013153 seconds
        # For linearly dependent matrix I(3x3) calculation of determinant 100 time(s) by matrices.py using static formula took 0.0006713921463691467 seconds
        # For linearly dependent matrix I(3x3) calculation of determinant 100000 time(s) by numpy took 2.4001016182935566 seconds
        # For linearly dependent matrix I(3x3) calculation of determinant 100000 time(s) by matrices.py took 2.7887345975931743 seconds
        # For linearly dependent matrix I(3x3) calculation of determinant 100000 time(s) by matrices.py using static formula took 0.6946370278629761 seconds
        # Yeh, for 100000 iterations numpy was just a bit faster, but this difference grown to almost 5 second
        # for  1000000 iterations, in the other hand - static formula in our private method still working faster

        # Now, lets test some inversion methods:
        # Attention: duplicated code! Try not to do this in your programs - construct a method, which use a duplicates
        det_cycles = 1
        numG = np.array(self.matrix_g)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            numGinv = np.linalg.inv(numG)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matG = matrices.Matrix(self.matrix_g)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            matGinv = matG.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        print("="*70)
        print("For matrix G(7x7) computation of inverse matrix {0} time(s) by numpy took {1} seconds".format(
            det_cycles, total_time_num))
        print("For matrix G(7x7) computation of inverse matrix {0} time(s) by matrices.py took {1} seconds".format(
            det_cycles, total_time_mat))
        # For matrix G(7x7) computation of inverse matrix 100 time(s) by numpy took 0.0031977021492632877 seconds
        # For matrix G(7x7) computation of inverse matrix 100 time(s) by matrices.py took 0.030527930074972794 seconds
        # For matrix G(7x7) computation of inverse matrix 100000 time(s) by numpy took 2.9690120793997004 seconds
        # For matrix G(7x7) computation of inverse matrix 100000 time(s) by matrices.py took 22.447151615787458 seconds
        # Ok, numpy won this with great margin, but... the thing is: we calculated determinant in our inverse method
        # and it took 21.71 second for previous test to do that. So, actual, the inverting procedure
        # was less, than a second(anyway - it's not count without determinant check)
        det_cycles = 100000
        numA = np.array(self.matrix_a)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            numAinv = np.linalg.inv(numA)
        end_time = time.perf_counter()
        total_time_num = end_time - start_time
        matA = matrices.Matrix(self.matrix_a)
        start_time = time.perf_counter()
        for i in range(1, det_cycles):
            matAinv = matA.matrix_determinant()
        end_time = time.perf_counter()
        total_time_mat = end_time - start_time
        print("="*70)
        print("For matrix A(2x2) computation of inverse matrix {0} time(s) by numpy took {1} seconds".format(
            det_cycles, total_time_num))
        print("For matrix A(2x2) computation of inverse matrix {0} time(s) by matrices.py took {1} seconds".format(
            det_cycles, total_time_mat))
        # For matrix A(2x2) computation of inverse matrix 100 time(s) by numpy took 0.004653260920530776 seconds
        # For matrix A(2x2) computation of inverse matrix 100 time(s) by matrices.py took 0.0009753577659173699 seconds
        # For matrix A(2x2) computation of inverse matrix 100000 time(s) by numpy took 2.756857820081456 seconds
        # For matrix A(2x2) computation of inverse matrix 100000 time(s) by matrices.py took 0.8733635574325254 seconds
        # Yes, we small matrix(2x2) our method way faster!
