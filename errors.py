class MatrixError(Exception):
    """base class for errors and exceptions for matrices"""
    pass


class WrongDimension(MatrixError):
    """Exception, which occur if we pass wrong number of columns/rows to matrix during it initialization"""
    def __init__(self, message):
        self.message = message
        print(self.message)


class WrongElementType(MatrixError):
    """Exception, which occur if some element in matrix not a numeric type"""
    def __init__(self, message):
        self.message = message
        print(self.message)


class WrongInputType(MatrixError):
    """Exception, which occur if we didn't pass list of list into matrix for initialization
        Even if matrix has one element, our passing parameter should look like [[element]]
        also, if passing a parameter into Matrix method from outside and its not a Matrix type(when needed)
    """
    def __init__(self, message):
        self.message = message
        print(self.message)
