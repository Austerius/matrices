# matrices
matrices repository was created to provide basic operation on matrices: 
addition, multiplication, transposition, reversion etc(check all operation below). 
Project, mostly, has educational purpose and will be usfull for students(it contains examples how to implement matrices
operation in python, and also examples of  posible pitfalls due to 'bad'  implementation). 
Need to mention, that main script('matrices.py') follow PEP8 convention. 
<p>To use 'matrices' in your own project - just copy 'matrices.py' and 'errors.py' in your project directory. 
Then, use import statement like this:</p> 
<p><i>import matrices</i></p>
<p>To provide more acurate calculation, matrices module using <b>decimal type</b> of numbers.
So, to initialize a new Matrix instance, you need to pass into constructor a list of lists,
that consist of decimal arguments or other numeric type arguments(they will be convert into decimal type inside a constructor).
Example: </p>
<p><i>import matrices<br/>
list_of_lists = [[1, 2, 3], 
                 [4, 5, 6], 
                 [7, 8, 9]]<br/>
A = matrices.Matrix(list_of_lists)</i></p>
Implemented matrices operations:
<li><b>is_zero_matrix</b> - checking, if a current matrix is a zero matrix. Returns 'True', if so:</li>
<i>A.is_zero_matrix()</i>
<li><b>is_square_matrix</b> - checking, if a current matrix is a square matrix(number of rows equal to number of columns)
Returns 'True', if so:</li>
<i>A.is_square_matrix()</i>
<li><b>is_diagonal_matrix</b> - checking if a matrix is a diagonal matrix: has all elements equal to zero (except of main diagonal). 
Returns 'True', if so:</li>
<i>A.is_diagonal_matrix()</i>
<li><b>is_identity_matrix</b> -Checking if a current matrix is an identity matrix:
diagonal, square matrix, where all elements on the main diagonal equals to one. Returns 'True', if so: </li>
<i>A.is_identity_matrix()</i>
<li><b>matrix_is_equal</b> - this method comparing matrix A to matrix B. Returns 'True', if matrices are equal: </li>
<i>A.matrix_is_equal(B)</i>
<li><b>matrix_transposition</b> - this method will transpose matrix A, and then - will return transposed matrix
(Matrix A will be not change by this operation. You can assign result to a new matrix, if needed):</li>
<i>B = A.matrix_transposition()</i>
<li><b>matrix_is_transpose</b> - checking, if a current matrix is transpose to a given one. Returns 'True', if so: </li>
<i>A.matrix_is_transpose(B)</i>
<li><b>matrix_addition</b> - method for adding one matrix to another, if it's possible.
A result of the addition will be returned as a new matrix:</li>
<i>C = A.matrix_addition(B)</i>
<li><b>multiply_by_number</b> - method for multiplying a current matrix by a scalar('k'). 
Also, it has a supplementary parameter of a precision - how much numbers will be after '.' symbol(default value is 2).
Note: try not to go for a precision greater than 20(error exception will be raised).
Method returns a new matrix:</li>
<i>B = A.multiply_by_number(k)</i><br/>
<i>B = A.multiply_by_number(k, precision)</i>
<li><b>matrix_subtraction</b> - method for subtracting one matrix from another( if it's possible). Returns a new matrix as a result: </li>
<i>C = A.matrix_subtraction(B)</i>
<li><b>divide_by_number</b> - method for dividing a current matrix by a scalar('k'). 
It has a supplementary parameter of a precision - how much numbers will be after '.' symbol(default value is 2).
Note: try not to go for a precision greater than 20(error exception will be raised).
Method returns a new matrix: </li>
<i>B = A.divide_by_number(k)</i><br/>
<i>B = A.divide_by_number(k, precision)</i>
<li><b>matrix_multiplication</b> - method for multiplying a current matrix by a given matrix(which written in parentheses).
Result will be returned in a form of a new matrix: </li>
<i>C = A.matrix_multiplication(B)</i>
<li><b>matrix_determinant</b> - calculating matrix determinant, using Gauss-Jordan method(elementary row operations).
It has a supplementary parameter of a precision with default value equal to 2. 
Method returns numeric result:  </li>
<i>det = A.matrix_determinant()</i><br/>
<i>det = A.matrix_determinant(precision)</i>
<li><b>matrix_inverse </b> - creating an inverse matrix to the current one, using Gauss-Jordan method(elementary row operations).
It also has a supplementary parameter of a precision with a default value equal to 3. 
Returns an instance of new Matrix() </li>
<i>B = A.matrix_inverse()</i><br/>
<i>B = A.matrix_inverse(precision)</i>
<li><b>matrix_show</b> - method for printing current matrix</li>
<i>A.matrix_show()</i>
<p>If you want to initialize a <b>Zero matrix</b> with a given sizes - use a child class of Matrix: ZeroMatrix().
It takes 2 parameters: number of rows('m' - integer number) and number of columns('n' integer number). Example:</p>
<i>O = matrices.ZeroMatrix(m, n)</i>
<p>For initializing an <b>Identity matrix</b> - use subclass IdentityMatrix().
Here you need to pass one parameter  - matrix dimension 
(n - integer digit, which represent numbers of rows/columns for square matrix). Example:</p>
<i>I = matrices.IdentityMatrix(n)</i>
