
# Representing Data as Matrices

![](./images/Matrix.svg.png)

### Numpy and Matrices
The numpy package will provide you a great toolbox of optimized mathematical operations including the numpy array, the most convenient way to store vector and matrix data for computation. Here, we'll look at some basic operations in numpy.


```python
import numpy as np
```


```python
A = np.array([np.random.randint(0,10) for i in range(12)]).reshape((3,4)) #Create a 3x4 matrix with random integers 0-9
B = np.array([np.random.randint(0,10) for i in range(12)]).reshape((3,4)) #Create a 3x4 matrix with random integers 0-9

x = np.random.randn(4) #An array of 4 samples from the standard normal distribution
y = np.random.randn(5) #An array of 5 samples from the standard normal distribution

print('A: {}'.format(A))
print('\n')
print('B: {}'.format(B))

print('\n\n')
print('x: {}'.format(x))
print('\n\n')
print('y: {}'.format(y))
```


```python
print('A+B:\n', A+B, '\n\n') # matrix addition
print('A-B:\n', A-B, '\n\n') # matrix subtraction
print('Be careful! This is not standarad matrix multiplication!')
print('A*B:\n', A*B, '\n\n') # ELEMENTWISE multiplication
print('A/B:\n', A/B, '\n\n') # ELEMENTWISE division


print('A*x:\n', A*x, '\n\n') # multiply columns by x
print('A.T:\n', A.T, '\n\n') # transpose (just changes row/column ordering)
print('x.T:\n', x.T, '\n\n') # does nothing (can't transpose 1D array)
```

### 1. Generating Test Data
Generate two matrices of random data, A and B.  
Make matrix A a 3x4 matrix, and make B 4x4 matrix. Print both.

Calculate and print the following:
* $A^T$
* $B^T$
* AB
* $AB^T$
* $BA^T$


```python
A = None #Your code goes here
B = None #Your code goes here
print('A :', A)
print('B :', B)
```


```python
transpose_of_b = None#Your answer goes here
print('Transpose of B: {}'.format(transpose_of_b))
```


```python
print('AB:',  ) #Your code goes here
print('AB^T:', ) #Your code goes here
print('BA^T:', ) #Your code goes here
```

#### 2. Describe what happens when you take the transpose of a matrix.

Describe the transpose of a matrix here.

### Systems of Equations
If you recall from your earlier life as a algebra student:

$2x +10 = 18$ has a unique solution; one variable, one equation, one solution

Similarly, two variables with two equations has one solution*   
$x+y=4$  
$2x+2y=10$

However, if we allow 2 variables with only 1 equation, we can have infinite solutions.
$x+y=4$

*(An inconsistent system will have no solution and a system where the second equation is a multiple of the first will have infinite solutions)

### 3. Representing Data as Matrices

#### A. Write a Matrix to represent this system:   
$x+y=4$  
$2x+2y=10$


```python
#Your matrix goes here
A = None
```

#### B. Multiply your matrix by 3. What is the resulting system of equations? 


```python
#Multiplying Matrix Here
```

## Write the resulting system here  
$3x+3y=12$  
$6x+6y=30$

### The Identity Matrix
The identity Matrix has ones running along the diagonal, and zeros everywhere else.
You can create an identity matrix of any given size by calling the built in numpy method:
numpy.identity(n) where n is the dimension of the desired matrix.


```python
np.identity(5)
```

### 4. Multiply a matrix by the identity matrix. What do you notice? Explain why this happens.


```python
#Multiply a matrix by the identity matrix here.
```

#Write your observations and explanation here.
