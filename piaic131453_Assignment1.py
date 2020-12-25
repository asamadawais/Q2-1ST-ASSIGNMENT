#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[7]:


import numpy as np


# 2. Create a null vector of size 10 

# In[97]:


x = np.zeros(10)
x


# 3. Create a vector with values ranging from 10 to 49

# In[96]:


x = np.arange(10, 50)
x


# 4. Find the shape of previous array in question 3

# In[8]:


x.shape


# 5. Print the type of the previous array in question 3

# In[6]:


type(x)


# 6. Print the numpy version and the configuration
# 

# In[9]:


print(np.__version__)


# 7. Print the dimension of the array in question 3
# 

# In[4]:


x.ndim


# 8. Create a boolean array with all the True values

# In[95]:


x = np.array(x, dtype = 'bool')
x


# 9. Create a two dimensional array
# 
# 
# 

# In[94]:


x = np.array([[1, 2, 3], [4, 5, 6]])
x.ndim


# 10. Create a three dimensional array
# 
# 

# In[93]:


x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
x.ndim


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[92]:


y = np.arange(10)
y[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[91]:


x = np.zeros(10)
x[4] = 1
x


# 13. Create a 3x3 identity matrix

# In[90]:


x = np.identity(3)
x


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[89]:


arr = np.array([1, 2, 3, 4, 5])
arr = arr.astype('float64')
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[86]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr3 = arr1 * arr2
arr3


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[85]:


arr1 = np.array([[1., 2., 3.],
            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],
            [7., 2., 12.]])
arr3 = arr1 == arr2
arr3


# 17. Extract all odd numbers from arr with values(0-9)

# In[27]:


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x[x % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[28]:


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x[x % 2 == 1] = -1
x


# 

# $a^2$

# 20. Create a 2d array with 1 on the border and 0 inside

# In[57]:


x = np.ones((5, 5))
x[1:-1, 1:-1] = 0
x


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[64]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1, 1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[63]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0][0][:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[79]:


x = np.arange(10).reshape((2, 5))
y = x[0]
y


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[80]:


x = np.arange(10).reshape((2, 5))
y = x[1]
y


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[84]:


x = np.arange(10).reshape((2, 5))
y = x[0][2], x[1][2]
y


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[134]:


x = np.random.randn(10, 10)
print(x)
print(np.amax(x))
print(np.amin(x))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[140]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
x = np.intersect1d(a, b)
x


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[142]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
x = np.where(a == b)
x


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[116]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
x = data[names != "Will"]
x


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[117]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
x = data[np.logical_and(names != "Will", names != "Joe")]
x


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[229]:


x = np.arange(1, 16).reshape(5, 3)
x


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[270]:


x = np.arange(1, 17).reshape(2, 2, 4)
x


# 33. Swap axes of the array you created in Question 32

# In[4]:


x = np.arange(1, 17).reshape(2, 2, 4)
np.swapaxes(x, 0, 1)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[18]:


x = np.arange(10)
y = np.sqrt(x)
y[y < 0.5]= 0
y


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[27]:


x = np.random.randn(1, 5)
y = np.random.randn(1, 10)
print(x)
print(y)
print(np.asarray((np.amax(x), np.amax(y))))


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[29]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
x = np.unique(names)
x


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[32]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
print(result)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[99]:


("[20.12.20", "18:53]")
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])

print("Actual Array: \n", sampleArray)

sampleArray[:,1] = newColumn

print("After Replacing comlumn: \n", sampleArray)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[105]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
z = np.dot(x, y)
z


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[114]:


x = np.random.randn(20)
y = np.cumsum(a, dtype = float)
y


# In[ ]:




