#!/usr/bin/env python
# coding: utf-8

# In[131]:


# Simple linear regression with very small dataset for practice
# Author : Shubham Ghoshal 7/14/2023
# Dataset link :- https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression


# In[132]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[133]:


df = pd.read_csv('Salary_dataset.csv')


# In[134]:


df.head()


# In[135]:


df.info()


# In[136]:


dff = pd.DataFrame()


# In[137]:


dff["X"] = df['YearsExperience']


# In[138]:


dff["Y"] = df['Salary']


# In[139]:


x = dff['X']
y = dff['Y']

# Create the scatter plot
plt.scatter(x, y)

# Set the labels for x and y axes
plt.xlabel('X')
plt.ylabel('Y')

# Set the title of the plot
plt.title('Scatter Plot of X vs Y')

# Display the plot
plt.show()


# In[140]:


#fwb = wx + b


# In[141]:


def plot_line(X, Y, w, b):
    # Generate x values within the range of X
    x = X

    # Calculate the y values using the line equation y = wx + b
    y = w * x + b
    #print(y.dtype)
    #print(y)
    #print(len(y))
    # Create the scatter plot of the data points
    plt.scatter(X, Y)

    # Plot the line
    plt.plot(x, y, color='red')

    # Set the labels for x and y axes
    plt.xlabel('X')
    plt.ylabel('Y')

    # Set the title of the plot
    plt.title('Scatter Plot with Line')

    # Display the plot
    plt.show()


# In[142]:


# random w and b just to help visualize and see how the sollution could look
w=11000
b=22000


# In[143]:


a= plot_line(dff['X'],dff['Y'], w, b)


# In[144]:


def compute_cost(X,Y,w,b):
    x = X

    '''
    n,m = X.shape  wont work as X is just 1 dimentional we can put
    m = 1 manually for now for data sets with m>1 this can be used
    '''
    m=1
    n = len(X)
    # Calculate the y values using the line equation y = wx + b
    y = w * x + b
    # y is a numoy array of n lenth n being number of rows of training data
    cost=0.0
    for i in range(len(X)):
        cost += ((Y[i] - y[i])**2)
        
    cost = cost/(2*m)
    
    return cost


# In[145]:


# random w and b just to help visualize and see how the sollution could look
w=9900
b=22000


# In[146]:


compute_cost(dff['X'],dff['Y'], w, b)


# In[147]:


compute_cost(dff['X'],dff['Y'], w, b)


# In[122]:


plot_line(dff['X'],dff['Y'], w, b)


# In[157]:


def compute_gradient(X,Y,initial_w,initial_b):
    
    cost = compute_cost(X,Y, initial_w,initial_b)
    m= 29
    

    
    dw = (1/m) * np.sum((cost - Y) * X)  # Gradient for w
    db = (1/m) * np.sum(cost - Y)  # Gradient for b
    
    return dw, db
    
    
    
    
    
    


# In[159]:


compute_gradient(dff['X'],dff['Y'],w,b)


# In[ ]:





# In[246]:


def gradient_descent(X,Y,initial_w,initial_b,number_of_iterations,alpha):
    
    #first we calculate cost at initial wb
    cost = compute_cost(X,Y, initial_w,initial_b)
    
    #then we loop till nummber of iterations are specified calculating gradient and updating each step
    #step 1 compute w and b gradients
    #step to update w and b
    #compute new cost and store in array
    #compute new w and b and store in seperate arrays
    #repeat until number of iters
    
    #all_costs = np.array()
    #all_Ws = np.array()
    #all_Bs =np.array()
    
    all_costs = []
    all_Ws = []
    all_Bs = []
    
    all_costs = np.append(all_costs, cost)
    all_Ws = np.append(all_Ws, initial_w)
    all_Bs = np.append(all_Bs, initial_b)
    
    w = initial_w
    b = initial_b
    
    for i in range(number_of_iterations):
        
        h = w * X + b  # Calculate the predicted values

        delta_w = (1/len(X)) * np.sum((h - Y) * X)  # Gradient for w
        delta_b = (1/len(X)) * np.sum(h - Y)  # Gradient for b

   
        
        w = w - delta_w*alpha
        b = b - delta_b*alpha
        
        cost = compute_cost(X,Y, w,b)
        
        all_costs = np.append(all_costs, cost)
        all_Ws = np.append(all_Ws, w)
        all_Bs = np.append(all_Bs, b)
        
        
    

    
    return cost , all_costs, all_Ws, all_Bs, w, b
        
        
        
        
        
        
        
        
    


# In[247]:


gradient_descent(dff['X'],dff['Y'],w,b,10,0.01)


# In[257]:


gradient_descent(dff['X'],dff['Y'],w,b,100,0.01)


# In[248]:


gradient_descent(dff['X'],dff['Y'],w,b,100000,0.01)


# In[249]:


test_001 = gradient_descent(dff['X'],dff['Y'],w,b,10000,0.01)


# In[250]:


test_001


# In[251]:


all_costs = test_001[1]


# In[252]:


all_costs


# In[253]:


number_of_iterations = 10000
iterations = range(number_of_iterations + 1)


# In[254]:


plt.plot(iterations, all_costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()


# In[255]:


final_W = test_001[4]


# In[259]:


final_W


# In[256]:


final_B = test_001[5]


# In[260]:


final_B


# In[258]:


# plotting line with final w and final B on data
plot_line(dff['X'],dff['Y'], final_W,final_B)


# In[ ]:




