import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import eigh

df_1 = pd.read_csv('pc1.csv', header = None)
df_2 = pd.read_csv('pc2.csv', header = None)
# print(df_1)
# print(df_2)

x_1 = df_1[0]
y_1 = df_1[1]
z_1 = df_1[2]
x_2 = df_2[0]
y_2 = df_2[1]
z_2 = df_2[2]
x_mean = np.mean(x_1)
y_mean = np.mean(y_1)
z_mean = np.mean(z_1)
Co_eff = []

# Function for finding the variance 
def variance(x_1, x_mean):
    sum = np.sum((x_1 - x_mean)*(x_1 - x_mean).T)
    var = (1/len(x_1))*sum
    return var

# Function for finding the co-variance 
def co_variance(x_1, x_mean, y_1, y_mean):
    cosum = np.sum((x_1 - x_mean)*(y_1 - y_mean))
    cov = (1/len(x_1))*cosum
    return cov

# Function for finding the co-variance matrix
def covariance_matrix(var, cov):
    co_mat = np.matrix([[var(x_1, x_mean), cov(x_1, x_mean, y_1, y_mean), cov(x_1, x_mean, z_1, z_mean)],
                       [cov(x_1, x_mean, y_1, y_mean), var(
                           y_1, y_mean), cov(y_1, y_mean, z_1, z_mean)],
                       [cov(x_1, x_mean, z_1, z_mean), cov(y_1, y_mean, z_1, z_mean), var(z_1, z_mean)]])
    return co_mat

# Function for finding the direction and magnitude of the surface normal 
def surface_normal(co_mat):
    eig_values, eig_vectors = eigh(co_mat)
    direction = eig_vectors[0]
    magnitude = np.linalg.norm(direction)
    return eig_values, direction, magnitude

# Function for standard least square 
def least_squares(x_i, y_i, z_i):
    ones = np.ones(len(x_i))
    XY = np.matrix([x_i, y_i, ones]).T
    Z = np.matrix([z_i]).T
    XTY = XY.T @ XY
    XTZ = XY.T @ Z
    Co_eff = (np.linalg.inv(XTY) @ XTZ)
    a1 = float(Co_eff[0])
    b1 = float(Co_eff[1])
    c1 = float(Co_eff[2])
    z_pred = []
    for i, j in zip(x_i, y_i):
        z_pred.append((a1*(i)) + (b1*(j)) + c1)
    return a1, b1, c1, z_pred

# Function for total least square 
def total_least_squares(x_i, y_i, z_i):
    ones = np.ones(len(x_i))
    X_i = np.matrix([x_i]).T
    Y_i = np.matrix([y_i]).T
    Z_i = np.matrix([z_i]).T

    X_i_mean = np.mean(X_i)
    Y_i_mean = np.mean(Y_i)
    Z_i_mean = np.mean(Z_i)
    
    X = X_i - X_i_mean
    Y = Y_i - Y_i_mean
    Z = Z_i - Z_i_mean
    
    U = np.column_stack([X , Y, Z])
    A = U.T @ U
    eig_values, eig_vectors = np.linalg.eigh(A)
    eig_vec_min = eig_vectors[:,0]
    a = eig_vec_min[0, 0]
    b = eig_vec_min[1, 0]
    c = eig_vec_min[2, 0]
    d = a * X_i_mean + b * Y_i_mean + c * Z_i_mean
    z_pred = []
    for i, j in zip(x_i, y_i):
      z_pred.append(-(a * X_i_mean + b * Y_i_mean -d)/ c)
    return  a, b, c, z_pred


# Function for RANSAC 
def ransac(df_i, s, p, e, thresh, inline):
    iterations_done = 0
    total_iterations = []
    max_iterations = int(np.log(1-p)/np.log(1-(1-e)**s))
#     print(max_iterations)
    count = []
    while iterations_done < max_iterations:
      S = df_i.sample(n=3)
      a, b, c, z_dummy = least_squares(S[0], S[1], S[2])
      inliners = []
      z_pred = []
      df_x = df_i.drop(S)
      z = df_x[2]
      for i, j in zip(df_x[0], df_x[1]):
        z_pred.append((a*(i)) + (b*(j)) + c)  
      error = (z_pred - z)**2
      for j in error: 
        if j < thresh: 
          inliners.append(j)
      count.append(len(inliners))
      desired_inliners = inline
      if len(inliners) > desired_inliners:
        if (len(df_i) == 300):
          print("No. of inliners for PC1 = ", max(count))
        elif (len(df_i) == 315):
          print("No. of inliners for PC2 = ", max(count))
        return a, b, c
      else: 
        total_iterations.append(iterations_done)
        iterations_done = 0
      iterations_done += 1
    
# Surface Plotting
def plot_realdata(fig):
  
  ax1 = fig.add_subplot(4, 2, 1, projection='3d')
  ax1.plot_trisurf(x_1, y_1, z_1, linewidth=0, antialiased=False)
  ax1.set_title('PC1 DATA')
  ax2 = fig.add_subplot(4, 2, 2, projection='3d')
  ax2.plot_trisurf(x_2, y_2, z_2, linewidth=0, antialiased=False)
  ax2.set_title('PC2 DATA')
  return 

# Plotting for standard least squares
def surface_plot_LS(fig):

  ax3 = fig.add_subplot(4, 2, 3, projection='3d')
  ax3.plot_trisurf(x_1, y_1, z_p1, linewidth=0, antialiased=False)
  ax3.scatter(x_1, y_1, z_p1, c='gray', s = 1, label='Real data')
  ax3.set_title('Standard Least Squares')
  ax4 = fig.add_subplot(4, 2, 4, projection='3d')
  ax4.plot_trisurf(x_2, y_2, z_p2, linewidth=0, antialiased=False)
  ax4.scatter(x_2, y_2, z_p2, c='gray', s = 1, label='Real data')
  ax4.set_title('Standard Least Squares')
  return 

# Plotting for total least squares
def surface_plot_TLS(fig):

  ax5 = fig.add_subplot(4, 2, 5, projection='3d')
  ax5.plot_trisurf(x_1, y_1, zt1, linewidth=0, antialiased=False)
  ax5.scatter(x_1, y_1, zt1, c='gray', s = 1, label='Real data')
  ax5.set_title('Total Least Squares')
  ax6 = fig.add_subplot(4, 2, 6, projection='3d')
  ax6.plot_trisurf(x_2, y_2, zt2, linewidth=0, antialiased=False)
  ax6.scatter(x_2, y_2, zt2, c='gray', s = 1, label='Real data')
  ax6.set_title('Total Least Squares')
  return
  
# Plotting for RANSAC
def surface_plot_RANSAC(fig):
  z_pred_1 = []
  z_pred_2 = []
  for i, j in zip(df_1[0], df_1[1]):
    z_pred_1.append((a1*(i)) + (b1*(j)) + c1)
  for i, j in zip(df_2[0], df_2[1]):
    z_pred_2.append((a2*(i)) + (b2*(j)) + c2) 

  ax7 = fig.add_subplot(4, 2, 7, projection='3d')
  ax7.plot_trisurf(x_1, y_1, z_pred_1, linewidth=0, antialiased=False)
  ax7.scatter(x_1, y_1, z_pred_1, c='gray', s = 1, label='Real data')
  ax7.set_title('RANSAC')
  ax8 = fig.add_subplot(4, 2, 8, projection='3d')
  ax8.plot_trisurf(x_2, y_2, z_pred_2, linewidth=0, antialiased=False)
  ax8.scatter(x_2, y_2, z_pred_2, c='gray', s = 1, label='Real data')
  ax8.set_title('RANSAC')
  return 

def surface_plots(fig):
    plot_realdata(fig)
    surface_plot_LS(fig)
    surface_plot_TLS(fig)
    surface_plot_RANSAC(fig)
    plt.show()
    return


co_mat = covariance_matrix(variance, co_variance)
print('The Co-Variance Matrix  is :')
print(co_mat)
print('\n')
eig_values, direction, magnitude = surface_normal(co_mat)
print("Smallest Eigen Value is : ", eig_values[0])
print('The direction of the surface normal is : ', direction)
print('The magnitude of the surface normal is : ', magnitude)


fig = plt.figure(figsize=plt.figaspect(0.5))

a1, b1 ,c1,z_p1 = least_squares(x_1, y_1, z_1)
a2, b2 ,c2, z_p2 = least_squares(x_2, y_2, z_2)

at1, bt1 ,ct1, zt1 = total_least_squares(x_1, y_1, z_1)
at2, bt2 ,ct2, zt2 = total_least_squares(x_2, y_2, z_2)

a1, b1, c1 = ransac(df_1, 3, 0.99, 0.75, 1, 190)
a2, b2, c2 = ransac(df_2, 3, 0.99, 0.75, 1, 190)

surface_plots(fig)