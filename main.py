#!/usr/bin/env python3
import numpy as np
from control import dare
from qpsolvers import solve_qp
import scipy.linalg as spla

# How many timesteps to forecast
n = 100

# Starting point
x = np.array([[0],
              [0],
              [0]])

# Define system dynamics
A = np.array([[ 0.9835000, 2.782, 0 ],
             [ -0.0006821, 0.978, 0 ],
              [ -0.0009730, 2.804, 1 ]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])

# Make system dynamics usable by the qpsolvers library
f = np.zeroes(3*n)
f[-3] = x[0]
f[-2] = x[1]
f[-1] = x[2]

F = np.zeroes(3*n,4*n-1)
for i in range(n-1):
    for j in range(3):
        for k in range(3):
            F[3*i+j,4*i+k] = A[j,k]
            F[3*i+j,4*i+3] = B[j]
F[-3,0]=1
F[-2,1]=1
F[-1,2]=1

# Define Q matrix
Q_alpha = 1
Q_q = 1
Q_theta = 1
Q = np.array([[Q_alpha,0,0],
              [0,Q_q,0],
              [0,0,Q_theta]])

# Define R scalar
R = 1

# Define constraints (G*z less than/equal to h)
C = np.array([[1,0,0,0],
              [-1,0,0,0],
              [0,1,0,0],
              [0,-1,0,0],
              [0,0,1,0],
              [0,0,-1,0],
              [0,0,0,1],
              [0,0,0,-1],
              [0,-1,0,1],
              [0,1,0,-1]])
d = np.array([[0.4712],
              [0.4188],
              [0.2007],
              [0.2007],
              [0.2443],
              [0.2443],
              [0.6108],
              [0.6108],
              [0.4014],
              [0.4014]])
h = np.array([[0.4712],[0.4188]])
for i in range(n-1):
    h = np.append(d,h,axis=0)
G = np.zeroes(10*(n-1)+2,4*n-1)
for i in range(n-1):
    for j in range(10):
        for k in range(4):
            G[10*i+j,3+4*i+k] = C[j,k]
G[-1,-4] = -1
G[-2,-4] = 1


# Define P
P_final = control.dare(A,B,Q,R)
P_vec = np.zeroes((4*n)+3)
for i in range(0, n, 4):
    P_vec[i:i+3] = np.array([Q_alpha,Q_q,Q_theta,R])

P_vec[-3] = P_final[0,0]
P_vec[-2] = P_final[1,1]
P_vec[-1] = P_final[2,2]
P = np.diag(P_vec)

# Define q (0 vector)
q = np.zeros(n)

# Solve quadratic optimisation problem: args P, q, G, h, A, b, solver
z = qpsolvers.solve_qp(P, q, G, h, F, f, solver="proxqp")
