import numpy as np
import math

def f(x):
	return 0.25 * (math.cos(x[0]-x[1])*math.cos(x[0]-x[1])+math.cos(x[0]-x[3])*math.cos(x[0]-x[3])+math.cos(x[2]-x[1])*math.cos(x[2]-x[1])+math.sin(x[2]-x[3])*math.sin(x[2]-x[3]))

def da0(x):
	return 0.25 * (-2*math.cos(x[0]-x[1])*math.sin(x[0]-x[1])-2*math.cos(x[0]-x[3])*math.sin(x[0]-x[3]))

def db0(x):
	return 0.25 * (2*math.cos(x[0]-x[1])*math.sin(x[0]-x[1])+2*math.cos(x[2]-x[1])*math.sin(x[2]-x[1]))

def da1(x):
	return 0.25 * (-2*math.cos(x[2]-x[1])*math.sin(x[2]-x[1])+2*math.sin(x[2]-x[3])*math.cos(x[2]-x[3]))

def db1(x):
	return 0.25 * (2*math.cos(x[0]-x[3])*math.sin(x[0]-x[3])-2*math.sin(x[2]-x[3])*math.cos(x[2]-x[3]))

gamma = 0.01
max_iters = 100000
precision = 0.0000001
previous_step_size = 1
previous_step_size1 = 1
previous_step_size2 = 1
previous_step_size3 = 1
previous_step_size4 = 1
iters = 0

start_val = 7.0

theta_a0 = start_val
theta_a1 = start_val + 1.0
theta_b0 = start_val + 0.5
theta_b1 = start_val - 1.0

thetas = [0.0,0.0,0.0,0.0]
print("Starting thetas:")
print(thetas)
#(previous_step_size > precision) & (previous_step_size1 > precision) & (previous_step_size2 > precision) & (previous_step_size3 > precision) & (previous_step_size4 > precision) &
while (iters < max_iters):
    prev_theta_a0 = theta_a0
    prev_theta_b0 = theta_b0
    prev_theta_a1 = theta_a1
    prev_theta_b1 = theta_b1
    thetas[0] = da0([theta_a0, theta_b0, theta_a1, theta_b1])
    theta_a0 += gamma * f(thetas)
    thetas[1] = db0([theta_a0, theta_b0, theta_a1, theta_b1])
    theta_b0 += gamma * f(thetas)
    thetas[2] = da1([theta_a0, theta_b0, theta_a1, theta_b1])
    theta_a1 += gamma * f(thetas)
    thetas[3] = db1([theta_a0, theta_b0, theta_a1, theta_b1])
    theta_b1 += gamma * f(thetas)

    #previous_step_size = abs(theta_a0 - prev_theta_a0)
    #previous_step_size1 = abs(theta_b0 - prev_theta_b0)
    #previous_step_size2 = abs(theta_a1 - prev_theta_a1)
    #previous_step_size3 = abs(theta_b1 - prev_theta_b1)

    if iters % (max_iters/10) == 0:
        print("{0}%".format((iters / max_iters) * 100))
    iters+=1
print("MAX: {0}".format(f(thetas)))
print("Maximums: A0, B0, A1, B1")
print(thetas)