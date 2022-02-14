import numpy as np

# Convergence limits
massive_number = 1e9
tiny_number = 1e-5

# Step taking
alpha = 0.5
beta = 0.8

# erf=loss function, x0 starting value (vector), initial massive stepsize, alpha value.
def descend(erf, x_0, step_0):
    x = x_0
    step = step_0
    e = massive_number
    while (e > tiny_number):
        x, step, e = take_step(erf, x, step)
    return x

def take_step(erf, x, step):
    # s is an infinitesimal bit of step
    s = 1.*step / 1000.
    # Find the gradient d(erf)/d(x) along all input dimensions.
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_s = np.copy(x)
        x_plus_s[i] += s
        x_minus_s = np.copy(x)
        x_minus_s[i] -= s
        
        grad[i] = 0.5 * (erf(x_plus_s) - erf(x_minus_s) ) / s
    
    # Shorten the step until error is greater than a step of alpha in the same direction.
    x_new = x - step * grad
    grad_mag = np.linalg.norm(grad)
    error_limit = alpha * step * grad_mag
    error = erf(x)
    trial_error = erf(x_new)
    
    while (error - trial_error < error_limit):
        step *= beta
        x_new = x - step * grad
        trial_error = erf(x_new)
        error_limit = alpha * step * grad_mag

    return x_new, step, trial_error