import sys
import os
import numpy as np
# from numdifftools import Jacobian
from scipy.optimize import differential_evolution
from scipy import stats
from CITC.constraints import Constraint

sys.path.insert(0, 'citrine-challenge/src/CITC/')


# Class to generate the testcase and output configurations required to run the optimization 
class Test_gen():
    def __init__(self):
        print("Testcase created...")

    # n-dimensional volume function
    def f(self, x, n):
        if (n < 0): return 1
        return x[n] * self.f(x, n-1)
     
    # objective function * -1 since the default algorithm find the minimum candidates. With this
    # transformation it will find the maximum conditions of the objective function   
    def objective(self, x, n):
        return -self.f(x, n)

    # generate unit hypercube boundary conditions
    def gen_bound(self, n):
        xmin = [1e-15 for j in range(n)]
        xmax = [1.0 for j in range(n)]
        return [xmin, xmax]
    
    # generates boundary constraints required by COBYLA optimization scheme
    def gen_bound_const(self, n, cons):
        for i in range(0, n+1):
            fun_min = 'x[' + str(i) + ']'
            fun_max = '1.0 - x[' + str(i) + ']'  
            cons.extend([{'type': 'ineq', 'fun': lambda x: eval(fun_min)}])
            cons.extend([{'type': 'ineq', 'fun': lambda x: eval(fun_max)}])
            return cons
    
    # Check if the solution satisfies the boundary conditions
    def check_bounds(self, x):
        for ijk in x:
            if ijk < 0.0 or ijk > 1.0:
                return False
        return True
    
    # Generate the constraints defined in the input file
    def gen_const(self, input_file):
        # List = input_file.get_constr()
        List = input_file.get_expressions()

        cons = []
        for jk in List:
            eqq = lambda x, jk = jk: eval(jk)
            cons.extend([{'type': 'ineq', 'fun': eqq}])
        return cons
    
    # Generate the output file
    def init_output(self, filename):
        os.chmod(filename, 0o644)
        File = open(filename, "w+")
        File.write('')
        File.close()
    
    # Write to the output file
    def write_output(self, filename, x): 
        File = open(filename, "a+", encoding='utf-8')
        for i in x:
            File.write("%1.6f\t" % i)
        File.write("\n")
        File.close()
    
    # Close the output file
    def close_output(self, filename):
        File = open(filename, "a+")
        File.close()


class MyBounds(object):
    def __init__(self, xmin, xmax):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

# generate random purturbation for the global stochastic search
class RandomDisplacementBounds():
    #random displacement with bounds
    def __init__(self, xmin, xmax, step_size):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = step_size
        self.xmin = xmin
        self.xmax = xmax
    def __call__(self, x):
        xnew = stats.cauchy.rvs(loc=0.0, scale=0.002, size=np.shape(x))
        np.clip(xnew, self.xmin, self.xmax, out = xnew)
        return xnew


# Optimization method
class Optimization():
    def __init__(self, input_file, output_file):
        self.input = Constraint(input_file)
        self.output = output_file
        self.test_gen = Test_gen()
    
    # calculate the derivative of n-dimensional volume objective
    # def fun_der(self, x, n):
    #     return Jacobian(lambda x: self.test_gen.objective(x, n))(x).ravel()
    
    # sampling high-dimnesional space
    def sample(self, N):
        # Set the output file
        self.test_gen.init_output(self.output)
        # Get the dimensions of the problem
        dim = self.input.get_ndim()
        # Set the constraints
        cons = self.test_gen.gen_const(self.input)
        # Set SLSQP constraints
        cons_sl = cons
        # set COBYLA constraints
        cons_co = self.test_gen.gen_bound_const(dim - 1, cons)
        # get the initial points
        initial_point = self.input.get_example()
        # define the bounds of the problem
        xmin = self.test_gen.gen_bound(dim)[0]
        xmax = self.test_gen.gen_bound(dim)[1]
        bounds = [(low, high) for low, high in zip(xmin, xmax)]
        mybounds = MyBounds(xmin, xmax)
        # initialize counters, set step_size, threshold
        count = 0
        count_f = 0
        thres = 1e-5
        step_size = 0.5
        # Start the search for candidates
        print("Searching for solutions...")
        while (count < N):
            minimizer_kwargs = dict(method="COBYLA", constraints = cons_co)
            if count_f > 0 and count_f % 5 == 0:
                minimizer_kwargs = dict(method="SLSQP", constraints = cons_sl, bounds = bounds ) # jac = lambda x: self.fun_der(x, dim-1)
            try:
# func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, 
# tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, 
# disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1

                # res = differential_evolution(lambda x: self.test_gen.objective(x, dim-1), initial_point, minimizer_kwargs=minimizer_kwargs,\
                #   niter_success = 3, niter = 5, accept_test = mybounds, stepsize = step_size)
                res = differential_evolution(lambda x: self.test_gen.objective(x, dim-1), bounds=bounds, strategy='best1bin', maxiter=1000, popsize=15)

                #  , minimizer_kwargs=minimizer_kwargs,\
                #   niter_success = 3, niter = 5, accept_test = mybounds, stepsize = step_size)

            except IndexError:
                print("Maximum number of candidates reached")
                return
                
            if self.test_gen.check_bounds(res.x) and self.input.apply(res.x) and -res.fun > 0:
                initial_point = res.x + np.random.uniform(-thres, thres, np.shape(res.x))
                if count == 0:
                    cons.extend([{'type': 'ineq', 'fun': lambda x: -res.fun - thres - self.test_gen.f(x, dim-1)}])
                else:
                    cons[len(cons) - 1] = {'type': 'ineq', 'fun': lambda x: -res.fun - thres - self.test_gen.f(x, dim-1)}
                count = count + 1
                print(count, " ", end='', flush=True)
                for i in res.x:
                    print("%1.6f" % i, " ", end="", flush=True)
                print("")
                self.test_gen.write_output(self.output, res.x)
            else:
                count_f = count_f + 1
        
            if -res.fun < 0:
                print("Maximum number of candidates reached")
                return 
        self.test_gen.close_output(self.output)
        return