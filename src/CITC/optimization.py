import sys
import os
import numpy as np
import mlrose
from scipy import stats
from CITC.constraints import Constraint
# sys.path.insert(0, 'citrine-challenge/src/CITC/')


# Class to generate the testcase and output configurations required to run the optimization 
class Test_gen():
    def __init__(self):
        print("Testcase created...")

    # n-dimensional volume function
    def f(self, x, n):
        if (n < 0): return 1
       
        return x[n] * self.f(x, n-1)

    def objective(self, x,  n):
        # if constraints(x):
        return self.f(x, n)
        # else: return 1000

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
    
    # sampling high-dimnesional space
    def sample(self, N):
        # Set the output file
        self.test_gen.init_output(self.output)
        # Get the dimensions of the problem
        dim = self.input.get_ndim()
        # get the initial points
        initial_point = self.input.get_example()
        # define the bounds of the problem
        xmin = self.test_gen.gen_bound(dim)[0]
        xmax = self.test_gen.gen_bound(dim)[1]
        # mybounds = MyBounds(xmin, xmax)
        # initialize counters, set step_size, threshold
        count = 0
        count_f = 0
        thres = 1e-5
        res_x = initial_point
        print("Searching for solutions...")
        while (count < N):
            if self.input.apply(res_x):
                fitness_cust = mlrose.CustomFitness(lambda x: self.test_gen.objective(x, dim-1))
            else: 
                fitness_cust = mlrose.CustomFitness(lambda x: sum(x)*10)
            problem = mlrose.ContinuousOpt(length = dim, fitness_fn = fitness_cust ,  min_val=1e-9 , max_val = 1, maximize = False) #   
            res_x, res_fun = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(), max_attempts = 2, max_iters = 4, init_state = initial_point, random_state = 1)
             
            print("the solutiuon is", res_x)
            print("cost function",res_fun)
            print("initial points are",initial_point)
            if self.input.apply(res_x) and res_fun > 0 and self.test_gen.check_bounds(res_x):  
                initial_point = res_x + np.random.uniform(-thres, thres, np.shape(res_x))
                count = count + 1
                print(count, " ", end='', flush=True)
                for i in res_x:
                    print("%1.6f" % i, " ", end="", flush=True)
                print("")
                self.test_gen.write_output(self.output, res_x)
            else:
                initial_point = res_x + np.random.uniform(-2*thres, 2*thres, np.shape(res_x))
                count_f = count_f + 1
        
            if res_fun < 0:
                print("Oops..!!")
                return 
        self.test_gen.close_output(self.output)
        return