import sys
import os
import numpy as np
from CITC.constraints import Constraint
sys.path.insert(0, './src')
from smt.sampling_methods import LHS, Random


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

    
    # Check if the solution satisfies the boundary conditions
    def check_bounds(self, x):
        for ijk in x:
            if ijk < 0.0 or ijk > 1.0:
                return False
        return True
    
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



# random sampling 
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
        # define the bounds of the problem
        xmin = self.test_gen.gen_bound(dim)[0]
        xmax = self.test_gen.gen_bound(dim)[1]
        bounds = np.array([(low, high) for low, high in zip(xmin, xmax)])
        # initialize counters, set step_size, threshold
        count = 0
        count_f = 0
        print("Searching for solutions...")
        sampling = Random(xlimits=bounds)

        while (count < N):
            res = sampling(1)
            res_x = res[0]
            print("Progress {:2.1%} || # of failed sampleing {:d}".format(count / N, count_f), end="\r")
            
            if self.input.apply(res_x) and self.test_gen.check_bounds(res_x):
                count = count + 1
                print(count, " ", end='', flush=True)
                for i in res_x:
                    print("%1.6f" % i, " ", end="", flush=True)
                print("")
                self.test_gen.write_output(self.output, res_x)
            else:
                count_f = count_f + 1
    
        self.test_gen.close_output(self.output)
        return
