"""
Citrine Informatics Technical Challenge
Scientific Software Engineer
Arash Nemati Hayati - 06/01/2018
Efficient Sampling of high-dimensional spaces with complex non-linear constraints
"""
from cgitb import reset
reset
import time
import sys
from pathlib import Path
#sys.path.insert(0, '../')
from CITC.optimization import Optimization
from CITC.optimization import Test_gen


def demo():
    # start the timer
    start = time.time() 
    # input file
    file_in = "mixture.txt"
    # output file
    file_out = "output.txt"
    # number of results
    N = 1000 
    # path where the input file exists
    path_in = Path("./")
    # path where the output file exists
    path_out = Path("./")
    # generate the complete paths
    input_file = path_in / file_in
    output_file = path_out / file_out 
    # Run the optimization scheme
    opt = Optimization(input_file, output_file)
    opt.sample(N)
    # stop the timer
    end = time.time()
    print("Elapsed time = ", "%1.3f" % (end - start), "seconds")
    return 
if __name__ == "__main__":
    demo()
