#!/usr/bin/env python

"""
Citrine Informatics Technical Challenge
Scientific Software Engineer

"""

import time
import sys
from pathlib import Path
sys.path.insert(0, '../src')
from CITC.optimization import Optimization
from CITC.optimization import Test_gen


def sampler(input_f, output_f, num):
    # start the timer
    start = time.time() 
    # input file
    file_in = input_f
    # output file
    file_out = output_f
    # number of results
    N = num
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

    if len(sys.argv) > 3:
        inp_filename = sys.argv[1]
        out_filename = sys.argv[2]
        with open(out_filename, "w+") as f:
            f.write('')
        num_sample = int(sys.argv[3])
        sampler(inp_filename, out_filename, num_sample)
    else:
        sampler('formulation.txt', 'output.txt', 100)

