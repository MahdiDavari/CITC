# import argparse
import sys

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()
# print args.accumulate(args.integers)


#  ./sampler <input_file> <output_file> <n_results>

inp_filename = sys.argv[1]
out_filename = sys.argv[2]
num_sample = sys.argv[3]


from constraints import Constraint
example = Constraint(inp_filename)

# print(example.get_ndim)
b = example.get_example()

# print()
with open(out_filename,'w') as f:
    f.write(str(example.apply(b)))







