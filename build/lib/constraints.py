class Constraint():
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

#         self.exprs = []
#         self.constr = []
#         for i in range(2, len(lines)):
#             # support comments in the first line
#             if lines[i][0] == "#":
#                 continue
#             else:
#                 LH = lines[i].partition(' >')[0]
#                 self.constr.append(LH)
#                 self.exprs.append(compile(LH, "<string>", "eval"))
# #            self.exprs.append(compile(lines[i], "<string>", "eval"))
#         return


        # Run through the rest of the lines and compile the constraints
        self.exprs = []
        for i in range(2, len(lines)):
            # support comments in the first line
            if lines[i][0] == "#":
                continue
            self.exprs.append(compile(lines[i], "<string>", "eval"))
        return

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim
    
    def get_expressions(self):
        return self.exprs
        
    # def get_constr(self):
    #     return self.constr

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                return False
        return True
