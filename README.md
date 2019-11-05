**Citrine Informatics Technical Challenge (CITC)**<br>

_Version: 513cabeb9657a449a7f487f24e438f81c7d5722b_ <br>


Software
=================

One of the core capabilities of Citrination is the ability to efficiently sample high dimensional spaces with complex, non-linear constraints. In this challenge, we are asking you to efficiently generate candidates that systematically explore as much of the valid space as possible. <br>
The **“API”** of the challenge is file based:

`./sampler <input_file> <output_file> <n_results>`

You need to first install dependencies by running:<br>
`$pip install -r requirements.txt`



I have uploaded this package to the [Python Package Index (PyPI)](http://). The easiest way to install CICT on any system is via pip:

`$pip install CICT`





`$python setup.py test` to run the package's tests.<br>




I used diffrential evolution:

Finds the global minimum of a multivariate function.

Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimium, and can search large areas of candidate space, but often requires larger numbers of function evaluations than conventional gradient based techniques.





>>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
>>> result = differential_evolution(rosen, bounds, updating='deferred',
...                                 workers=2)




#Global Optimization methods implemented in SciPy.optimize package.
| Optimization method | Description|
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
|basinhopping(func, x0[, niter, T, stepsize, …]) | Find the global minimum of a function using the basin-hopping algorithm|
|brute(func, ranges[, args, Ns, full_output, …])| Minimize a function over a given range by brute force.|
|differential_evolution(func, bounds[, args, …])| Finds the global minimum of a multivariate function.|
|shgo(func, bounds[, args, constraints, n, …])| Finds the global minimum of a function using SHG optimization.|
| dual_annealing(func, bounds[, args, …])| Find the global minimum of a function using Dual Annealing.|



basinhopping, brute, and differential_evolution are the methods available for global optimization. Brute-force global optimization is not going to be particularly efficient.

Differential evolution is a stochastic method that should do better than brute-force, but may still require a large number of objective function evaluations. If you want to use it, you should play with the parameters and see what will work best for your problem. This tends to work better than other methods if you know that your objective function is not "smooth": there could be discontinuities in the function or its derivatives.

Basin-hopping, on the other hand, makes stochastic jumps but also uses local relaxation after each jump. This is useful if your objective function has many local minima, but due to the local relaxation used, the function should be smooth. If you can't easily get at the gradient of your function, you could still try basin-hopping with one of the local minimizers which doesn't require this information.
