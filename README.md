**Citrine Informatics Technical Challenge (CITC)**<br>

_Version: 513cabeb9657a449a7f487f24e438f81c7d5722b_ <br>


# Software
=================

One of the core capabilities of Citrination is the ability to efficiently sample high dimensional spaces with complex, non-linear constraints. In this challenge, we are asking you to efficiently generate candidates that systematically explore as much of the valid space as possible. <br>
The **“API”** of the challenge is file based:

`./sampler <input_file> <output_file> <n_results>`

**This code are written in Python3 (3.7.3)<br>** <br>
<br>
In this sampling of the high dimensional space challenge, I used the surrogate modeling toolbox (SMT), which is a Python package that contains a collection of surrogate modeling methods, sampling techniques, and benchmarking functions.<br>


You need to first install SMT by running: <br>
`$conda install -c conda-forge smt` <br>
<br>
<br>
and then install other dependencies by running:<br>
`$pip install -r requirements.txt`
<br>
<br>

The sampler exist in the test_files folder. <br>
`cd test_files`<br>
`./sampler <input_file> <output_file> <n_results>`<br>

or if you do not pass any arguments, by defualt, it runs for `formulation.txt` with outputs in the `output.txt` file  and number of smapling equial to `100`.<br>

`./sampler formulation.txt output.txt 100` <br>


<!-- I have uploaded this package to the [Python Package Index (PyPI)](http://). The easiest way to install CICT on any system is via pip: -->

<!-- `$pip install CICT` -->
