import setuptools
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext



with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='CITC',  
     version='1.0',
     scripts=['CITC'] ,
     author="Mahdi Davari",
     author_email="Mahdi.Davari@icloud.com",
     description="Citrine Informatics Technical Challenge",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/MahdiDavari/CITC",
     packages=setuptools.find_packages('src/CITC'),
     package_dir={'': 'src/CITC'},
     py_modules=[splitext(basename(path))[0] for path in glob('src/CITC/*.py')],
     include_package_data=True,


     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
