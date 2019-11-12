#!/bin/bash
python setup.py bdist_wheel
python -m pip install dist/CITC-1.0-py3-none-any.whl   --force-reinstall 


cp src/CITC/*py  /Users/msg/anaconda3/lib/python3.7/site-packages/CITC/
