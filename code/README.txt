***PREREQUISITES***

The program uses built-in python libraries; matplotlib; and numpy.

To install numpy, use:
pip install numpy

To install matplotlib, use:
pip install matplotlib

***INPUT DATA***

The code expects a CSV as an input. The CSV we are using is stored in data/drone.csv

If you are using a different input, it is to be formatted as:

t1, x1, y1, z1
t2, x2, y2, z2
...

where ti is an independent variable (in our case, time)
xi, yi, zi are dependent variables (in our case, x, y, z coordinates)
i is number of row

***RUNNING THE CODE***

The code should be run in the terminal of your choosing in the subfolder 'code' because relative file paths rely on it.

The command should be formatted as:
python main.py [PATH-TO-INPUT-FILE] [ARGS]

For example:
python main.py ..\data\drone.csv --iter 100000 --lr 0.0001 --tol 0.00001

the flags are:
--iter: maximum number of iterations prior to termination
--lr: learning rate
--tol: tolerance