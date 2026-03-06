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

The code should be run in the terminal in the subfolder 'code' because relative file paths rely on it.

The command should be formatted as:
python main.py [PATH-TO-INPUT-FILE] [ARGUMENTS]

For example:
cd GEO5017-A1/code
python main.py ../data/drone.csv --lr 0.0001 --tol 0.00001

the ARGUMENTS (except input file) are non-positional and use flags:
--iter: maximum number of iterations prior to termination (INT)
--lr: learning rate (FLOAT)
--tol: tolerance (FLOAT).

The given data is stored in a CSV file: ../data/drone.csv.

LSS solutions are not part of main.py.
The linear and quadratic solution can be viewed by running LSS_linear.py and LSS_quadratic.py respectively.
No arguments are to be provided, the input is hardcoded into the files.

***OUTPUT***
The program outputs the required results of the estimation into the terminal.

The program also generates pop-up windows with plots of the drone's position — these can be interacted with and saved as file.
The pop-up windows need to be closed for the program to continue.