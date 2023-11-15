This git is used for the training of various regression models used to predict eg. the power production of a wind farm near Roskilde, Zealand.

## File overview
The git contains a sub-folder 'ClimateData' which contains the raw data, a script to handle and clean the data, and the resulting csv file of meteorological data.
Besides that 3 supporting scripts containing functions used in the two main scripts can be found. These are 'regression.py', 'createOptBids.py', and 'createXY.py'.
The functions from these scripts are called by 'model1.py' and 'model2.py', which contain the framework for solving the steps requested in the assignment description.

