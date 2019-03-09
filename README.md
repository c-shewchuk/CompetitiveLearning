# CMPE 452 Assignment 3
## Kohonen and kMeans Competitive Neural Networks
### Curtis Shewchuk
### SN: 10189026

## Instructions for Execution

To run the program, all you will need is Python 3.7 installed on your computer.
This can be installed from https://www.python.org/. Once installed, open up a terminal
(or command prompt on Windows), navigate to the directory where the code files are.
Make sure all files submitted are in this folder, they are all necessary. Once in this folder,
simply type `python main.py` and the program will run, train the network and produce
the 'outputs.txt' file. When a run completes, you should see in the terminal, "Run Complete". If an error occurs claiming that `numpy` is not installed, simply type in
the terminal (or command prompt) `pip install numpy` (I have had random issues with numpy
not being installed sometimes which is why I mention this). If not installed, you will also need to install 
pandas via the terminal (`pip install pandas`) and matplotlib (`pip install matplotlib`)The outputs of each run
will be in the 'outputs.txt'. A sample is provided named 'outputs_bestrun.txt'. 

## Design Decisions
### Linear Separability
Becauase we're in a 3D space (as originally seen in preliminary plot) it could be seen that our data is 
linearly separable in a 3D space.

### Design Choice for Multiple Winning Nodes in K-Means
Even if there are tied winning nodes, the weights will always be updated. This simplifies some of the 
checking that needs to be done. 