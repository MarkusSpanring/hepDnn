Deep Neural Network for High Energy Physics
-----------

Python code to optimize Deep neural network created with **pylearn2** using **Spearmint**.

To setup Pylearn2 and Spearmint please follow the instructions given on the respective page.

-   https://github.com/HIPS/Spearmint
-   http://deeplearning.net/software/pylearn2/

Additionally the ''*Data Analysis Framework: ROOT*'' including python bindings needs to be installed.

To install just clone the repository to your home directory

	git clone https://github.com/MarkusSpanring/hepDnn

The files in the folders 
- pylearn2
- Spearmint

in the master branch need to be copied into the respective folder of the 
pylearn2 and Spearmint installation directories.

The files **main.py** and **launcher.py** in the *spearmint* folder need to be merged 
with the respective file in this folder. The modifications are necessery to run
**optimize_DNN.py** and **scheduler.py**.

The file **higgs_dataset.py** in the *pylearn2/datasets* needs to be copied to the
respective installation directory of Pylearn2.

The file **mlp.py** in the *pylearn2/models* folder needs to be merged with 
the respective file in the installation directory.
