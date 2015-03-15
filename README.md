# random_acts_of_pizza_time
A team effort for Kaggle's Random Acts of Pizza competition using various data mining and natural language processing concepts.

##Summary
A project demonstrating the concept of *Predicting altruism through free pizza* through data analysis and modeling techniques, via Kaggle's knowledge competition—<a href="https://www.kaggle.com/c/random-acts-of-pizza">*Random Acts of Pizza*</a>

A collaborative team project for UT Austin's <a href="http://www.ideal.ece.utexas.edu/courses/ee380l_ese/">*Data Mining*</a> class involving the following students:
- Clayton Lemons
- Sierra Dai
- Hua Zhong
- Vince Kim
- David Liu

##Project Folders
- *data*—contains the competition data from the Kaggle competition
- *preprocessing*—IPython Notebook documentation on the team's standard method of preprocessing the competition data

##Environment Setup
You can use whatever development environment you would like, but it is recommended that you at least use anaconda as the basis for this project. There are several benefits to this approach:
- it is simple to get started
- anaconda is well supported
- in terms of packages and package versions, the resulting environment is consistent across platforms and for all contributors


After cloning this repository, follow these steps to get started:

1. Install Anaconda
    - Linux Install
        * download Anaconda
        * run the following command in a terminal:
```
bash Anaconda-2.x.x-Linux-x86[_64].sh
```
    - Windows or OSX Install
        * refer to the [Anaconda installation documentation](http://docs.continuum.io/anaconda/install.html)

2. Create Anaconda environment

    In the root directory of this repository, run the following commands:
```
conda env create
source activate pizza_time
```
    The *create* command makes use of the environment.yml file to download and install all necessary packages, including numpy, scipy, nltk, scikit-learn, matplotlib, seaborn, etc.

    The *activate* command adds the pizza_time environment to the PATH variable of your current terminal (this can also be accomplished using the launcher GUI). All subsequent calls to python, IPython, etc. will have all python packages in the pizza_time environment available for use.
    
    To remove the pizza_time environment, run the following command:
```
source deactivate
```

    Note: You can make pizza_time your default environment by manually adding it to your PATH variable. The root directory for Anaconda environments are typically in ~/anaconda/envs/ for Linux/OSX and C:\Anaconda\envs\ for Windows. Refer [here](http://www.continuum.io/blog/conda) for more details.

3. Upgrade Anaconda environment

   If you upgrade or add new package dependencies, make sure to run the following command in the root directory of this repository:
```
conda env export -n pizza_time > environment.yml
```

   Then, push the new environment.yml file to GitHub. This will allow other contributors to upgrade or install the same dependencies using:
```
conda env update
```   

   Note: removing packages will not automatically remove them for other contributors!
