# INSTALLATION GUIDE:

Anaconda and Miniconda are distributions of the Python and R programming languages for scientific computing that aim to simplify package management and deployment.
Both Anaconda Distribution and Miniconda include the conda package and environment manager. The main difference is that Anaconda has a graphical interface to manage and includes more Python packages (250+) than Miniconda(<70).
**We recommend the installation of [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).** We also encourage you to use it from the terminal.

### Basic usage
Conda allows you to create separate environments, each containing their own files, packages, and package dependencies. The contents of each environment do not interact with each other.

We can use a YAML file as [`keras.yml`](https://github.com/Mathmode/COURSE-NNs-PDEs/install/keras.yml) to create an environment with the desired packages. These files create the keras with useful packages such as  Keras 3.7, Tensorflow 2.17 and JAX. It includes also the installation of the IDE [Spyder](https://www.spyder-ide.org/)
   ```
   conda env create -f keras.yml 
   ```
 
You can see the list of the environments you have
  ```
  conda env list
  ```
To go inside the environment we have created, we need to activate it
   ```
   conda activate keras
   ```

To list all of the packages in the active environment:
  ```
  conda list
  ```
To remove a package such as Keras in the current environment:
  ```
  conda remove keras
  ```
To remove an environment, you need to be outside of the environment you want remove.
  ```
  conda deactivate
  ```
  ```
  conda remove --name keras --all
  ```
# TEST THE INSTALLATION:
