# Installation guide

Anaconda and Miniconda are distributions of the Python and R programming languages for scientific computing that aim to simplify package management and deployment.
Both Anaconda Distribution and Miniconda include the conda package and environment manager. The main difference is that Anaconda has a graphical interface to manage and includes more Python packages (250+) than Miniconda(<70).
**We recommend the installation of [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).** We also encourage you to use it from the terminal.

### Basic usage
Conda allows you to create separate environments, each containing its own files, packages, and package dependencies. The contents of each environment do not interact with each other.

We can use a YAML file as [`course_env.yml`](https://github.com/Mathmode/COURSE-NNs-PDEs/blob/main/install/course_env.yml) to create an environment, named 'course' with the desired packages (specifically, Keras 3.7, Tensorflow 2.17, JAX, etc.). It includes also the installation of the IDE [Spyder](https://www.spyder-ide.org/)
   ```
   conda env create -f course_env.yml 
   ```
 
You can see the list of the environments you have
  ```
  conda env list
  ```
To go inside the environment we have created, we need to activate it
   ```
   conda activate course
   ```

To list all of the packages in the active environment:
  ```
  conda list
  ```
To remove a package, such as Keras, in the current environment:
  ```
  conda remove keras
  ```
To remove an environment, you need to be outside of the environment you want remove.
  ```
  conda deactivate
  ```
  ```
  conda remove --name course --all
  ```
# TEST THE INSTALLATION:

Once you have created the environment, you can test the backend installations by executing the files [`PINNs1D_tf.py`](https://github.com/Mathmode/COURSE-NNs-PDEs/blob/main/install/PINNs1D_tf.py) and [`PINNs1D_jax.py`](https://github.com/Mathmode/COURSE-NNs-PDEs/blob/main/install/PINNs1D_jax.py). Activate the environment and run the tests using the command:
   ```
   python filename.py
   ```

Note: You can run the tests on a CPU or GPU (if you have one supported). By default, it tries to run on the GPU if possible. Running on the GPU with the TensorFlow backend usually is trouble on the first run. In the test, we have enforced CPU execution.  If you are interested, you can try the GPU execution by following the comments in the test code [`PINNs1D_tf.py`](https://github.com/Mathmode/COURSE-NNs-PDEs/blob/main/install/PINNs1D_tf.py).
