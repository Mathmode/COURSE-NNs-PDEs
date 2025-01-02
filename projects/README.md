
During the course, we will provide an overview of how to use NN to solve PDEs. In addition, we will establish groups of three to four people to work together on one of the proposed projects. Each group must open a public GitHub project that will be linked to this page.

| Project |  Members | 
| :---:   | :---: |
| [Improving weights initialization](#initialization)  | Student 1 <br/> Student 2 <br/> Student 3 <br/> Student 4|
| [FOSLS Neural Networks](#fols)    | Student 1 <br/> Student 2 <br/> Student 3 <br/> Student 4|
| [Memory-based numerical integration](#int) | Student 1 <br/> Student 2 <br/> Student 3 <br/> Student 4|
| [Differentiable FEM](#diffFEM) | Student 1 <br/> Student 2 <br/> Student 3 <br/> Student 4|
| [Multi-level error correction for PINNs](#Multi-levelPINNs)  | Student 1 <br/> Student 2 <br/> Student 3 <br/> Student 4|


# <a id="initialization"></a> Improving weights initialization
We will explore different methods to initialize weights when solving PDEs. We will start with a simple 1D problem using ReLU activation functions.
# <a id="fols"></a> First-Order System of Least Squares Neural Networks
Following [J. A. Opschoor, P. C. Petersen, C. Schwab (2024). First Order System Least Squares Neural Networks. arXiv preprint [arXiv:2409.20264.](https://arxiv.org/pdf/2409.20264)], we propose to implement a First-Order System of Least Squares scheme for solving PDEs. 
# <a id="int"></a> Memory-based numerical integration
Following [C. Uriarte, J. M. Taylor, D. Pardo, O. A. Rodríguez, P. Vega (2023). Memory-Based Monte Carlo Integration for Solving Partial Differential Equations Using Neural Networks. In International Conference on Computational Science (pp. 509-516). Cham: Springer Nature Switzerland], we propose a further theoretical investigation and experimentation for improving numerical integration errors. The final application of this proposal is to improve the instabilities arising when using the hybrid LS/GD optimizer [C. Uriarte, M. Bastidas, D. Pardo, J. M. Taylor, S. Rojas (2024). Optimizing variational physics-informed neural networks using least squares. arXiv preprint [arXiv:2407.20417](https://arxiv.org/pdf/2407.20417)].
# <a id="diffFEM"></a> Differentiable FEM
Following a finite-element scheme, we propose an optimization strategy that adapts the location and sizes of the elements in the mesh. We will start with a simple 1D case with piecewise-linear approximations.
# <a id="Multi-levelPINNs"></a> Multi-level error correction for PINNs
Following [Z. Aldirany, R. Cottereau, M. Laforest, S. Prudhomme (2024). Multi-level neural networks for accurate solutions of boundary-value problems, Computer Methods in Applied Mechanics and Engineering, Volume 419, 116666, ISSN 0045-7825, https://doi.org/10.1016/j.cma.2023.116666], we propose to replicate the results replacing the approximation of the PDE solution in the form of a sum of neural networks by a single neural network that is iteratively retrained.

