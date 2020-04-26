# Optimization of a Terzaghi vertical consolidation model

This repository contains a numerical implementation of the Terzaghi model to simulate vertical consolidation. The numerical implementation allows for multiple load steps, loading and unloading and is linked to a <img src="https://render.githubusercontent.com/render/math?math=$C_c$">, <img src="https://render.githubusercontent.com/render/math?math=$C_r$"> compressibility model. It is set-up with an explicit finite difference method, solving the Terzaghi partial differential equation given by:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial u}{\partial t}=C_v\frac{\partial^2 u}{\partial z^2}+\frac{\partial\sigma}{\partial t}">

The goal is to automaticlaly fit the model to data by use of numerical optimization. An example of this automatic parameter fitting is given with a number of algorithms.

This repository is associated with a paper presented at the 16th International Conference of IACMAG. A preprint of the paper is available at researchgate. 

[comment]: <> (Installation; if I compile the package as an installable module)
