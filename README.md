## Finite Element Method

A 2D/3D Finite Element Method (FEM) implemented by [Taichi Lang](https://github.com/taichi-dev/taichi).

### Feature


- Single obj FEM simulation
- Hyperelastic material model: Neo-Hookean solid
- Explicit method:
  - [Taichi autodiff system](https://docs.taichi-lang.org/docs/differentiable_programming#limitations-of-taichi-autodiff-system)
  - Analytically differentiating
- Implicit method: 
  - Jacobi iterative method
  - Conjugate gradient method

### Prerequisites


Taichi supports multiple difference backends, such as `cuda`, `metal`. For more detail, please refer to [this](https://docs.taichi-lang.org/docs/hello_world#supported-systems-and-backends).
The code test successfully passed on both Linux 22.04 and Windows 10, with CUDA as the backend. The necessary Python packages are listed below:
- taichi
- numpy
- pyvista
- tetgen
- trimesh

### Usage


#### Quickstart
- Clone the repository
  ```bash
  git clone https://github.com/Jukgei/Finite-Element-Method.git
  ```
- Install the dependencies
  ```bash
  cd Finite-Element-Method
  pip install -r requirements.txt
  ```
- Run simulation
  ```bash
  python main.py  # use default config
  ```
- Use custom config
  ```bash
  python main.py --config ./config/demo_3d.json 
  # Please change the "./config/demo_3d.json" to your config path
  ```
  
### Reference
- [FEM Simulation of 3D Deformable Solids](https://viterbi-web.usc.edu/~jbarbic/femdefo/)
- [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
