## Finite Element Method

<p align=center >
  <img src=demo/demo3_3d.gif width="75%" height="100%"/>
</p>
<p align=center >
3D Demo: A simulation of an elasticity spot rendered using Houdini.
</p>

<p align=center >
  <img src=demo/demo1_2d.gif width="35%" height="25%"/>
  <img src=demo/demo2_2d.gif width="35%" height="25%"/> 
</p>

<p align=center >
2D Demo: A simulation of an elasticity square mesh passing through a narrow passage.
</p>


A Finite Element Method (FEM) implemented in [Taichi Lang](https://github.com/taichi-dev/taichi) that supports both 2D and 3D simulations. 
The FEM implementation includes support for both explicit and implicit integer methods. Have fun!

## Feature

- Single object FEM simulation
- Hyperelastic material model: Neo-Hookean solid
- Explicit method:
  - [Taichi autodiff system](https://docs.taichi-lang.org/docs/differentiable_programming#limitations-of-taichi-autodiff-system)
  - Analytically differentiating
- Implicit method: 
  - Jacobi iterative method
  - Conjugate gradient method
- Support `*.obj` and `*.stl` format 3D model input

## Prerequisites

Taichi supports multiple difference backends, such as `cuda`, `metal`. For more detail, please refer to [this](https://docs.taichi-lang.org/docs/hello_world#supported-systems-and-backends).
We utilize the `pyvista`, `tetgen` and `trimesh` packages to convert triangle meshes into tetrahedral meshes.
The code test successfully passed on both Linux 22.04 and Windows 10, with CUDA as the backend. The necessary Python packages are listed below:

- taichi
- numpy
- pyvista
- tetgen
- trimesh

If you intend to output your simulation results in `*.gif` or `*.mp4` format, it is necessary to have `ffmpeg` installed.

## Usage

### Quickstart
- Clone the repository
  ```bash
  git clone https://github.com/Jukgei/Finite-Element-Method.git
  ```
- Install the dependencies
  ```bash
  cd Finite-Element-Method
  pip install -r requirements.txt
  sudo apt install ffmpeg # if you intend to output your simulation results in `*.gif` of `*.mp4`
  ```
- Run simulation
  ```bash
  python main.py  # use default config
  ```
  If you are able to view the simulation, such as the 2D demo mentioned earlier, 
it indicates that your environment configuration is correct.
- Use custom config
  ```bash
  python main.py --config ./config/demo_3d.json 
  # Please change the "./config/demo_3d.json" to your config file path
  ```
### Config explanation
The project utilizes the JSON format for its configuration files. The JSON config contains the following parameters:
- `dim` (integer): Specifies the dimension of the simulation. Valid values are 2 or 3.
- `delta_time` (float): Represents the step size of the simulation.
- `sim_count` (integer): Determines the number of simulation steps per render.
- `auto_diff` (boolean): Specifies whether to use the Taichi autodiff system. The parameter is only valid when `"use_explicit_method": true`
- `use_explicit_method` (boolean): Determines whether to use the explicit method for simulation, if set to `false`, the implicit method is used. If set to true, the explicit method is used.
- `implicit_method` (integer): Configures the solver for the implicit method. This parameter is only valid when `"use_explicit_method": false`. The following options are available:
  - `0`: Jacobi iterative method
  - `1`: Conjugate gradient method
- `preconditioned` (integer): Specifies whether to use the preconditioning technique before applying the conjugate gradient method to solve the system. 
This parameter is only valid when `"use_explicit_method": false` and `implicit_method: 1`. The following options are available:
  - `0`: No use of the preconditioning technique.
  - `1`: Use of the preconditioning technique with the equation $A^TAx = A^Tb$
- `g_dir` (list): Represents the gravity direction of the simulation. For example, `[0, -1, 0]` indicates gravity in the negative y-axis direction.
- `is_output_gif` (boolean): Determines whether to output the simulation process as a `*.gif` format result.
- `is_output_obj` (boolean): Determines whether to output the simulation process as a `*.obj` format result.
This parameter is only valid in 2D simulation.
- `output_fps` (integer): Specifies the frames per second of the output.
- `objects` (list): Contains the simulation objects and their related parameters.
- `blocks` (list): Contains the simulation blocks and their related parameters.
This parameter is only valid in 2D simulation. Currently only circular obstacles are supported.

Each object within the `objects` list contains the following parameters:
- `id` (integer): Represents the object's ID.
- `rho` (integer): Specifies the density of the object.
- `center`: Represents the center of the object.
- `side_length` (float) (2D only): Represents the side length of the simulation square.
- `subdivisions` (integer) (2D only): Represents the subdivisions of each side of the square.
- `E` (integer): Specifies the Young's modulus of the object.
- `nu` (float): Specifies the Poisson ratio of the object.
- `damping` (float): Specifies the simulation damping of the object.
- `obj` (string) (3D only): Specifies the path of the simulation object. 

Each block within the `blocks` list contains the following parameters:
- `id` (integer): Represents the block's ID.
- `block_center` (list): Specifies the center of the block.
- `block_radius` (float): Represents the radius of the circular block.

## Assets
We provide three object models located in the `./obj` directory. It includes `cube.stl`, [Spot](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot) and `spot100.obj`. 
The `spot100.obj` is generated by dividing from [Spot](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot) model.
You can simulate your own objects by adding your `*.obj` or `*.stl` model and modifying the JSON configuration file.

For 2D simulations, you can construct your 2D mesh by overwriting the `construct_2d_mesh` function in `object.py` file. 
  
## Reference
- [FEM Simulation of 3D Deformable Solids](https://viterbi-web.usc.edu/~jbarbic/femdefo/)
- [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
- [Keenan's 3D Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot)