# coding=utf-8
import numpy as np
import taichi as ti
import time
import solver.kinematic as ki

ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

dim = 3
if dim == 2:
	vec = ti.math.vec2
	mat = ti.math.mat2
	index = ti.math.ivec3
else: # 3d
	vec = ti.math.vec3
	mat = ti.math.mat3
	index = ti.math.ivec4

width = 640
height = 640


if dim == 2:
	block_center = [0.5, 0.5]
	center = ti.Vector([0.72, 0.8])
	g_dir = [0, -1]
	# g_dir = [0, 0]
	block_radius = 0.33
	# delta_time = 2e-2
	delta_time = 5e-4
	damping = 14.5
	rho = 500
else:
	block_center = [0.5, 0.5, 0.5]
	center = ti.Vector([2, 1, 2])
	g_dir = [0, -1, 0]
	block_radius = 0.0
	delta_time = 5e-4
	# delta_time = 1e-3
	damping = 5
	rho = 1000

E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, s_lambda = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)

auto_diff = False
explicit_method = False

if dim == 3:
	# Boundary Box
	box_min = ti.Vector([0, 0, 0])
	box_max = ti.Vector([5, 5, 5])
	box_vert = ti.Vector.field(3, ti.f32, shape=12)
	box_vert[0] = ti.Vector([box_min.x, box_min.y, box_min.z])
	box_vert[1] = ti.Vector([box_min.x, box_max.y, box_min.z])
	box_vert[2] = ti.Vector([box_max.x, box_min.y, box_min.z])
	box_vert[3] = ti.Vector([box_max.x, box_max.y, box_min.z])
	box_vert[4] = ti.Vector([box_min.x, box_min.y, box_max.z])
	box_vert[5] = ti.Vector([box_min.x, box_max.y, box_max.z])
	box_vert[6] = ti.Vector([box_max.x, box_min.y, box_max.z])
	box_vert[7] = ti.Vector([box_max.x, box_max.y, box_max.z])
	box_lines_indices = ti.field(int, shape=(2 * 12))
	for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
		box_lines_indices[i] = val


# import solver.explicit_auto_diff as
import utils
from solver.explicit_auto_diff import compute_energy
from solver.explicit import neo_hookean_1_grad
from solver.implicit import implicit_solver_neo_hookean, advect_implicit
from render import render
from object import Object

def fem(soft_obj):
	if explicit_method:
		neo_hookean_1_grad(soft_obj)
	else:
		implicit_solver_neo_hookean(soft_obj)


if __name__ == '__main__':
	if dim == 2:
		gui = ti.GUI('Finite Element Method', (width, height))
		widget = gui
		camera = None
	else:
		window = ti.ui.Window('Finite Element Method', res=(width, height), pos=(150, 150))

		canvas = window.get_canvas()

		scene = ti.ui.Scene()

		camera = ti.ui.Camera()
		camera.position(-6.36, 3.49, 2.44)
		camera.lookat(-5.40, 3.19, 2.43)
		camera.up(0, 1, 0)
		scene.set_camera(camera)

		gui = window.get_gui()
		widget = window

	frame_cnt = 0
	soft_obj = Object()
	run = True
	now_frame = 0
	while widget.running:
		widget.get_event()
		if widget.is_pressed('c'):
			print('Camera position [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_position)))
			print('Camera look at [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_lookat)))
			print('Camera up [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_up)))
		frame_cnt += 1
		if widget.is_pressed('r') and frame_cnt - now_frame > 20:
			print('Press R')
			run = True
			now_frame = frame_cnt
		if widget.is_pressed('p'):
			run = False
		# if frame_cnt == 3:
		# 	soft_obj.particles.pos[0].y -= 2.5e-1
		# 	print('drag')
		if run:

			for i in range(10):
				if not auto_diff:
					fem(soft_obj)
				else:
					with ti.ad.Tape(loss=soft_obj.U):
						compute_energy(soft_obj)
				if explicit_method:
					ki.kinematic(soft_obj)
				else:
					advect_implicit(soft_obj)
				# ti.profiler.print_kernel_profiler_info()
				# ti.profiler.clear_kernel_profiler_info()

		if dim == 2:
			render.render2d(widget, soft_obj)
		else:
			render.render3d(widget, camera, soft_obj, box_vert, box_lines_indices)
