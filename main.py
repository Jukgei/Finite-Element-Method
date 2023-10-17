# coding=utf-8

import taichi as ti
import argparse
import utils


def fem(soft_obj):
	if constants.use_explicit_method:
		neo_hookean_1_grad(soft_obj)
	else:
		implicit_solver_neo_hookean(soft_obj)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FEM in Taichi')
	parser.add_argument('--config', help="Please input a config json file.", type=str, default='default.json')
	args = parser.parse_args()
	config = utils.read_config(args.config)
	utils.sys_init(config)

	ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

	import constants
	from render.render import Render
	import solver.kinematic as ki
	from solver.implicit import implicit_solver_neo_hookean, advect_implicit
	from solver.explicit_auto_diff import compute_energy
	from solver.explicit import neo_hookean_1_grad
	from object import Object
	from circle_blocks import CircleBlocks

	render = Render(config)
	widget, camera = render.widget, render.camera

	frame_cnt = 0
	soft_objects = []
	circleBlocks = CircleBlocks(config.get('blocks'))

	for i in config.get('objects'):
		soft_obj = Object(i)
		soft_objects.append(soft_obj)

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
			print('Resume')
			run = True
			now_frame = frame_cnt
		if widget.is_pressed('p'):
			print('Pause')
			run = False

		if run:
			for soft_obj in soft_objects:
				for i in range(10):
					if constants.auto_diff == 0:
						fem(soft_obj)
					else:
						with ti.ad.Tape(loss=soft_obj.U):
							compute_energy(soft_obj)
					if constants.use_explicit_method or constants.auto_diff == 1:
						ki.kinematic(soft_obj, circleBlocks)
					else:
						advect_implicit(soft_obj, circleBlocks)
				# ti.profiler.print_kernel_profiler_info()
				# ti.profiler.clear_kernel_profiler_info()

		if constants.dim == 2:
			render.render2d(soft_objects, circleBlocks)
		else:
			render.render3d(soft_objects)
