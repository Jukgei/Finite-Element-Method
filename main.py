# coding=utf-8
import numpy as np
import taichi as ti
import argparse
import utils


def fem(soft_obj):
	if constants.use_explicit_method:
		neo_hookean_1_grad(soft_obj)
	else:
		implicit_solver_neo_hookean(soft_obj)

		## debug code
		# A = soft_obj.matrix_ATA.to_numpy()
		# from constants import dim
		# AA = np.zeros([soft_obj.particle_cnt * dim, soft_obj.particle_cnt * dim])
		# rows, cols, _, __ = A.shape
		# for i in range(rows):
		# 	for j in range(cols):
		# 		AA[i*dim: i*dim+dim, j*dim:j*dim+dim] = A[i, j]
		#
		# n = AA.shape[0]
		# leading_minors = []
		# for k in range(1, n + 1):
		# 	sub_matrix = AA[:k, :k]
		# 	minor = np.linalg.det(sub_matrix)
		# 	leading_minors.append(minor)
		# print('\n')


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='FEM in Taichi')
	parser.add_argument('--config', help="Please input a config json file.", type=str, default='default.json')
	args = parser.parse_args()
	config = utils.read_config(args.config)
	utils.sys_init(config)

	ti.init(ti.gpu, debug=False, device_memory_fraction=0.65, kernel_profiler=True)

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
	sim_count = config.get('sim_count')
	soft_objects = []
	circle_blocks = CircleBlocks(config.get('blocks'))

	for i in config.get('objects'):
		soft_obj = Object(i)
		soft_objects.append(soft_obj)

	run = True
	now_frame = 0

	# is_output_gif = config.get('is_output_gif')
	is_output_ply = config.get('is_output_ply')
	output_fps = config.get('output_fps', 60)
	frame_time = 1.0 / output_fps
	virtual_time = 0.0
	dt = config.get("delta_time")
	ply_cnt = 0
	video_manager = ti.tools.VideoManager(output_dir="./output", framerate=output_fps, automatic_build=False)

	while widget.running:
		widget.get_event()
		if widget.is_pressed('c'):
			print('Camera position [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_position)))
			print('Camera look at [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_lookat)))
			print('Camera up [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_up)))
		frame_cnt += 1
		if widget.is_pressed('r') and frame_cnt - now_frame > 20:
			print('Resume.')
			run = True
			now_frame = frame_cnt
		if widget.is_pressed('p'):
			print('Pause.')
			run = False

		# if widget.is_pressed('q'):
		# 	print('Quit.')
		# 	break

		if run:
			for soft_obj in soft_objects:
				for i in range(sim_count):
					if constants.auto_diff == 0:
						fem(soft_obj)
					else:
						with ti.ad.Tape(loss=soft_obj.U):
							compute_energy(soft_obj)
					if constants.use_explicit_method or constants.auto_diff == 1:
						ki.kinematic(soft_obj, circle_blocks)
					else:
						advect_implicit(soft_obj, circle_blocks)
				virtual_time += sim_count * dt
				# ti.profiler.print_kernel_profiler_info()
				# ti.profiler.clear_kernel_profiler_info()

		if is_output_ply and (virtual_time / frame_time) > ply_cnt and constants.dim == 3:
			# np_pos = np.reshape(ps.fluid_particles.pos.to_numpy(), (ps.particle_num, 3))
			# writer = ti.tools.PLYWriter(num_vertices=ps.particle_num)
			# writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
			# writer.add_vertex_rgba(
			# 	np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
			# writer.export_frame_ascii(ply_cnt, series_prefix)
			# if ps.exist_rigid[None] == 1:
			# 	ps.update_mesh_vextics()

			for soft_obj in soft_objects:
				soft_obj.update_obj()
				soft_obj.save_obj(f"output/obj_{ply_cnt:06}.obj")
			# 	with open(f"output/obj_{ply_cnt:06}.obj", "w") as f:
			# 		e = ps.mesh.export(file_type='obj')
			# 		f.write(e)
			ply_cnt += 1

		msgs = []
		msgs.append("frame_cnt: {}".format(frame_cnt))
		msgs.append("time: {:.4f}".format(virtual_time))

		render.render(soft_objects, circle_blocks, virtual_time, msgs)

	# if render.is_output_gif:
	# 	render.video_manager.make_video(gif=True, mp4=True)
		# ffmpeg -i %6d.png -r 60 output.mp4