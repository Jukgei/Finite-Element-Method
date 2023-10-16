# coding=utf-8

import taichi as ti
from main import block_center, block_radius, width


def render2d(gui, obj):
	pos_ = obj.particles.pos.to_numpy()
	phi_ = obj.phi.to_numpy()

	base_ = 0.13
	gui.triangles(a=pos_[obj.meshs.p0.to_numpy()], b=pos_[obj.meshs.p1.to_numpy()], c=pos_[obj.meshs.p2.to_numpy()],
				  color=ti.rgb_to_hex([phi_ + base_, base_, base_]))
	gui.circles(pos_, radius=2, color=0xAAAA00)
	gui.circle(block_center, color=0x343434, radius=block_radius * width)
	gui.show()


def render3d(window, camera, obj, box_vert, box_lines_indices):
	# from main import box_vert, box_lines_indices
	canvas = window.get_canvas()
	scene = ti.ui.Scene()

	# Camera & light
	camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
	scene.set_camera(camera)
	scene.ambient_light((0.8, 0.8, 0.8))
	scene.point_light(pos=(3.5, 3.5, 3.5), color=(1, 1, 1))
	scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)
	# pos_ = obj.particles.pos.to_numpy()

	scene.particles(obj.particles.pos, color=(1.0, 1.0, 1), radius=.0001)
	scene.mesh(obj.particles.pos, obj.indices, show_wireframe=False)
	canvas.scene(scene)
	window.show()