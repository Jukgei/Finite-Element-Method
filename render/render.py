# coding=utf-8

import taichi as ti
import constants


class Render:

	def __init__(self, config):

		self.width = 640
		self.height = 640
		self.box_vert = None
		self.box_lines_indices = None
		self.widget, self.camera = self.render_init(config)


	def render_init(self, config):
		if constants.dim == 2:
			gui = ti.GUI('Finite Element Method', (self.width, self.height))
			widget = gui
			camera = None
		else:
			window = ti.ui.Window('Finite Element Method', res=(self.width, self.height), pos=(150, 150))

			canvas = window.get_canvas()

			scene = ti.ui.Scene()

			camera = ti.ui.Camera()
			camera.position(-6.36, 3.49, 2.44)
			camera.lookat(-5.40, 3.19, 2.43)
			camera.up(0, 1, 0)
			scene.set_camera(camera)

			gui = window.get_gui()
			widget = window

			# if dim == 3:
			# Boundary Box
			box_min = ti.Vector([0, 0, 0])
			box_max = ti.Vector([5, 5, 5])
			self.box_vert = ti.Vector.field(3, ti.f32, shape=12)
			self.box_vert[0] = ti.Vector([box_min.x, box_min.y, box_min.z])
			self.box_vert[1] = ti.Vector([box_min.x, box_max.y, box_min.z])
			self.box_vert[2] = ti.Vector([box_max.x, box_min.y, box_min.z])
			self.box_vert[3] = ti.Vector([box_max.x, box_max.y, box_min.z])
			self.box_vert[4] = ti.Vector([box_min.x, box_min.y, box_max.z])
			self.box_vert[5] = ti.Vector([box_min.x, box_max.y, box_max.z])
			self.box_vert[6] = ti.Vector([box_max.x, box_min.y, box_max.z])
			self.box_vert[7] = ti.Vector([box_max.x, box_max.y, box_max.z])
			self.box_lines_indices = ti.field(int, shape=(2 * 12))
			for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
				self.box_lines_indices[i] = val

		return widget, camera

	def render2d(self, objects, circle_blocks):
		base_ = 0.13
		for obj in objects:
			pos_ = obj.particles.pos.to_numpy()
			phi_ = obj.phi.to_numpy()

			self.widget.triangles(a=pos_[obj.meshs.p0.to_numpy()], b=pos_[obj.meshs.p1.to_numpy()], c=pos_[obj.meshs.p2.to_numpy()],
						  color=ti.rgb_to_hex([phi_ + base_, base_, base_]))
			self.widget.circles(pos_, radius=2, color=0xAAAA00)
		for index in range(circle_blocks.blocks_count):
			block = circle_blocks.blocks[index]
			self.widget.circle(block.center, color=0x343434, radius=block.radius * self.width)
		self.widget.show()

	def render3d(self, objects):
		canvas = self.widget.get_canvas()
		scene = ti.ui.Scene()

		# Camera & light
		self.camera.track_user_inputs(self.widget, movement_speed=0.05, hold_key=ti.ui.RMB)
		scene.set_camera(self.camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(3.5, 3.5, 3.5), color=(1, 1, 1))
		scene.lines(self.box_vert, indices=self.box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)
		# pos_ = obj.particles.pos.to_numpy()

		for obj in objects:
			scene.particles(obj.particles.pos, color=(1.0, 1.0, 1), radius=.0001)
			scene.mesh(obj.particles.pos, obj.indices, show_wireframe=False)

		canvas.scene(scene)
		self.widget.show()