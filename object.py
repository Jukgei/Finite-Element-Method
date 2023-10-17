# coding=utf-8

import taichi as ti
import numpy as np
from constants import dim, vec, mat, index
import trimesh as tm
from trimesh.interfaces import gmsh
import meshio as mio
import utils

Particle = ti.types.struct(
	pos=vec,
	vel=vec,
	vel_g=vec,
	vel_next=vec,
	acc=vec,
	mass=ti.f32,
	force=vec,
	ref_pos=vec,
	implicit_A=mat,
	implicit_b=vec
)

Mesh = ti.types.struct(
	p0=ti.i32,
	p1=ti.i32,
	p2=ti.i32,
	ref=mat
)

Element = ti.types.struct(
	vertex_indices=index,
	volume=ti.f32,
	ref=mat
)

# side_length = 0.2  #
# subdivisions = 10  #

@ti.data_oriented
class Object:

	def __init__(self, config):
		self.rho = config.get("rho")
		# Young's modulus and Poisson's ratio
		E, nu = config.get('E'), config.get('nu')
		self.mu, self.s_lambda = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
		self.E, self.nu = E, nu
		self.damping = config.get('damping')
		vertices, faces, element_indices, mass, num_sides = self.load_obj(config)
		self.mass = mass
		self.particle_cnt = vertices.shape[0]
		self.mesh_cnt = faces.shape[0]
		self.element_cnt = element_indices.shape[0]
		self.meshs = Mesh.field(shape=self.mesh_cnt)
		self.elements = Element.field(shape=self.element_cnt)
		self.tensors = ti.ndarray(dtype=mat, shape=(self.element_cnt, dim, dim))
		self.tensors_com = ti.ndarray(dtype=mat, shape=(self.element_cnt, dim, dim))
		self.tensors_type = ti.types.ndarray(dtype=ti.math.mat3, ndim=3)
		self.tensors_cache = ti.field(dtype=ti.f32, shape=(self.element_cnt, dim, dim, dim, dim))
		self.particles = Particle.field(shape=self.particle_cnt, needs_grad=True)

		self.ti_vertices = ti.Vector.field(dim, ti.f32, shape=vertices.shape[0])
		self.ti_vertices.from_numpy(vertices)
		self.ti_faces = ti.Vector.field(3, ti.i32, shape=faces.shape[0])
		self.ti_faces.from_numpy(faces)
		self.ti_element = ti.Vector.field(num_sides, ti.i32, shape=element_indices.shape[0])
		self.ti_element.from_numpy(element_indices)
		self.phi = ti.field(dtype=ti.f32, shape=faces.shape[0])
		self.U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
		self.indices = ti.field(dtype=ti.i32, shape=faces.shape[0] * 3)

		self.particles_init()
		self.mesh_init()
		self.elements_init()

		# solve linear system
		self.matrix_A = ti.Matrix.field(n=dim, m=dim, shape=(self.particle_cnt, self.particle_cnt), dtype=ti.f32)
		self.vec_b = ti.Vector.field(n=dim, dtype=ti.f32, shape=self.particle_cnt)
		self.vec_x = ti.Vector.field(n=dim, dtype=ti.f32, shape=self.particle_cnt)

		print('Vertex count: {}'.format(self.particle_cnt))
		print('Mesh count: {}'.format(self.mesh_cnt))
		print('Element count: {}'.format(self.element_cnt))
		print('Element mass: {}'.format(mass))

	def load_obj(self, config):
		if dim == 2:
			side_length = config.get('side_length')
			subdivisions = config.get('subdivisions')
			x = np.linspace(0, side_length, subdivisions + 1)  #
			y = np.linspace(0, side_length, subdivisions + 1)  #
			vertices = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  #

			faces = []
			for i in range(subdivisions):
				for j in range(subdivisions):
					p1 = i * (subdivisions + 1) + j
					p2 = p1 + 1
					p3 = p1 + subdivisions + 1
					p4 = p3 + 1
					faces.append([p1, p2, p4])
					faces.append([p1, p4, p3])
			faces = np.array(faces)
			element_indices = faces
			A = ((side_length / subdivisions) ** 2) / 2
			mass = self.rho * A
			num_sides = 3
			# self.center = ti.Vector([0.72, 0.32])
			self.center = ti.Vector([0.72, 0.8])

			# vertices = np.array([[0.0, 0.0],[0.0,0.2],[0.2,0.0]])
			# faces = np.array([[0, 1, 2]])
			# element_indices = faces
			# A = (0.2 **2 )/ 2
			# mass = self.rho * A

		else:
			obj = tm.load_mesh('./obj/cube2.stl')
			obj.apply_scale(1)
			msh = gmsh.to_volume(obj, './obj/cube2.msh', mesher_id=7)
			mesh4 = mio.read('./obj/cube2.msh')
			vertices = mesh4.points

			# Get the indices of the points of the tetrahedron
			tetra_indices = mesh4.cells[0].data
			element_indices = tetra_indices

			faces = []
			for tetra in tetra_indices:
				x, y, z, w = tetra
				faces.append([x, y, z])
				faces.append([x, y, w])
				faces.append([x, z, w])
				faces.append([y, z, w])
			faces = np.array(faces)
			# faces = tetra_indices # Volume
			# vertices = mesh.vertices
			mass = self.rho / tetra_indices.shape[0]
			num_sides = 4
			self.center = ti.Vector([2, 1, 2])
		return vertices, faces, element_indices, mass, num_sides

	@ti.kernel
	def particles_init(self):

		for i in range(self.particle_cnt):
			self.particles[i].pos = self.ti_vertices[i] + self.center
			self.particles[i].ref_pos = self.particles[i].pos
			self.particles[i].mass = self.mass

	@ti.kernel
	def elements_init(self):
		for i in range(self.element_cnt):
			self.elements[i].vertex_indices = self.ti_element[i]

			r = mat(0)
			p_0 = self.particles[self.ti_element[i].x].pos
			for j in ti.static(range(dim)):
				if j + 1 <= dim:
					p_i = self.particles[self.ti_element[i][j + 1]].pos
					r[:, j] = p_i - p_0
			self.elements[i].volume = self.compute_volume(r)
			self.elements[i].ref = ti.math.inverse(r)

	@ti.kernel
	def mesh_init(self):
		for i in range(self.mesh_cnt):
			self.meshs[i].p0 = self.ti_faces[i][0]
			self.meshs[i].p1 = self.ti_faces[i][1]
			self.meshs[i].p2 = self.ti_faces[i][2]

			self.indices[i * 3 + 0] = self.ti_faces[i][0]
			self.indices[i * 3 + 1] = self.ti_faces[i][1]
			self.indices[i * 3 + 2] = self.ti_faces[i][2]

	@staticmethod
	@ti.func
	def compute_volume(x: mat) -> ti.f32:
		m = ti.math.mat3(0)
		V = 0.0
		for i in ti.static(range(x.m)):
			for j in ti.static(range(x.n)):
				m[i, j] = x[i, j]

		p0 = m[:, 0]
		p1 = m[:, 1]
		p2 = m[:, 2]

		if dim == 2:
			V = ti.abs(p0.cross(p1).norm()) / 2
		else:
			V = (1 / 6) * ti.abs(p0.dot(p1.cross(p2)))
		return V