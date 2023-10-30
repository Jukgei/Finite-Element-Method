# coding=utf-8
import os
import time

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
		self.mesh4 = None
		vertices, faces, element_indices, mass, num_sides = self.load_obj(config)
		self.mass = mass
		self.particle_cnt = vertices.shape[0]
		self.mesh_cnt = faces.shape[0]
		self.element_cnt = element_indices.shape[0]
		self.meshs = Mesh.field(shape=self.mesh_cnt)
		self.elements = Element.field(shape=self.element_cnt)
		self.tensors = ti.ndarray(dtype=mat, shape=(self.element_cnt, dim, dim))
		# self.tensors_com = ti.ndarray(dtype=mat, shape=(self.element_cnt, dim, dim))
		# self.tensors_type = ti.types.ndarray(dtype=ti.math.mat3, ndim=3)
		# self.tensors_cache = ti.field(dtype=ti.f32, shape=(self.element_cnt, dim, dim, dim, dim))
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
		# self.matrix_A = ti.Matrix.field(n=dim, m=dim, shape=(self.particle_cnt, self.particle_cnt), dtype=ti.f32)
		# self.vec_b = ti.Vector.field(n=dim, dtype=ti.f32, shape=self.particle_cnt)
		# self.vec_x = ti.Vector.field(n=dim, dtype=ti.f32, shape=self.particle_cnt)

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
			self.center = ti.Vector(config.get('center'))

			# vertices = np.array([[0.0, 0.0],[0.0,0.2],[0.2,0.0]])
			# faces = np.array([[0, 1, 2]])
			# element_indices = faces
			# A = (0.2 **2 )/ 2
			# mass = self.rho * A

		else:
			obj_path = config.get('obj')
			file_type = obj_path.split('.')[-1]
			mesh4 = None
			if file_type == 'stl' or file_type == 'obj':
				msh_file_path = '.'.join((obj_path.split('.')[:-1])) + '.msh'
				if os.path.exists(msh_file_path):
					mesh4 = mio.read(msh_file_path)
					print('The tetrahedron already exists.')
				else:
					obj = tm.load_mesh(obj_path)
					obj.apply_scale(1.0)
					# method 7 :MMG3D is nice
					msh = gmsh.to_volume(obj, msh_file_path, 20, mesher_id=7)
					mesh4 = mio.read(msh_file_path)
					print('Use to_volume method create tetrahedron from "stl" model.')
			elif file_type == 'msh':
				print('Access "msh" model directly.')
				mesh4 = mio.read(obj_path)


			else:
				mesh4 = mio.read(obj_path)
			# obj = tm.load_mesh("./obj/spot_triangulated.obj")
			obj = tm.load_mesh("./obj/spot20.obj")
			self.obj = obj

			import pyvista as pv
			import tetgen

			# sphere = pv.Sphere()
			# spot = pv.read('./obj/spot_triangulated.obj')
			sphere = pv.Sphere()
			spot = pv.read('./obj/spot20.obj')
			self.duplicate_index_2_obj_index = self.process_obj_duplicate_point('./obj/spot20.obj')
			# spot = pv.read('./obj/cube.stl')
			self.tet = tetgen.TetGen(spot)
			'''
			mindihedral:
			minratio:
			'''
			self.tet.tetrahedralize(order=1, mindihedral=10, minratio=5.0)

			self.mesh4 = mesh4
			# vertices = mesh4.points
			# tet.grid.extract_cells()

			cells = self.tet.grid.cells.reshape(-1, 5)[:, 1:]
			cell_center = self.tet.grid.points[cells].mean(1)

			# extract cells below the 0 xy plane
			mask = cell_center[:, 2] < 0
			cell_ind = mask.nonzero()[0]

			vertices = self.tet.node

			# Get the indices of the points of the tetrahedron
			# tetra_indices = mesh4.cells[0].data
			# element_indices = tetra_indices
			element_indices = self.tet.elem

			faces = []
			# # for tetra in tetra_indices:
			for tetra in self.tet.elem:
				x, y, z, w = tetra
				faces.append([x, y, z])
				faces.append([x, y, w])
				faces.append([x, z, w])
				faces.append([y, z, w])
			faces = np.array(faces)
			self.surface, self.surface_vertex = self.extract_surface(faces, vertices)
			faces = self.surface
			self.uv = self.recover_uv(obj, self.surface, vertices)



			self.remap_surface = np.copy(self.surface)
			self.remap_surface_index(self.remap_surface, self.surface_vertex)
			self.map_index = self.link_obj_vertex(obj, self.surface, vertices)
			# self.uv = self.recover_uv_from_disk()
			# self.texture = tm.visual.texture.TextureVisuals(uv=self.uv)
			# faces = tet.f
			# faces = tetra_indices # Volume
			# vertices = mesh.vertices
			# mass = self.rho / tetra_indices.shape[0]
			mass = self.rho / self.tet.elem.shape[0]
			num_sides = 4
			self.center = ti.Vector(config.get('center'))
		return vertices, faces, element_indices, mass, num_sides

	@staticmethod
	def process_obj_duplicate_point(path):
		duplicate_point = {}
		duplicate_point_list = []
		temp = {}
		ret = {}
		with open(path, 'r') as file:
			for line in file:
				data_list = line.split(' ')
				if len(data_list) > 0 and data_list[0] == 'f':
					for index in range(1, 4):
						v_id, t_id = data_list[index].split('/')
						if not int(v_id) in duplicate_point:
							duplicate_point[int(v_id)] = set()
						duplicate_point[int(v_id)].add(int(t_id.replace('\n', '')))

			for i in range(len(duplicate_point)):
				count = len(duplicate_point[i+1])
				for j in range(count):
					duplicate_point_list.append(i+1)

			for i in range(len(duplicate_point)):
				indices = [j for j, x in enumerate(duplicate_point_list) if x == i+1]
				temp[i+1] = indices

			for k, v in temp.items():
				if len(v) == 1:
					ret[v[0]] = v
				else:
					for i in v:
						ret[i] = v

		print('HHH')
		return ret

	@staticmethod
	def recover_uv_from_disk():
		import pickle
		with open('list_data.pkl', 'rb') as file:
			loaded_list = pickle.load(file)
			return loaded_list

	def link_obj_vertex(self, ori_obj, surface, vertices):
		mesh = tm.Trimesh(vertices=self.tet.grid.points[self.surface_vertex], faces=self.remap_surface)
		map_index = []
		error_list = []
		for pos in ori_obj.vertices:
			v_d, v_index = mesh.nearest.vertex(pos)
			error_list.append(v_d)
			map_index.append(v_index)
		# vertex_set = set()
		# for tri in surface:
		# 	v0, v1, v2 = tri
		# 	vertex_set.add(v0)
		# 	vertex_set.add(v1)
		# 	vertex_set.add(v2)
		# map_index = []
		# error_list = []
		# for indice in vertex_set:
		# 	pos = vertices[indice]
		# 	v_d, v_index = ori_obj.nearest.vertex(pos)
		# 	error_list.append(v_d)
		# 	if abs(v_d) < 1e-7:
		# 		map_index.append(v_index)
		return map_index

	def recover_uv(self, ori_obj, surface, vertices):
		start_time = time.time()
		vertex_set = set()
		for tri in surface:
			v0, v1, v2 = tri
			vertex_set.add(v0)
			vertex_set.add(v1)
			vertex_set.add(v2)

		total_max = -np.inf
		error_list = []
		uv = []
		map_index = []
		for indice in vertex_set:
			pos = vertices[indice]
			v_d, v_index = ori_obj.nearest.vertex(pos)
			error_list.append(v_d)
			map_index.append(v_index)

		for indice in vertex_set:
			pos = vertices[indice]
			v_d, v_index = ori_obj.nearest.vertex(pos)
			if v_d < 1e-7:
				uv.append(ori_obj.visual.uv[v_index])
			else:
				p_in_tri, d, tri_index = ori_obj.nearest.on_surface(pos.reshape(-1, 3))
				p0, p1, p2 = ori_obj.vertices[ori_obj.faces[tri_index[0]]]
				alpha, beta, gamma = self.barycentric_coordinates(p0, p1, p2, p_in_tri)
				ori_uv0, ori_uv1, ori_uv2 = ori_obj.visual.uv[ori_obj.faces[tri_index[0]]]
				# if alpha < 1e-7:
				# 	test = np.array([2402, 2410])
				# 	result = np.any(np.isin(ori_obj.faces, test), axis=1)
				# 	matching_indices = np.where(result)
				# 	kk = ori_obj.faces[matching_indices]
				# 	print('yjk')
				# if alpha < 1e-7 or beta < 1e-7 or gamma < 1e-7:
				# 	index_ = [alpha, beta, gamma].index(min([alpha, beta, gamma]))
				# 	uv.append([0, 0])
				# else:
				uv.append(alpha * ori_uv0 + beta * ori_uv1 + gamma * ori_uv2)
				# uv.append(alpha * ori_uv0 + beta * ori_uv1 + gamma * ori_uv2)
				# uv.append([0.0, 0.0])

			# if d[0] > max_d:
			# 	max_d = d[0]
			# 	max_index = indice

		# min_d_list = []
		# for indice in vertex_set:
		# 	min_d = np.inf
		# 	# min_f_indice = 0
		# 	# indice = 1509
		# 	pos = vertices[indice]
		# 	min_judge = np.inf
		# 	uv_point = 0
		# 	for index in range(ori_obj.face_normals.shape[0]):
		# 		normal = ori_obj.face_normals[index]
		# 		D = - np.dot(normal, ori_obj.vertices[ori_obj.faces[index][0]])
		# 		d = np.dot(pos, normal) + D
		# # 		d2 = d**2
		# 		p_in_plane = pos - normal * d
		# 		p0, p1, p2 = ori_obj.vertices[ori_obj.faces[index]]
		# 		# a = np.dot(np.cross(p1-p0, p_in_plane-p0), normal)
		# 		# b = np.dot(np.cross(p2-p1, p_in_plane-p1), normal)
		# 		# c = np.dot(np.cross(p0-p2, p_in_plane-p2), normal)
		# 		aa, bb, cc = self.barycentric_coordinates(p0, p1, p2, p_in_plane)
		# 		judge_value = ((aa + bb + cc) - 1) ** 2
		# 		min_judge = min(judge_value, min_judge)
		# 		is_inside1 = True if ((aa + bb + cc) - 1) ** 2 < 1e-7 else False
		# 		# is_inside = True if a > 0 and b > 0 and c > 0 else False
		#
		# 		if is_inside1 and abs(d) < min_d:
		# 			# print(d)
		# 			min_d = abs(d)
		# 			ori_uv0, ori_uv1, ori_uv2 = ori_obj.visual.uv[ori_obj.faces[index]]
		# 			uv_point = aa * ori_uv0 + bb * ori_uv1 + cc * ori_uv2
		# 	# if type(uv_point) == type(0):
		# 	# 	print('FXXXXX')
		# 	uv.append(uv_point)
		# 	min_d_list.append(min_d)
		# 	print('Recover uv  process {}%'.format(indice/ len(vertex_set) * 100))


		# 		if d2 < min_d and is_inside and is_inside1:
		# 			min_d = d2
		# 			min_f_indice = index
		#
		# 	# ori_v0, ori_v1, ori_v2 = ori_obj.faces[min_f_indice]
		# 	p0, p1, p2 = ori_obj.vertices[ori_obj.faces[min_f_indice]]
		# 	a, b, c = self.barycentric_coordinates(p0, p1, p2, pos)
		# 	ori_uv0, ori_uv1, ori_uv2 = ori_obj.visual.uv[ori_obj.faces[min_f_indice]]
		# 	uv = a * ori_uv0 + b * ori_uv1 + c* ori_uv2
		# 	print(min_d, min_f_indice)
		# pass
		print('Recover uv consume time: {}'.format(time.time() - start_time))
		print('hh')
		print('total max is: {}'.format(total_max))
		return uv

	@staticmethod
	def barycentric_coordinates(p0, p1, p2, p):
		A, B, C = p0, p1, p2
		P = p

		volume_ABC = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
		volume_PBC = 0.5 * np.linalg.norm(np.cross(B - P, C - P))
		volume_PAC = 0.5 * np.linalg.norm(np.cross(C - P, A - P))
		volume_PAB = 0.5 * np.linalg.norm(np.cross(A - P, B - P))

		lambda1 = volume_PBC / volume_ABC
		lambda2 = volume_PAC / volume_ABC
		lambda3 = volume_PAB / volume_ABC

		return lambda1, lambda2, lambda3


	def extract_surface(self, f, vertices):
		faces_dict= {}

		for tetra in self.tet.elem:
			x, y, z, w = tetra
			# faces_dict
			faces_dict[tuple(sorted([x, y, z]))] = tetra
			faces_dict[tuple(sorted([x, y, w]))] = tetra
			faces_dict[tuple(sorted([x, z, w]))] = tetra
			faces_dict[tuple(sorted([y, z, w]))] = tetra

		d = {}
		surface = []
		for tri in f:
			key = tuple(sorted(tri))
			if key in d:
				d[key] += 1
			else:
				d[key] = 1

		for k, v in d.items():
			if v == 1:
				tetra = faces_dict[k]
				inner_point_indice = self.difference(tetra, k)[0]
				inner_point = vertices[inner_point_indice]
				f0, f1, f2 = k
				p0, p1, p2 = vertices[f0], vertices[f1], vertices[f2]

				if np.cross(p1-p0, p2-p0).dot(inner_point - p0) < 0:
					surface.append([f0, f1, f2])
				else:
					surface.append([f0, f2, f1])

		vertex = set()
		for f in surface:
			v0, v1, v2 = f
			vertex.add(v0)
			vertex.add(v1)
			vertex.add(v2)

		vertex = list(vertex)

		return np.array(surface), list(vertex)

	@staticmethod
	def remap_surface_index(surface, vertex):
		for index in range(len(surface)):
			for dim in range(len(surface[index])):
				if surface[index][dim] > len(vertex):
					surface[index][dim] = vertex.index(surface[index][dim])

	@staticmethod
	def difference(s1, s2):
		ret = list(set(s1) - set(s2))
		return ret

	def update_obj(self):
		points = self.particles.pos.to_numpy()
		# self.tet.grid.points = points
		test_set = set()
		for i in range(len(self.map_index)):
			# obj_index_list = self.duplicate_index_2_obj_index[self.map_index[i]]
			# for index in obj_index_list:
			self.obj.vertices[i] = points[self.map_index[i]]
				# test_set.add(index)
		print('hhh')
		# self.tmmesh.vertices = self.particles.pos.to_numpy()

	def save_obj(self, file_name):
		# texture = tm.visual.texture.TextureVisuals(uv=self.uv)
		# mesh = tm.Trimesh(vertices=self.tet.grid.points[self.surface_vertex], faces=self.remap_surface, visual=texture)
		# mesh.visual.uv = self.uv
		with open(file_name, 'w') as f:
			# e = mesh.export(file_type='obj')
			e = self.obj.export(file_type='obj')
			f.write(e)
		# self.tet.grid.save(file_name, False)
		# self.tmmesh.vertices
		# with open(file_name, "w") as f:
		# 	e = self.tmmesh.export(file_type='obj')
		# 	f.write(e)
		# mio
		# points = self.mesh4.points
		# # # tet_cells = self.mesh4.cells['tetra']
		# tet_cells = None
		# for cell_block in self.mesh4.cells:
		# 	if cell_block.type == 'tetra':
		# 		tet_cells = cell_block.data
		# 		break
		#
		# if tet_cells is None:
		# 	print("No tetrahedral cells found in the MSH file.")
		#
		# tri_cells = []
		# for tet in tet_cells:
		# 	tri_cells.append([tet[0], tet[1], tet[2]])
		# 	tri_cells.append([tet[0], tet[1], tet[3]])
		# 	tri_cells.append([tet[1], tet[2], tet[3]])
		# 	tri_cells.append([tet[0], tet[2], tet[3]])

		# # tet_cells = None
		# # for cell_block in self.mesh4.cells:
		# # 	if cell_block.type == 'tetra':
		# # 		tet_cells = cell_block.data
		# # 		break
		# cells = [("tetra", tet_cells)]
		# obj_mesh = mio.Mesh(points=points, cells=[('triangle', tri_cells)])
		# mio.write(file_name, obj_mesh, file_format='obj')

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