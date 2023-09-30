# coding=utf-8

import taichi as ti
import numpy as np
import trimesh as tm
from trimesh.interfaces import gmsh
import meshio as mio
import solver.kinematic as ki

ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

dim = 3
if dim == 2:
	vec = ti.math.vec2
	mat = ti.math.mat2
	matA = ti.types.matrix(dim**2, dim**2, ti.f32)
	vecb = ti.types.vector(dim**2, ti.f32)
	index = ti.math.ivec3
else: # 3d
	vec = ti.math.vec3
	mat = ti.math.mat3
	matA = ti.types.matrix(dim**2, dim**2, ti.f32)
	vecb = ti.types.vector(dim**2, ti.f32)

	index = ti.math.ivec4

# dim_ndarray = ti.types.ndarray
width = 640
height = 640


if dim == 2:
	block_center = [0.5, 0.5]
	center = ti.Vector([0.72, 0.8])
	g_dir = [0, -1]
	block_radius = 0.33
	delta_time = 5e-5
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

# center = ti.Vector([0.55, 0.3])
v_refect = -0.3
# delta_time = 5e-4
side_length = 0.2  #
subdivisions = 10  #

auto_diff = False

if dim == 2:
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
	mass = rho * A
	num_sides = 3
else:
	# mm = tm.load_mesh('./obj/stanford-bunny.obj')
	obj = tm.load_mesh('./obj/cube2.stl')
	obj.apply_scale(1)
	msh = gmsh.to_volume(obj, './obj/cube2.msh', mesher_id=7)
	mesh4 = mio.read('./obj/cube2.msh')
	vertices = mesh4.points

	# 获取四面体的点的indices
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
	mass = rho / tetra_indices.shape[0]
	num_sides = 4

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

ti_vertices = ti.Vector.field(dim, ti.f32, shape=vertices.shape[0])
ti_vertices.from_numpy(vertices)
ti_faces = ti.Vector.field(3, ti.i32, shape=faces.shape[0])
ti_faces.from_numpy(faces)
ti_element = ti.Vector.field(num_sides, ti.i32, shape=element_indices.shape[0])
ti_element.from_numpy(element_indices)

phi = ti.field(dtype=ti.f32, shape=faces.shape[0])
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
indices = ti.field(dtype=ti.i32, shape=faces.shape[0] * 3)

Particle = ti.types.struct(
	pos=vec,
	vel=vec,
	vel_f=vec,
	acc=vec,
	mass=ti.f32,
	force=vec,
	ref_pos=vec
)

Mesh = ti.types.struct(
	p0=ti.i32,
	p1=ti.i32,
	p2=ti.i32,
	ref=mat
)

Element = ti.types.struct(
	vertex_indices=index,
	ref=mat,
	A=matA,
	b=vecb,
	x=vecb
)

particle_cnt = vertices.shape[0]
mesh_cnt = faces.shape[0]
element_cnt = element_indices.shape[0]
meshs = Mesh.field(shape=mesh_cnt)
elements = Element.field(shape=element_cnt)
# tensors = ti.field(ti.f32, shape=(element_cnt, dim, dim, dim, dim))
tensors = ti.ndarray(dtype=mat, shape=(element_cnt, dim, dim))
tensors_com = ti.ndarray(dtype=mat, shape=(element_cnt, dim, dim))
tensors_type = ti.types.ndarray(dtype=ti.math.mat3, ndim=3)
# tensors_cache = ti.types.ndarray(dtype=mat, ndim=(element_cnt, dim, dim))
tensors_cache = ti.field(dtype=ti.f32, shape=(element_cnt, dim, dim, dim, dim))
particles = Particle.field(shape=particle_cnt, needs_grad=True)
print('Vertex count: {}'.format(particle_cnt))
print('Mesh count: {}'.format(mesh_cnt))
print('Element count: {}'.format(element_cnt))
print('Element mass: {}'.format(mass))

# import solver.explicit_auto_diff as
import utils
from solver.explicit_auto_diff import compute_energy
from solver.explicit import neo_hookean_1_grad

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
		V = (1/6) * ti.abs(p0.dot(p1.cross(p2)))
	return V


@ti.kernel
def particles_init():

	for i in range(particle_cnt):
		particles[i].pos = ti_vertices[i] + center
		particles[i].ref_pos = particles[i].pos
		particles[i].mass = mass

@ti.kernel
def elements_init():
	for i in range(element_cnt):
		elements[i].vertex_indices = ti_element[i]

		r = mat(0)
		p_0 = particles[ti_element[i].x].pos
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_i = particles[ti_element[i][j+1]].pos
				r[:, j] = p_i - p_0

		elements[i].ref = ti.math.inverse(r)


@ti.kernel
def mesh_init():
	for i in range(mesh_cnt):
		meshs[i].p0 = ti_faces[i][0]
		meshs[i].p1 = ti_faces[i][1]
		meshs[i].p2 = ti_faces[i][2]

		indices[i * 3 + 0] = ti_faces[i][0]
		indices[i * 3 + 1] = ti_faces[i][1]
		indices[i * 3 + 2] = ti_faces[i][2]


def fem_implicit():
	neo_hookean_2_grad(tensors, tensors_com)
	# a = tensors.to_numpy()
	# print(elements[0].A.to_numpy())
	# print(a)

	# print(a)
	solve()
	advect_implicit()

def fem():
	neo_hookean_1_grad(particles, elements)
	# neo_hookean_2_grad(tensors, tensors_com)
	# print(tensors.to_numpy())
	# solve()

# dim1 = 9
@ti.func
def kronecker_product(a: ti.template(), b: ti.template()):

	ret = ti.Matrix([[0.0 for _ in ti.static(range(dim **2))] for __ in range(dim **2)])
	for i in ti.static(range(dim)):
		for j in ti.static(range(dim)):
			ret[i * dim: (i+1) *dim, j * dim: (j + 1)*dim] = a[i, j] * b
	# print(ret.to_numpy())
	return ret


@ti.kernel
def testt(tensors: tensors_type):
	kronecker_product_tensor(0, ti.math.eye(3), ti.math.eye(3), tensors)
	order = ti.math.ivec4(3,1,2,0)
	reshape(tensors, order)


@ti.kernel
def test() -> ti.types.matrix(9, 9, ti.f32):
	# I = ti.math.eye(3)
	# a = ti.math.mat3([1,2,3,4,5,6,7,8,9])
	# ret = kronecker_product(a, I)
	# print(ret[0:3,0:3])
	# print(ret.n, ret.m)
	# bb = ti.Matrix([1,2,3,4])
	# bb.to_numpy()
	F = mat(1,2,3,4,5,6,7,8,9)
	part3_1 = F
	part3_1_1 = ti.Matrix([[0.0 for _ in ti.static(range(dim ** 2))] for __ in range(dim ** 2)])

	# part3_1_1[0:3, 0:3] = ti.Matrix.cols([part3_1[:, 0], part3_1[:, 0], part3_1[:, 0]])
	# part3_1_1[0:3, 3:6] = ti.Matrix.cols([part3_1[:, 1], part3_1[:, 1], part3_1[:, 1]])
	# part3_1_1[0:3, 6:9] = ti.Matrix.cols([part3_1[:, 2], part3_1[:, 2], part3_1[:, 2]])
	# part3_1_1[3:6, :] = part3_1_1[0: 3, :]
	# part3_1_1[6:9, :] = part3_1_1[0: 3, :]

	return part3_1_1

@ti.func
def kronecker_product_tensor(i: ti.i32, a: ti.template(), b: ti.template(), tensors: tensors_type):

	for j in ti.static(range(dim)):
		for k in ti.static(range(dim)):
			tensors[i, j, k] = a[j, k] * b


@ti.func
def reshape(tensors: tensors_type, order: ti.types.vector(4, ti.i32)):
	for I in ti.grouped(tensors):
		index, i, j = I
		m = tensors[I]
		for k in ti.static(range(dim)):
			for l in ti.static(range(dim)):
				tensors_cache[index, i, j, k, l] = m[k, l]

	for I in ti.grouped(ti.ndrange(element_cnt, dim, dim, dim, dim)):
		index = I[0]
		i, j, k, l = I[order[0]+1], I[order[1]+1], I[order[2]+1], I[order[3]+1]
		tensors[index, i, j][k, l] = tensors_cache[I]

@ti.func
def tensors2matrix(tensors: tensors_type, index: ti.i32) -> matA:
	m = matA(0)
	for i in ti.static(range(dim)):
		for j in ti.static(range(dim)):
			m[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = tensors[index, i, j]
	return m

@ti.func
def tensors_plus(src: tensors_type, ten: tensors_type, index: ti.i32):
	for I in ti.grouped(ten):
		if I[0] == index:
			src[I] += ten[I]

@ti.kernel
def neo_hookean_2_grad(tensors: tensors_type, tensors_com: tensors_type):# -> ti.types.matrix(dim*dim, dim*dim, ti.f32):
	# ret = ti.Matrix([[0.0 for _ in ti.static(range(dim ** 2))] for __ in range(dim ** 2)])
	for i in range(element_cnt):
		# t = tensor[i]
		element = elements[i]
		p_0 = particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = particles.pos[element.vertex_indices[j + 1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		F = X @ R_inv
		X_inv = ti.math.inverse(X)
		V = compute_volume(X)
		# part1 = mu * kronecker_product(R_inv @ R_inv.transpose(), ti.math.eye(dim))
		kronecker_product_tensor(i, ti.math.eye(dim), R_inv@R_inv.transpose(), tensors_com)
		order = ti.math.ivec4(3, 1, 2, 0)
		reshape(tensors_com, order)
		tensors_plus(tensors, tensors_com, i)

		F_inv = ti.math.inverse(F)
		# print('F det {}, X det {}, i {}'.format(F.determinant(), X.determinant(), i))
		kronecker_product_tensor(i, (s_lambda * ti.log(F.determinant()) - mu) * X_inv.transpose(), - X_inv, tensors_com)
		order = ti.math.ivec4(1, 2, 3, 0)
		reshape(tensors, order)
		tensors_plus(tensors, tensors_com, i)

		R_inv_F_inv = R_inv @ F_inv
		# part2 = - mu * kronecker_product(R_inv_F_inv.transpose(), R_inv_F_inv).transpose()
		# part3_1 = s_lambda * F_inv.transpose()
		# part3_1_1 = ti.Matrix([[0.0 for _ in ti.static(range(dim ** 2))] for __ in range(dim ** 2)])
		# TODO: print to check

		# part3_1_1[0:3, 0:3] = ti.Matrix.cols([part3_1[:, 0], part3_1[:, 0], part3_1[:, 0]])
		# part3_1_1[0:3, 3:6] = ti.Matrix.cols([part3_1[:, 1], part3_1[:, 1], part3_1[:, 1]])
		# part3_1_1[0:3, 6:9] = ti.Matrix.cols([part3_1[:, 2], part3_1[:, 2], part3_1[:, 2]])
		# part3_1_1[3:6, :] = part3_1_1[0: 3, :]
		# part3_1_1[6:9, :] = part3_1_1[0: 3, :]
		# part3 = part3_1_1 + s_lambda * part2
		#
		# grad2 = (part1 + part2 + part3) * V
		# elements[i].A = ti.math.eye(dim**2) - delta_time **2 * grad2 / mass
		elements[i].A = ti.math.eye(dim **2) - V * tensors2matrix(tensors, i) * delta_time ** 2/ mass
		# elements[i].A *= V
		elements[i].x = vecb(0)

		_1_grad = mu * F @ R_inv.transpose() + (- mu * R_inv_F_inv).transpose() + (s_lambda * ti.log(F.determinant()) * R_inv_F_inv).transpose()
		_1_grad *= V
		f1 = vec(_1_grad[:, 0])
		f2 = vec(_1_grad[:, 1])
		f3 = vec(_1_grad[:, 2])
		b = vecb(0)
		v = vecb(0)
		b[0:3] = vec(f1.x, f2.x, f3.x)
		b[3:6] = vec(f1.y, f2.y, f3.y)
		b[6:9] = vec(f1.z, f2.z, f3.z)
		b *= delta_time / mass
		p0, p1, p2, p3 = element.vertex_indices
		v1, v2, v3 = particles.vel[p1], particles.vel[p2], particles.vel[p3]
		v[0:3] = vec(v1.x, v2.x, v3.x)
		v[3:6] = vec(v1.y, v2.y, v3.y)
		v[6:9] = vec(v1.z, v2.z, v3.z)
		b += v
		elements[i].b = b
	#     ret += grad2
	# return ret

# @ti.kernel
def solve():
	for i in range(element_cnt):
		jacobi_iter(i)

# @ti.func
@ti.kernel
def jacobi_iter(index: ti.i32):
	element = elements[index]
	A = element.A
	b = element.b
	x = element.x
	iter_cnt = 0
	err = ti.math.inf
	threshold = 1e-1
	# last_err = 0
	ti.loop_config(serialize=True)
	while err > threshold:
		for i in ti.static(range(dim**2)):
			if ti.abs(A[i, i] ) < 1e-5:
				# print('i {} is {}, index {}'.format(i, A[i, i], index))
				x[i] = 0.0
				if index == 1:
					print('up, {}'.format(i))
			else:
				# continue
				x[i] = (b[i] - A[i,:]@x + A[i, i]*x[i]) / A[i, i]
				if index == 1:
					print('down, {}, {}, {}, {}'.format(i, (b[i] - A[i,:]@x + A[i, i]*x[i]), A[i, i], x[i]))
		if index == 1:
			print('x is {}'.format(x))
		last_err = err
		err = (A@x - b).norm()
		iter_cnt += 1
		# if ti.abs(last_err - err) < threshold:
		#     break
		# if index == 1:
		print('loss {}, iter {}, index {}, loss diff {}, last_err {}'.format(err, iter_cnt, index, ti.abs(last_err - err), last_err))
	elements[index].x = x
	p0, p1, p2, p3 = element.vertex_indices
	# v1, v2, v3 = x[0:3], x[3:6], x[6:9]
	v1 = vec(x[0], x[3], x[6])
	v2 = vec(x[1], x[4], x[7])
	v3 = vec(x[2], x[5], x[8])
	# f1 = (v1 - particles.vel_f[p1]) * mass
	# f2 = (v2 - particles.vel_f[p2]) * mass
	# f3 = (v3 - particles.vel_f[p3]) * mass
	# f0 = -f1-f2-f3
	v0 = -v1-v2-v3
	particles.vel[p0] = v0
	particles.vel[p1] = v1
	particles.vel[p2] = v2
	particles.vel[p3] = v3
	# print('loss {}, iter {}'.format(err, iter_cnt))


@ti.kernel
def advect_implicit():
	for index in range(particle_cnt):
		# particles[index].vel +=
		v = particles[index].vel + 9.8 * ti.Vector(g_dir)
		v *= ti.exp(-delta_time * damping)

		for i in ti.static(range(dim)):
			if particles.pos[index][i] < 0 and v[i] < 0:
				particles.vel[index][i] = 0
				v[i] = 0

			if particles.pos[index][i] > 1 and v[i] > 0:
				particles.vel[index][i] = 0
				v[i] = 0

		particles.pos[index] += v * delta_time


def render2d(gui):
	pos_ = particles.pos.to_numpy()
	phi_ = phi.to_numpy()

	base_ = 0.13
	gui.triangles(a=pos_[meshs.p0.to_numpy()], b=pos_[meshs.p1.to_numpy()], c=pos_[meshs.p2.to_numpy()],
				  color=ti.rgb_to_hex([phi_ + base_, base_, base_]))
	gui.circles(pos_, radius=2, color=0xAAAA00)
	gui.circle(block_center, color=0x343434, radius=block_radius * width)
	gui.show()

def render3d(window, camera):

	canvas = window.get_canvas()
	scene = ti.ui.Scene()

	# Camera & light
	camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
	scene.set_camera(camera)
	scene.ambient_light((0.8, 0.8, 0.8))
	scene.point_light(pos=(3.5, 3.5, 3.5), color=(1, 1, 1))
	scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)
	pos_ = particles.pos.to_numpy()

	# scene.particles(particles.pos, color=(1.0, 1.0, 1), radius=.0001)
	scene.mesh(particles.pos, indices, show_wireframe=True)
	canvas.scene(scene)
	window.show()

if __name__ == '__main__':
	# utils.neo_hookean_3d_2ord()
	# testt(tensors)
	# a = tensors.to_numpy()
	# b = a[0]
	# print(b)
	# m = test()
	# print(m.to_numpy())
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

	particles_init()
	mesh_init()
	elements_init()
	frame_cnt = 0
	# element = elements[1]
	# x, y, z, w = element.vertex_indices
	# print('{}, {}, {}, {}'.format(particles.pos[x], particles.pos[y], particles.pos[z], particles.pos[w]))
	debug1 = True
	while widget.running:
		if widget.is_pressed('c'):
			print('Camera position [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_position)))
			print('Camera look at [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_lookat)))
			print('Camera up [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_up)))
		frame_cnt+=1
		# if frame_cnt == 120:
		#     particles[2].pos = center + ti.Vector([0.1, 0.1])
		# U[None] = 0
		for i in range(50):
			if not auto_diff:
				fem()
			else:
				# if frame_cnt == 1:
				with ti.ad.Tape(loss=U):
					# compute_energy()
					compute_energy(particles, elements, U, phi)
			ki.kinematic(particles)
			# if debug1:
			# fem_implicit()
		debug1 = False
		if dim == 2:
			render2d(widget)
		else:
			render3d(widget, camera)
