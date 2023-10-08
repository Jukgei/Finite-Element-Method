# coding=utf-8

import utils
import taichi as ti
from main import delta_time, damping, s_lambda, mu, mass, g_dir
from main import element_cnt, particle_cnt
from main import matA, vecb, vec
from main import dim, tensors_type, mat


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
		V = utils.compute_volume(X)
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
			if ti.abs(A[i, i]) < 1e-5:
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