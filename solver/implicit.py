# coding=utf-8

import utils
import taichi as ti
from main import delta_time, damping, s_lambda, mu, g_dir
# from main import element_cnt, particle_cnt
from main import dim, mat, vec
from main import block_radius, block_center


@ti.func
def kronecker_product(a: ti.template(), b: ti.template()):
	ret = ti.types.matrix(a.n*b.n, a.m*b.m, ti.f32)(0)
	for i in ti.static(range(a.n)):
		for j in ti.static(range(a.m)):
			ret[i*b.n:i*b.n+b.n, j*b.m:j*b.m+b.m] = a[i, j] * b
	return ret

@ti.func
def tensor_transpose(a: ti.template(), rows: ti.template(), cols: ti.template()):
	ret = ti.types.matrix(a.n, a.m, ti.f32)(0)
	# for i in ti.static(range(rows)):
	# 	for j in ti.static(range(cols)):
	# 		ret[i*rows:i*rows+rows, j*cols:j*cols+cols] = a[j*cols:j*cols+cols, i*rows:i*rows+rows]
	ret[0, :] = a[0, :]
	ret[1, :] = a[2, :]
	ret[2, :] = a[1, :]
	ret[3, :] = a[3, :]

	# ret[:, 0] = a[:, 0]
	# ret[:, 1] = a[:, 2]
	# ret[:, 2] = a[:, 1]
	# ret[:, 3] = a[:, 3]

	return ret

@ti.func
def compute_linear_system_vector_b(obj: ti.template()):
	# ret = ti.types.vector(obj.particle_cnt * dim, ti.f32)(0.0)
	for i in range(obj.particle_cnt):
		obj.vec_b[i] = obj.particles.vel[i]
		# for j in ti.static(range(dim)):
		# 	ret[i*dim+j] = obj.particles.vel[i][j]

	for i in range(obj.element_cnt):
		element = obj.elements[i]
		p_0 = obj.particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = obj.particles.pos[element.vertex_indices[j + 1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		# deformation gradient
		F = X @ R_inv
		F_inv = ti.math.inverse(F)
		V = element.volume

		force = (mu * F - mu * F_inv.transpose() + s_lambda / 2 * ti.log((F.transpose() @ F).determinant()) * F_inv.transpose()) @ R_inv.transpose()
		force *= -V

		# f = ti.types.vector(dim*obj.particle_cnt, ti.f32)(0.0)

		f0 = vec(0.0)
		m_ = 1.0 / obj.mass
		for i in ti.static(range(dim)):
			f_n = vec(force[:, i])
			f0 -= f_n
			particle_index = element.vertex_indices[i+1]
			obj.vec_b[particle_index] += delta_time * m_ * f_n
			# for j in ti.static(range(dim)):
			# 	f[particle_index*dim+j] = f_n[j]

		particle_index = element.vertex_indices.x
		obj.vec_b[particle_index] += delta_time * m_ * f0
		# for i in ti.static(range(dim)):
		# 	f[particle_index*dim+i] = f0[i]
		# print('f is ', f)
		# M = 1.0 / obj.mass * ti.math.eye(dim*obj.particle_cnt)
	# 	ret += delta_time * M @ f
	# return ret

@ti.func
def compute_linear_system_matrix_a(obj: ti.template()):
	# ret = ti.types.matrix(obj.particle_cnt * dim, obj.particle_cnt * dim, ti.f32)(0.0)
	for i in range(obj.element_cnt):
		element = obj.elements[i]
		p_0 = obj.particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = obj.particles.pos[element.vertex_indices[j + 1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		# deformation gradient
		F = X @ R_inv
		F_inv = ti.math.inverse(F)
		V = element.volume

		log_J = ti.log(ti.max(F.determinant(), 1e-4))
		F_inv_T = F_inv.transpose()
		dF_dx00 = ti.types.matrix(dim, dim, ti.f32)(0.0)
		temp = ti.types.matrix(dim, dim**2, ti.f32)(0.0)
		for i in ti.static(range(dim)):
			dF_dxi0 = ti.types.matrix(dim, dim, ti.f32)(0.0)
			for j in ti.static(range(dim)):
				dF = mat(0.0)
				if i == j:
					dF = ti.math.eye(dim)
				dF = dF@R_inv
				dF_T = dF.transpose()
				dF_dxij = mu * dF + (mu - s_lambda * log_J) * F_inv_T @ dF_T @ F_inv_T + s_lambda * (F_inv @ dF).trace() * F_inv_T
				dF_dxij = -V * dF_dxij @ R_inv.transpose()
				dF_dxij = (delta_time ** 2) * (1.0 / obj.mass * ti.math.eye(dim))@dF_dxij
				dF_dx00 += dF_dxij
				rows = element.vertex_indices[i+1]
				cols = element.vertex_indices[j+1]
				obj.matrix_A[rows, cols] += dF_dxij
				# for k in ti.static(range(dim)):
				# 	for l in ti.static(range(dim)):
				# 		ret[rows*dim + k, cols*dim + l] += dF_dxij[k, l]
				dF_dxi0 -= dF_dxij

				temp[0:dim, j*dim:j*dim+dim] -= dF_dxij

			rows = element.vertex_indices[i+1]
			cols = element.vertex_indices[0]
			obj.matrix_A[rows, cols] += dF_dxi0
			# for m in ti.static(range(dim)):
			# 	for n in ti.static(range(dim)):
			# 		ret[rows*dim+m, cols*dim+n] += dF_dxi0[m, n]

		rows = element.vertex_indices[0]
		cols = element.vertex_indices[0]
		obj.matrix_A[rows, cols] += dF_dx00
		# for i in ti.static(range(dim)):
		# 	for j in ti.static(range(dim)):
		# 		ret[rows*dim+i, cols*dim+j] += dF_dx00[i, j]

		# dF0_dx1
		# dF0_dx2
		for i in ti.static(range(dim)):
			r = element.vertex_indices[0]
			c = element.vertex_indices[i+1]
			obj.matrix_A[r, c] += temp[:, i*dim:i*dim+dim]
			# for k in ti.static(range(dim)):
			# 	for l in ti.static(range(dim)):
			# 		ret[r*dim+k, c*dim+l] += temp[k, l+i*dim]

	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			I = mat(0.0)
			if i == j:
				I = ti.math.eye(dim)
			obj.matrix_A[i, j] = I - obj.matrix_A[i, j]

	# ret = ti.math.eye(obj.particle_cnt * dim) - ret
	# return ret


@ti.kernel
def neo_hookean_2_grad(obj: ti.template()):# -> ti.types.vector(8, ti.f32):
	# ret = ti.Matrix([[0.0 for _ in ti.static(range(dim ** 2))] for __ in range(dim ** 2)])
	# print('Once!')
	obj.matrix_A.fill(mat(0.0))
	obj.vec_b.fill(vec(0.0))
	obj.vec_x.fill(vec(0.0))
	compute_linear_system_matrix_a(obj)
	compute_linear_system_vector_b(obj)
	# v = b
	# v = jacobi_iter(A, b)
	# print('before', obj.vec_x[0])
	jacobi_iter_field(obj)
	# print('after',obj.vec_x[0])
	# print(v)
	# print(b)
	# x = b
	# print(b)
	# print(x)
	# A.to_numpy()
	# b = com
	# for i in range(obj.element_cnt):
	# 	# t = tensor[i]
	# 	element = obj.elements[i]
	# 	p_0 = obj.particles[element.vertex_indices.x].pos
	# 	X = mat(0)
	# 	for j in ti.static(range(dim)):
	# 		if j + 1 <= dim:
	# 			p_j = obj.particles.pos[element.vertex_indices[j + 1]]
	# 			X[:, j] = p_j - p_0
	#
	# 	R_inv = element.ref
	# 	F = X @ R_inv
	# 	F_inv = ti.math.inverse(F)
	# 	X_inv = ti.math.inverse(X)
	# 	V = element.volume
	# 	part1 = mu * kronecker_product(R_inv@R_inv.transpose(), ti.math.eye(dim))
	# 	part2_ = tensor_transpose(kronecker_product(-X_inv.transpose(), X_inv), dim, dim)
	# 	part2 = -mu * part2_
	# 	part3 = s_lambda * (kronecker_product((R_inv @ X_inv @ R_inv).transpose(), X_inv.transpose()) + ti.log(F.determinant()) * part2_ )
	#
	# 	M = 1.0 / obj.mass * ti.math.eye(dim ** 2)
	# 	df_dx = -(part1 + part2 + part3) * V
	# 	df_dx = (delta_time ** 2) * M @ df_dx
	# 	A0 = mat(0)
	# 	# for i in ti.static(range(dim)):
	# 	# 	for j in ti.static(range(dim)):
	# 			# rows = element.vertex_indices[i+1]
	# 			# cols = element.vertex_indices[j+1]
	# 			# ret[element.vertex_indices[i+1]:element.vertex_indices[i+1]+dim, element.vertex_indices[j+1]:element.vertex_indices[j+1]+dim] += df_dx[i*dim:i*dim+dim, j*dim:j*dim+dim]
	# 			# obj.matrix_A[None][element.vertex_indices[i+1]:element.vertex_indices[i+1]+dim, element.vertex_indices[j+1]:element.vertex_indices[j+1]+dim] += df_dx[i*dim:i*dim+dim, j*dim:j*dim+dim]
	# 		# A = mat(0)
	# 		# index = element.vertex_indices[i+1]
	# 		# A = df_dx[i*dim:i*dim+dim, i*dim:i*dim+dim]
	# 		# obj.particles.implicit_A[index] += A
	# 		# A0 -= A
	# 		# print(A)
	# 	obj.particles.implicit_A[element.vertex_indices[0]] += A0
	# 	# print(A0[0:dim, 0:dim], A0[dim:dim+dim, dim:dim+dim])
	# 	# print('A00 {}'.format(A00))
	# 	# # print()
	# 	# print('\n')
	#
	# 	# force = mu * F @ R_inv.transpose() + (- mu * X_inv).transpose() + (s_lambda * ti.log(F.determinant()) * X_inv).transpose()
	# 	force = (mu * F - mu * F_inv.transpose() + s_lambda / 2 * ti.log(
	# 		(F.transpose() @ F).determinant()) * F_inv.transpose()) @ R_inv.transpose()
	# 	force *= V
	# 	# print(mu * F @ R_inv.transpose() + (- mu * X_inv).transpose())
	#
	# 	# A = ti.math.eye(dim**2) - (delta_time **2) * M @ df_dx
	# 	# v = ti.types.vector(dim**2, ti.f32)(0)
	# 	# f = ti.types.vector(dim**2, ti.f32)(0)
	# 	f0 = vec(0.0)
	# 	for i in ti.static(range(dim)):
	# 		f = vec(force[:, i])
	# 		f0 -= f
	# 		index = element.vertex_indices[i+1]
	# 		obj.particles.implicit_b[index] += f
	# 	obj.particles.implicit_b[element.vertex_indices[0]] += f0
		# 	v[i*dim:(i+1)*dim] = obj.particles.vel[element.vertex_indices[i+1]]
		# 	f[i*dim:(i+1)*dim] = vec(force[:, i])

		# b = v - delta_time * M@f
		# if ti.math.isnan(b[0]):
		# 	print('Error: x_inv{}'.format(X_inv))
		# 	print('Error: F det {}'.format(F.determinant()))
		# x = jacobi_iter(A, b)
		# x = ti.math.inverse(A) @ b
		# v0 = vec(0)
		# # print(force)
		# for i in ti.static(range(dim)):
		# 	v_ = x[i*dim:i*dim+dim]
		# 	# v0 -= (v_ - obj.particles.vel[element.vertex_indices[i+1]])# * obj.mass
		# 	v0 -= v_ #- obj.particles.vel[element.vertex_indices[i+1]])# * obj.mass
		# 	# print('index {}, origin {}, v_ {}'.format(element.vertex_indices[i + 1],
		# 	# 										  obj.particles.vel_next[element.vertex_indices[i + 1]], v_))
		# 	obj.particles.vel_next[element.vertex_indices[i+1]] = v_
		# # 	print("index {}, v_ {}, new v {}".format(element.vertex_indices[i+1], v_, obj.particles.vel[element.vertex_indices[i+1]]))
		# # print('\n')
		# obj.particles.vel_next[element.vertex_indices[0]] = v0
	# for i in range(obj.particle_cnt):
	# 	for j in ti.static(range(dim)):
	# 		obj.particles.vel[i][j] = v[i*dim+j]
	for i in range(obj.particle_cnt):
		obj.particles.vel[i] = obj.vec_x[i]
	# return b

@ti.kernel
def update_vel(obj: ti.template(), v: ti.template()):
	obj.particles.vel[0] = v[0:dim]
	obj.particles.vel[1] = v[dim:2*dim]
	obj.particles.vel[2] = v[2*dim:3*dim]
	# for index in range(obj.particle_cnt):
	# 	# A = ti.math.eye(dim) - obj.particles.implicit_A[index]
	# 	# b = obj.particles.vel[index] - delta_time * obj.particles.implicit_b[index] / obj.mass
	# 	# x = b
	# 	# x = ti.math.inverse(A) @ b
	# 	# x = jacobi_iter(A, b)
	# 	obj.particles.vel[index] = x
	# 	obj.particles.implicit_A[index] = mat(0.0)
	# 	obj.particles.implicit_b[index] = vec(0.0)
		# obj.particles.vel_next[index] = vec(0.0)

# @ti.kernel
# def solve():
# 	for i in range(element_cnt):
# 		jacobi_iter(i)

@ti.func
def jacobi_iter_field(obj: ti.template()):
	iter_cnt = 0

	# Set init point
	for i in range(obj.particle_cnt):
		obj.vec_x[i] = obj.vec_b[i]

	err = compute_error(obj)
	threshold = 1e-5
	max_iter = 2000
	print('jacobi error first {}'.format(err))
	while err > threshold and iter_cnt < max_iter:
		jacobi_iter_field_once(obj)
		err = compute_error(obj)
		iter_cnt += 1
	print('jacobi field iter cnt: {}, loss {}'.format(iter_cnt, err))


@ti.func
def compute_error(obj: ti.template()):
	err = 0.0
	for i in range(obj.particle_cnt):
		v = vec(0.0)
		for j in range(obj.particle_cnt):
			v += obj.matrix_A[i, j] @ obj.vec_x[j]

		err += (obj.vec_b[i] - v).norm() ** 2
	err = ti.sqrt(err)
	return err


@ti.func
def jacobi_iter_field_once(obj: ti.template()):
	for i in range(obj.particle_cnt):
		b = obj.vec_b[i]
		for j in range(obj.particle_cnt):
			dF_dxij = obj.matrix_A[i, j]
			b -= dF_dxij@obj.vec_x[j]
		for k in ti.static(range(dim)):
			a_ii = obj.matrix_A[i, i][k, k]
			if ti.abs(a_ii) < 1e-6:
				obj.vec_x[i][k] = 0.0
			else:
				b[k] += a_ii * obj.vec_x[i][k]
				obj.vec_x[i][k] = b[k] / a_ii

@ti.func
def jacobi_iter(A: ti.template(), b:ti.template()):
	iter_cnt = 0
	x = b
	err = (A@x - b).norm()
	threshold = 1e-5
	max_iter = 20
	# last_err = 0
	# ti.loop_config(serialize=True)
	print('normal error first {}'.format(err))
	while err > threshold and iter_cnt < max_iter:
		for i in ti.static(range(dim*4)):
			if ti.abs(A[i, i]) < 1e-6:
				# print('i {} is {}, index {}'.format(i, A[i, i], index))
				print('aaaaaaaaaaaaaaaaaaaa')
				x[i] = 0.0
			else:
				# continue
				# print('bbbbbbbbbbbbbbbbbbbbbbbbb', A[i, i])
				y = x[i]
				x[i] = (b[i] - A[i,:]@x + A[i, i]*x[i]) / A[i, i]
				if ti.math.isnan(x[i]):
					# print('Fxxxk: ',b[i], A[i,:]@x, i, A[i, i]*y, A[i, i])
					print(A[i, :])
					print('\n')
		# 		if index == 1:
		# 			print('down, {}, {}, {}, {}'.format(i, (b[i] - A[i,:]@x + A[i, i]*x[i]), A[i, i], x[i]))
		# if index == 1:
		# 	print('x is {}'.format(x))
		# last_err = err
		err = (A@x - b).norm()
		iter_cnt += 1
		# if ti.abs(last_err - err) < threshold:
		#     break
		# if index == 1:
		# print('iter cnt: {}, loss {}, x {}'.format(iter_cnt, err, x))
		# print(b)
		print('iter cnt: {}, loss {}'.format(iter_cnt, err))

	# print('iter cnt: {}, loss {}, x {}'.format(iter_cnt, err, x))
	print('iter cnt: {}, loss {}'.format(iter_cnt, err))
	return x

	# elements[index].x = x
	# p0, p1, p2, p3 = element.vertex_indices
	# # v1, v2, v3 = x[0:3], x[3:6], x[6:9]
	# v1 = vec(x[0], x[3], x[6])
	# v2 = vec(x[1], x[4], x[7])
	# v3 = vec(x[2], x[5], x[8])
	# # f1 = (v1 - particles.vel_f[p1]) * mass
	# # f2 = (v2 - particles.vel_f[p2]) * mass
	# # f3 = (v3 - particles.vel_f[p3]) * mass
	# # f0 = -f1-f2-f3
	# v0 = -v1-v2-v3
	# particles.vel[p0] = v0
	# particles.vel[p1] = v1
	# particles.vel[p2] = v2
	# particles.vel[p3] = v3
	# print('loss {}, iter {}'.format(err, iter_cnt))


@ti.kernel
def advect_implicit(obj: ti.template()):
	for index in range(obj.particle_cnt):
		# particles[index].vel +=
		obj.particles[index].vel_g += 9.8 * ti.Vector(g_dir) * delta_time
		obj.particles[index].vel *= ti.exp(-delta_time * damping)
		obj.particles[index].vel_g *= ti.exp(-delta_time * damping)
		v = obj.particles[index].vel + obj.particles[index].vel_g

		for i in ti.static(range(dim)):
			if obj.particles.pos[index][i] < 0 and v[i] < 0:
				obj.particles.vel[index][i] = 0.0
				obj.particles.vel_g[index][i] = 0.0
				v[i] = 0.0

			if obj.particles.pos[index][i] > 1 and v[i] > 0:
				obj.particles.vel[index][i] = 0.0
				obj.particles.vel_g[index][i] = 0.0
				v[i] = 0.0

		if (obj.particles[index].pos - ti.Vector(block_center)).norm() < block_radius and v.dot(
				ti.Vector(block_center) - obj.particles[index].pos) > 0:
			disp = obj.particles[index].pos - ti.Vector(block_center)
			v -= v.dot(disp) * disp / disp.norm_sqr()

		obj.particles.pos[index] += v * delta_time
		# obj.particles.vel[index] = obj.particles[index].vel