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
def implicit_solver_neo_hookean(obj: ti.template()):
	obj.matrix_A.fill(mat(0.0))
	obj.vec_b.fill(vec(0.0))
	obj.vec_x.fill(vec(0.0))

	compute_linear_system_matrix_a(obj)
	compute_linear_system_vector_b(obj)
	jacobi_iter_field(obj)

	for i in range(obj.particle_cnt):
		obj.particles.vel[i] = obj.vec_x[i]


@ti.func
def jacobi_iter_field(obj: ti.template()):
	iter_cnt = 0

	# Set init point
	for i in range(obj.particle_cnt):
		obj.vec_x[i] = 0.5 * obj.vec_b[i]

	err = compute_error(obj)
	p_err = err
	threshold = 1e-5
	max_iter = 20000
	print('jacobi error first {}'.format(err))
	while err > threshold and iter_cnt < max_iter:
		jacobi_iter_field_once(obj)
		err = compute_error(obj)
		iter_cnt += 1
		# if err > p_err:
		# 	break
		p_err = err
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

@ti.kernel
def advect_implicit(obj: ti.template()):
	for index in range(obj.particle_cnt):
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
				# obj.particles.vel_g[index][i] = 0.0
				v[i] = 0.0

		if (obj.particles[index].pos - ti.Vector(block_center)).norm() < block_radius and v.dot(
				ti.Vector(block_center) - obj.particles[index].pos) > 0:
			disp = obj.particles[index].pos - ti.Vector(block_center)
			v -= v.dot(disp) * disp / disp.norm_sqr()
			obj.particles.vel[index] -= obj.particles.vel[index].dot(disp) * disp / disp.norm_sqr()
			obj.particles.vel_g[index] -= obj.particles.vel_g[index].dot(disp) * disp / disp.norm_sqr()

		obj.particles.pos[index] += v * delta_time
		# obj.particles.vel[index] = obj.particles[index].vel