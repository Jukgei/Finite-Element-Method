# coding=utf-8
import numpy as np
import taichi as ti
from constants import delta_time, dim, mat, vec, g_dir


@ti.func
def kronecker_product(a: ti.template(), b: ti.template()):
	ret = ti.types.matrix(a.n*b.n, a.m*b.m, ti.f32)(0)
	for i in ti.static(range(a.n)):
		for j in ti.static(range(a.m)):
			ret[i*b.n:i*b.n+b.n, j*b.m:j*b.m+b.m] = a[i, j] * b
	return ret


@ti.func
def check_symmetry(obj: ti.template()):
	ret = 1
	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			mat1 = obj.matrix_A[i, j]
			mat2 = obj.matrix_A[j, i]
			for k in ti.static(range(dim)):
				for l in ti.static(range(dim)):
					if ti.abs(mat1[k, l] - mat2[l, k]) > 1e-7:
						print('Value {}'.format(ti.abs(mat1[k, l] - mat2[l, k])))
						ret = 0
	return ret


@ti.kernel
def update_debug_a(obj:ti.template()) -> ti.types.matrix(6, 6, ti.f32):
	m = ti.types.matrix(obj.particle_cnt * dim , obj.particle_cnt * dim, ti.f32)(0)
	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			mat = obj.matrix_A[i, j]
			for k in ti.static(range(dim)):
				for l in ti.static(range(dim)):
					m[i*dim+k, j*dim+l] = mat[k, l]
	# 				print(m)# = print(mat[k, l])
	return m

@ti.func
def check_diagonally_dominant(obj: ti.template()):
	ret = 1
	for i in range(obj.particle_cnt):
		for l in ti.static(range(dim)):
			diag_element = 0.0
			other_element = 0.0
			for j in range(obj.particle_cnt):
				for k in ti.static(range(dim)):
						mat = obj.matrix_A[i, j]
						if i == j and k == l:
							diag_element += ti.abs(mat[l, k])
						else:
							other_element += ti.abs(mat[l, k])
			if diag_element < other_element:
				ret = 0
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

		force = (obj.mu * F - obj.mu * F_inv.transpose() + obj.s_lambda / 2 * ti.log((F.transpose() @ F).determinant()) * F_inv.transpose()) @ R_inv.transpose()
		force *= -V

		# f = ti.types.vector(dim*obj.particle_cnt, ti.f32)(0.0)

		f0 = vec(0.0)
		# m_ = 1.0 / obj.mass
		for i in ti.static(range(dim)):
			f_n = vec(force[:, i])
			f0 -= f_n
			particle_index = element.vertex_indices[i+1]
			m_ = 1.0 / obj.particles.mass[particle_index]
			obj.vec_b[particle_index] += delta_time * m_ * f_n
			# for j in ti.static(range(dim)):
			# 	f[particle_index*dim+j] = f_n[j]

		particle_index = element.vertex_indices.x
		m_ = 1.0 / obj.particles.mass[particle_index]
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
				rows = element.vertex_indices[i + 1]
				cols = element.vertex_indices[j + 1]
				# mass = obj.particles.mass[cols]
				dF = mat(0.0)
				if i == j:
					dF = ti.math.eye(dim)
				dF = dF@R_inv
				dF_T = dF.transpose()
				dF_dxij = obj.mu * dF + (obj.mu - obj.s_lambda * log_J) * F_inv_T @ dF_T @ F_inv_T + obj.s_lambda * (F_inv @ dF).trace() * F_inv_T
				dF_dxij = -V * dF_dxij @ R_inv.transpose()
				# dF_dxij = (delta_time ** 2) * (1.0 / obj.mass) * dF_dxij
				dF_dx00 += dF_dxij

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
			# m_ = obj.mass
			m_ = obj.particles.mass[i]
			obj.matrix_A[i, j] = (delta_time ** 2) * ((1.0 / m_) * ti.math.eye(dim))@obj.matrix_A[i, j]

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

	obj.vec_d.fill(vec(0.0))
	obj.vec_ATb.fill(vec(0.0))
	obj.matrix_ATA.fill(mat(0.0))
	obj.matrix_AT.fill(mat(0.0))

	compute_linear_system_matrix_a(obj)
	compute_linear_system_vector_b(obj)
	# jacobi_iter_field(obj)
	cg(obj)
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

	# symmetry = check_symmetry(obj)
	#
	# if symmetry == 1:
	# 	print('Fine!')
	# else:
	# 	print('Not a symmetry matrix')

	# convergent = check_diagonally_dominant(obj)
	# if convergent == 1:
	# 	print('convergent!')
	# else:
	# 	print('divergent!')
	print('jacobi error first {}'.format(err))
	while err > threshold and iter_cnt < max_iter:# and convergent == 1:

		jacobi_iter_field_once(obj)
		err = compute_error(obj)
		iter_cnt += 1
		if err >= p_err:
			recover_past_frame_x(obj)
			break
		p_err = err
		cache_x(obj)
	print('jacobi field iter cnt: {}, loss {}'.format(iter_cnt, err))


@ti.func
def cache_x(obj: ti.template()):
	for i in range(obj.particle_cnt):
		obj.past_vec_x[i] = obj.vec_x[i]


@ti.func
def recover_past_frame_x(obj: ti.template()):
	for i in range(obj.particle_cnt):
		obj.vec_x[i] = obj.past_vec_x[i]

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

# conjugate gradient in taichi field
@ti.func
def cg(obj: ti.template()):

	# for i in range(obj.particle_cnt):
	# 	obj.vec_x[i] = obj.vec_b[i]

	iter_cnt = 0
	r = ti.types.vector(obj.particle_cnt * dim, ti.f32)(0.0)
	# d = ti.types.vector(obj.particle_cnt * dim, ti.f32)(0.0)
	q = ti.types.vector(obj.particle_cnt * dim, ti.f32)(0.0)

	# Pre-conditioner
	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			obj.matrix_AT[i, j] = obj.matrix_A[j, i].transpose()
	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			obj.vec_ATb[i] += obj.matrix_AT[i, j] @ obj.vec_b[j]
	for i in range(obj.particle_cnt):
		for j in range(obj.particle_cnt):
			for k in range(obj.particle_cnt):
				obj.matrix_ATA[i, j] += obj.matrix_AT[i, k] @ obj.matrix_A[k, j]

	# Non-pre conditioner
	# for i in range(obj.particle_cnt):
	# 	obj.vec_ATb[i] = obj.vec_b[i]
	# for i in range(obj.particle_cnt):
	# 	for j in range(obj.particle_cnt):
	# 		obj.matrix_ATA[i, j] = obj.matrix_A[i, j]

	# r = b - A@x
	for i in range(obj.particle_cnt):
		ax_ij = vec(0.0)
		for j in range(obj.particle_cnt):
			ax_ij += obj.matrix_ATA[i, j] @ obj.vec_x[j]
		for k in range(dim):

			r[i*dim+k] = obj.vec_ATb[i][k] - ax_ij[k]


	for i in range(obj.particle_cnt):
		for j in range(dim):
			obj.vec_d[i][j] = r[i*dim+j]

	d = r
	delta_new = r @ r
	delta_0 = delta_new
	iter_max = 50000
	epsilon = 5e-3
	print('first error is {}'.format(delta_new))
	# while iter_cnt < iter_max and delta_new > delta_0 * epsilon**2:
	while iter_cnt < iter_max and delta_new > 1e-5:
		# q = A @ d
		for i in range(obj.particle_cnt):
			ad_ij = vec(0.0)
			for j in range(obj.particle_cnt):
				ad_ij += obj.matrix_ATA[i, j] @ obj.vec_d[j]
			for k in range(dim):
				q[i*dim+k] = ad_ij[k]

		alpha = delta_new / (d@q)

		# x = x + alpha * d
		for i in range(obj.particle_cnt):
			for j in range(dim):
				obj.vec_x[i][j] = obj.vec_x[i][j] + alpha * d[i*dim + j]

		# if iter_cnt % 50 == 0:
		# 	# r = b - A@x
		# 	for i in range(obj.particle_cnt):
		# 		ax_ij = vec(0.0)
		# 		for j in range(obj.particle_cnt):
		# 			ax_ij += obj.matrix_ATA[i, j] @ obj.vec_x[j]
		# 		for k in range(dim):
		# 			r[i * dim + k] = obj.vec_ATb[i][k] - ax_ij[k]
		#
		# else:
			# r = r - alpha * q
		r = r - alpha * q
			# for i in range(obj.particle_cnt):
			# 	for j in range(dim):
			# 		r[i*dim + j] = r[i*dim + j] - alpha * q[i*dim+j]


		delta_old = delta_new
		delta_new = r @ r
		beta = delta_new / delta_old
		d = r + beta * d
		for i in range(obj.particle_cnt):
			for j in range(dim):
				obj.vec_d[i][j] = d[i * dim + j]
		iter_cnt = iter_cnt + 1
		# if iter_cnt > 20:
		# 	print('OK!', delta_new, iter_cnt)
	print('OK!', ti.sqrt(delta_new), iter_cnt, compute_error(obj))

@ti.func
def jacobi_iter_field_once(obj: ti.template()):
	omega = 0.75
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
				obj.vec_x[i][k] = omega * b[k] / a_ii + (1 - omega) * obj.past_vec_x[i][k]

@ti.kernel
def advect_implicit(obj: ti.template(), circle_blocks: ti.template()):
	for index in range(obj.particle_cnt):
		obj.particles[index].vel_g += 9.8 * ti.Vector(g_dir) * delta_time
		obj.particles[index].vel *= ti.exp(-delta_time * obj.damping)
		obj.particles[index].vel_g *= ti.exp(-delta_time * obj.damping)
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
		for i in range(circle_blocks.blocks_count):
			block = circle_blocks.blocks[i]
			if block.radius <= 0.0:
				continue
			center = block.center
			radius = block.radius
			if (obj.particles[index].pos - center).norm() < radius and v.dot(
					center - obj.particles[index].pos) > 0:
				disp = obj.particles[index].pos - center
				v -= v.dot(disp) * disp / disp.norm_sqr()
				obj.particles.vel[index] -= obj.particles.vel[index].dot(disp) * disp / disp.norm_sqr()
				obj.particles.vel_g[index] -= obj.particles.vel_g[index].dot(disp) * disp / disp.norm_sqr()

		obj.particles.pos[index] += v * delta_time
		# obj.particles.vel[index] = obj.particles[index].vel


def gen_random_matrix(n):
	np.random.seed(42)
	return np.random.rand(n, n)

def gen_n_dim_positive_diag_matrix(n):
	np.random.seed(42)

	m = np.zeros((n, n))
	for i in range(n):
		m[i, :i+1] = np.random.rand(i+1)

	return m@m.transpose()

def gen_n_dim_positive_matrix(n):
	# np.random.seed(42)
	A = np.random.rand(n, n)

	n = A.shape[0]
	leading_minors = []
	for k in range(1, n + 1):
		sub_matrix = A[:k, :k]
		minor = np.linalg.det(sub_matrix)
		leading_minors.append(minor)
	while not all(num > 0 for num in leading_minors):
		A = np.random.rand(n, n)
		leading_minors = []
		for k in range(1, n + 1):
			sub_matrix = A[:k, :k]
			minor = np.linalg.det(sub_matrix)
			leading_minors.append(minor)
	print(leading_minors)
	return A

def gen_n_dim_b(n):
	return np.random.rand(n)# * 2 - 1

def steepest_descent_np(A, b):
	print('\n')
	print('steepest descent ===============================')
	i = 0
	rows, _ = A.shape
	x = np.zeros(rows)
	r = b - A@x
	delta = r@r
	delta_0 = delta

	i_max = 1000
	epsilon = 1e-4
	while i < i_max and delta > 1e-4:
		q = A@r
		alpha = delta / (r@q)
		x = x + alpha * r

		if i % 50 == 0:
			r = b - A@x
		else:
			r = r - alpha*q

		delta = r @ r
		i = i+1
	print('x is ', x)
	print('iter count is ', i)
	print('loss is ', np.linalg.norm(A @ x - b))


def conjugate_gradient_np(A, b):
	print('Input: A is ', A)
	print('Input: b is ', b)

	i = 0
	rows, _ = A.shape
	x = np.zeros(rows)
	r = b - A @ x
	d = r
	delta_new = r.transpose() @ r
	delta_0 = delta_new
	i_max = 300000
	epsilon = 1e-4
	while i < i_max and delta_new > delta_0 * epsilon ** 2:
		q = A @ d
		alpha = delta_new / (d.transpose() @ q)
		x = x + alpha * d
		if i % 10 == 0:
			r = b - A@x
		else:
			r = r - alpha * q
		delta_old = delta_new
		delta_new = r.transpose() @ r
		beta = delta_new / delta_old
		d = r + beta * d
		i = i + 1
		# print('loss is', delta_new)
	print('x is ', x)
	print('iter count is ', i)
	print('loss is ', np.linalg.norm(A@x - b))
	return x