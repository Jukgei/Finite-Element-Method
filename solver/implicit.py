# coding=utf-8

import utils
import taichi as ti
from main import delta_time, damping, s_lambda, mu, g_dir
# from main import element_cnt, particle_cnt
from main import matA, vecb, vec
from main import dim, mat
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
	for i in ti.static(range(rows)):
		for j in ti.static(range(cols)):
			ret[i*rows:i*rows+rows, j*cols:j*cols+cols] = a[j*cols:j*cols+cols, i*rows:i*rows+rows]
	return ret

@ti.kernel
def neo_hookean_2_grad(obj: ti.template()):# -> ti.types.matrix(dim*dim, dim*dim, ti.f32):
	# ret = ti.Matrix([[0.0 for _ in ti.static(range(dim ** 2))] for __ in range(dim ** 2)])
	for i in range(obj.element_cnt):
		# t = tensor[i]
		element = obj.elements[i]
		p_0 = obj.particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = obj.particles.pos[element.vertex_indices[j + 1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		F = X @ R_inv
		X_inv = ti.math.inverse(X)
		V = utils.compute_volume(X)
		part1 = mu * kronecker_product(R_inv@R_inv.transpose(), ti.math.eye(dim))
		part2_ = kronecker_product(X_inv.transpose(), X_inv)
		part2 = mu * tensor_transpose(part2_, dim, dim)
		part3 = s_lambda * tensor_transpose((kronecker_product((R_inv @ X_inv @ R_inv).transpose(), X_inv) + ti.log(F.determinant()) * part2_),dim,dim) # WRONG

		df_dx = (part1 + part2 + part3) * V

		force = mu * F @ R_inv.transpose() + (- mu * X_inv).transpose() + (s_lambda * ti.log(F.determinant()) * X_inv).transpose()
		force *= V
		# print(mu * F @ R_inv.transpose() + (- mu * X_inv).transpose())
		M = 1.0/obj.mass #* ti.math.eye(dim**2)
		A = ti.math.eye(dim**2) - (delta_time **2) * M * df_dx
		v = ti.types.vector(dim**2, ti.f32)(0)
		f = ti.types.vector(dim**2, ti.f32)(0)
		for i in ti.static(range(dim)):
			v[i*dim:(i+1)*dim] = obj.particles.vel[element.vertex_indices[i+1]]
			f[i*dim:(i+1)*dim] = vec(force[:, i])

		b = v - delta_time * M*f
		# x = jacobi_iter(A, b)
		# print(A.determinant())
		# x = ti.math.inverse(A) @ b
		x = delta_time * M*f
		# print(A.trace(), A.sum() - A.trace())
		v0 = vec(0)
		print(force)
		for i in ti.static(range(dim)):
			v_ = x[i*dim:i*dim+dim]
			v0 -= v_
			obj.particles.vel[element.vertex_indices[i+1]] -= v_
			print("index {}, v_ {}, new v {}".format(element.vertex_indices[i+1], v_, obj.particles.vel[element.vertex_indices[i+1]]))
		print('\n')
		obj.particles.vel[element.vertex_indices[0]] -= v0

# @ti.kernel
# def solve():
# 	for i in range(element_cnt):
# 		jacobi_iter(i)


@ti.func
def jacobi_iter(A: ti.template(), b:ti.template()):
	iter_cnt = 0
	x = ti.types.vector(b.n, ti.f32)(0)
	err = (A@x - b).norm()
	threshold = 5e-5
	# last_err = 0
	# ti.loop_config(serialize=True)
	while err > threshold:
		for i in ti.static(range(dim**2)):
			if ti.abs(A[i, i]) < 1e-7:
				# print('i {} is {}, index {}'.format(i, A[i, i], index))
				# print('aaaaaaaaaaaaaaaaaaaa')
				x[i] = 0.0
			else:
				# continue
				# print('bbbbbbbbbbbbbbbbbbbbbbbbb', A[i, i])
				y = x[i]
				x[i] = (b[i] - A[i,:]@x + A[i, i]*x[i]) / A[i, i]
				if ti.math.isnan(x[i]):
					print('Fxxxk: ',b[i], A[i,:]@x, i, A[i, i]*y, A[i, i])
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
		print('iter cnt: {}, loss {}, x {}'.format(iter_cnt, err, x))
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
		v = obj.particles[index].vel + 9.8 * ti.Vector(g_dir) * delta_time
		v *= ti.exp(-delta_time * damping)

		for i in ti.static(range(dim)):
			if obj.particles.pos[index][i] < 0 and v[i] < 0:
				# obj.particles.vel[index][i] = 0
				v[i] = 0

			if obj.particles.pos[index][i] > 1 and v[i] > 0:
				# obj.particles.vel[index][i] = 0
				v[i] = 0

		if (obj.particles[index].pos - ti.Vector(block_center)).norm() < block_radius and v.dot(
				ti.Vector(block_center) - obj.particles[index].pos) > 0:
			disp = obj.particles[index].pos - ti.Vector(block_center)
			v -= v.dot(disp) * disp / disp.norm_sqr()

		obj.particles.pos[index] += v * delta_time
		obj.particles.vel[index] = v