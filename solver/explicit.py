# coding=utf-8

import utils
import taichi as ti
from main import element_cnt, dim, mat, mu, s_lambda, vec

@ti.kernel
def neo_hookean_1_grad(particles: ti.template(), elements: ti.template()):
	for i in range(element_cnt):
		element = elements[i]
		p_0 = particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = particles.pos[element.vertex_indices[j + 1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		F = X @ R_inv
		V = utils.compute_volume(X)
		R_inv_F_inv = R_inv @ ti.math.inverse(F)
		force = mu * F @ R_inv.transpose() + (- mu * R_inv_F_inv).transpose() + (s_lambda * ti.log(F.determinant()) * R_inv_F_inv).transpose()
		force *= V

		log_J_i = ti.log(F.determinant())
		phi_i = mu / 2 * ((F.transpose() @ F).trace() - dim)
		phi_i -= mu * log_J_i
		phi_i += s_lambda / 2 * log_J_i ** 2

		# factor = 1 / math.factorial(dim)
		# factor = 1 / 6
		#
		# p10, p20, p30 = vec(X[:, 0]), vec(X[:, 1]), vec(X[:, 2])
		# V_t = p10.dot(p20.cross(p30))
		# if V_t < 0:
		#     n = p20.cross(p30).normalized()
		#     l = p10.dot(n)
		#     p10 = 2 * l * n - p10

		f1 = vec(force[:, 0])# + factor * p20.cross(p30) * phi_i
		f2 = vec(force[:, 1])# + factor * p10.cross(p30) * phi_i
		f3 = vec(force[:, 2])# + factor * p10.cross(p20) * phi_i
		f0 = -f1 - f2 - f3
		p0, p1, p2, p3 = element.vertex_indices
		particles.force[p0] += f0
		particles.force[p1] += f1
		particles.force[p2] += f2
		particles.force[p3] += f3
		# print(f1, particles[mesh.p1].pos, particles[mesh.p1].force)
		# if ti.math.isnan(f1[0]):
		#     print(F, S)