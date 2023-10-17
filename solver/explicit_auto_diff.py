# coding=utf-8

import taichi as ti
from constants import dim, mat


@ti.kernel
def compute_energy(obj: ti.template()):
	for i in range(obj.element_cnt):
		element = obj.elements[i]
		p_0 = obj.particles[element.vertex_indices.x].pos
		X = mat(0)
		for j in ti.static(range(dim)):
			if j + 1 <= dim:
				p_j = obj.particles.pos[element.vertex_indices[j+1]]
				X[:, j] = p_j - p_0

		R_inv = element.ref
		F = X @ R_inv

		V = element.volume

		# Neo-Hookean
		F_i = F
		log_J_i = ti.log(F_i.determinant())
		phi_i = obj.mu / 2 * ((F_i.transpose() @ F_i).trace() - dim)
		phi_i -= obj.mu * log_J_i
		phi_i += obj.s_lambda / 2 * log_J_i ** 2
		obj.phi[i] = phi_i * V
		obj.U[None] += V * phi_i

		# StVK
		# I = ti.math.eye(dim)
		# G = 0.5 * (F.transpose() @ F - I)
		# phi_i = (G ** 2).sum() * mu
		# phi_i += s_lambda/2 * G.trace() **2
		# phi[i] = phi_i * V
		# U[None] += V * phi_i