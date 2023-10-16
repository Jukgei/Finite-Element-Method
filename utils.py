# coding=utf-8

import taichi as ti
from main import dim, mat


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