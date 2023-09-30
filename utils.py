# coding=utf-8

import sympy as sp
import taichi as ti
from main import dim, mat
from sympy import symbols, Matrix, diff, MatrixSymbol, Inverse
from sympy import det, log, Trace, Transpose

def neo_hookean_3d():
	
	x0, y0, z0 = symbols('x0 y0 z0')
	x1, y1, z1 = symbols('x1 y1 z1')
	x2, y2, z2 = symbols('x2 y2 z2')
	x3, y3, z3 = symbols('x3 y3 z3')
	mu, lam = symbols('mu lam')
	dim = symbols('dim')

	p0 = Matrix([[x0], [y0], [z0]])
	p1 = Matrix([[x1], [y1], [z1]])
	p2 = Matrix([[x2], [y2], [z2]])
	p3 = Matrix([[x3], [y3], [z3]])
	p10 = Matrix([p1 - p0])
	p20 = Matrix([p2 - p0])
	p30 = Matrix([p3 - p0])
	F = p10.row_join(p20).row_join(p30)
	# F = Matrix([[p1 - p0], [p2 - p0], [p3 - p0]])
	J = det(F)
	S = abs(p10.dot(p20.cross(p30)))
	H = p10.cross(p30)
	U = (mu / 2 * (Trace(Transpose(F) * F) - dim) - mu * log(J) + lam / 2 * log(J) * log(J)) * S

	# F = MatrixSymbol('F', 3, 3)
	# F = (p1 - p0).row_join(p2 - p0).row_join(p3-p0)
	dU_dx = diff(U, p0)
	# K = MatrixSymbol('K', 3, 3)
	# p0 = MatrixSymbol('p0', 3, 1)
	# p1 = MatrixSymbol('p0', 3, 1)
	# p2 = MatrixSymbol('p0', 3, 1)
	# p3 = MatrixSymbol('p0', 3, 1)

	dS_dx = diff(S, p2)
	# K = Matrix.blockmatrix([[p1-p0], [p2-p0], [p3-p0]])
	# K = Matrix.hstack(p1-p0, p2-p0, p3-p0)
	# print('K ', K)
	print('Yuki: ', dU_dx.shape)
	# print('Yuki: ', dU_dx)
	# print('Yuki: ', dS_dx)
	# print('Yuki: ', dU_dx[0, 0])
	# print('Yuki: ', dU_dx[1, 0])
	# print('Yuki: ', dU_dx[2, 0])
	print('Yuki: ', dS_dx[0, 0])
	print('Yuki: ', dS_dx[1, 0])
	print('Yuki: ', dS_dx[2, 0])
	print('Yuki: ', H[0, 0])
	print('Yuki: ', H[1, 0])
	print('Yuki: ', H[2, 0])

def neo_hookean_3d_2ord():
	X = MatrixSymbol('X', 3, 3)
	X0 = MatrixSymbol('X0', 3, 3)
	R = MatrixSymbol('R', 3, 3)

	F = (X-X0) * R
	K = Transpose(R * F.inverse())
	mu, lam = symbols('mu lam')
	# print(F)
	# _1_order_diff = lam * log(det(F)) * K - mu * K
	_1_order_diff = - mu * K + lam * log(det(F)) * K + mu * F * Transpose(R)
	_2_order_diff = _1_order_diff.diff(X)
	print(_2_order_diff)
	# print(_2_order_diff.shape)

	# _2_order_diff = diff(_1_order_diff, X)
	# print(_2_order_diff)

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