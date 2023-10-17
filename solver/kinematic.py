# coding=utf-8

import taichi as ti
from constants import delta_time
from constants import vec, use_explicit_method, g_dir, dim, auto_diff


@ti.kernel
def kinematic(obj: ti.template(), circle_blocks: ti.template()):
	for i in range(obj.particle_cnt):
		kinematic_particle(i, delta_time, obj.particles, obj.damping, circle_blocks)

@ti.func
def kinematic_particle(index: ti.int32, dt: ti.f32, particles: ti.template(), damping: ti.f32, circle_blocks: ti.template()):
	if auto_diff == 1:
		particles[index].vel += (particles[index].acc + 9.8 * ti.Vector(g_dir) - particles.pos.grad[index] /
								 particles[index].mass) * dt
	else:
		particles[index].vel += (particles[index].acc + 9.8 * ti.Vector(g_dir) - particles[index].force / particles[
			index].mass) * dt


	particles[index].vel *= ti.exp(-dt * damping)

	for i in ti.static(range(dim)):
		if particles.pos[index][i] < 0 and particles.vel[index][i] < 0:
			particles.vel[index][i] = 0

		if particles.pos[index][i] > 1 and particles.vel[index][i] > 0:
			particles.vel[index][i] = 0

	for i in range(circle_blocks.blocks_count):
		block = circle_blocks.blocks[i]
		if block.radius <= 0.0:
			continue
		center = block.center
		radius = block.radius
		if (particles[index].pos - center).norm() < radius and particles[index].vel.dot(
				center - particles[index].pos) > 0:
			disp = particles[index].pos - center
			particles[index].vel -= particles[index].vel.dot(disp) * disp / disp.norm_sqr()

	particles.pos[index] += particles.vel[index] * dt
	if auto_diff == 0:
		particles[index].force = vec(0.0)
