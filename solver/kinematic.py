import taichi as ti
from main import delta_time, auto_diff, damping, dim, g_dir
from main import block_radius, block_center, vec


@ti.kernel
def kinematic(obj: ti.template()):
	for i in range(obj.particle_cnt):
		kinematic_particle(i, delta_time, obj.particles)

@ti.func
def kinematic_particle(index: ti.int32, dt: ti.f32, particles:ti.template()):
	if auto_diff:
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

	if (particles[index].pos - ti.Vector(block_center)).norm() < block_radius and particles[index].vel.dot(
			ti.Vector(block_center) - particles[index].pos) > 0:
		disp = particles[index].pos - ti.Vector(block_center)
		particles[index].vel -= particles[index].vel.dot(disp) * disp / disp.norm_sqr()

	particles.pos[index] += particles.vel[index] * dt
	if not auto_diff:
		particles[index].force = vec(0.0)
