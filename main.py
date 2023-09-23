# coding=utf-8

import taichi as ti
import numpy as np
ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

width = 640
height = 640

block_radius = 0.32

# mu = 16666
# s_lambda = 11111.0
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, s_lambda = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
damping = 14.5
center = ti.Vector([0.72, 0.8])
# center = ti.Vector([0.55, 0.3])
v_refect = -0.3
delta_time = 5e-5
side_length = 0.2  #
subdivisions = 8  #
A = ((side_length / subdivisions) ** 2) / 2
rho = 1000
mass = rho * A

auto_diff = True

x = np.linspace(0, side_length, subdivisions + 1)  #
y = np.linspace(0, side_length, subdivisions + 1)  #
vertices = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  #
ti_vertices = ti.Vector.field(2, ti.f32, shape=vertices.shape[0])
ti_vertices.from_numpy(vertices)

faces = []
for i in range(subdivisions):
    for j in range(subdivisions):
        p1 = i * (subdivisions + 1) + j
        p2 = p1 + 1
        p3 = p1 + subdivisions + 1
        p4 = p3 + 1
        faces.append([p1, p2, p4])
        faces.append([p1, p4, p3])
faces = np.array(faces)
ti_faces = ti.Vector.field(3, ti.i32, shape=faces.shape[0])
ti_faces.from_numpy(faces)
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

Particle = ti.types.struct(
    pos=ti.math.vec2,
    vel=ti.math.vec2,
    acc=ti.math.vec2,
    mass=ti.f32,
    force=ti.math.vec2,
    ref_pos=ti.math.vec2
)

Mesh = ti.types.struct(
    p0=ti.i32,
    p1=ti.i32,
    p2=ti.i32
)

particle_cnt = vertices.shape[0]
mesh_cnt = faces.shape[0]
meshs = Mesh.field(shape=mesh_cnt)
particles = Particle.field(shape=particle_cnt, needs_grad=True)
pos1 = ti.Vector.field(2, ti.f32, shape=particle_cnt, needs_grad=True)

@ti.kernel
def compute_energy():
    for i in range(mesh_cnt):
        mesh = meshs[i]
        p0 = particles[mesh.p0]
        p1 = particles[mesh.p1]
        p2 = particles[mesh.p2]
        pos = particles[mesh.p0].pos
        ref_pos = p0.ref_pos
        x1_pos = particles[mesh.p1].pos
        x1_ref_pos = p1.ref_pos
        x2_pos = particles[mesh.p2].pos
        x2_ref_pos = p2.ref_pos
        x10 = x1_pos - pos
        x20 = x2_pos - pos
        r10 = x1_ref_pos - ref_pos
        r20 = x2_ref_pos - ref_pos
        I = ti.math.mat2([1, 0, 0, 1])
        X = ti.math.mat2([x10.x, x20.x, x10.y, x20.y])
        R = ti.math.mat2([r10.x, r20.x, r10.y, r20.y])
        R_inv = ti.math.inverse(R)
        F = X @ R_inv
        # G = 0.5 * (F.transpose() @ F - I)
        S = ti.abs(x10.cross(x20))
        # K = G.transpose() @ G
        # U[None] += S * (0.5 * s_lambda * G.trace() **2 + mu * K.trace())

        # Neo-Hookean
        F_i = F
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += s_lambda / 2 * log_J_i ** 2
        # phi[i] = phi_i
        U[None] += S * phi_i


@ti.kernel
def kinematic_mesh():
    for i in range(particle_cnt):
        kinematic(i, delta_time)

@ti.func
def kinematic(index: ti.int32, dt: ti.f32):
    if auto_diff:
        particles[index].vel += (particles[index].acc + 9.8 * ti.Vector([0, -1]) - particles.pos.grad[index] /
                                 particles[index].mass) * dt
    else:
        particles[index].vel += (particles[index].acc + 9.8 * ti.Vector([0, -1]) + particles[index].force / particles[index].mass) * dt
    #

    particles[index].vel *= ti.exp(-dt * damping)

    if particles[index].pos.x > 1 and particles[index].vel.x > 0:
        # particles[index].pos.x = 1
        particles[index].vel.x = 0

    if particles[index].pos.y > 1 and particles[index].vel.y > 0:
        # particles[index].pos.y = 1
        particles[index].vel.y = 0

    if particles[index].pos.x < 0 and particles[index].vel.x < 0:
        # particles[index].pos.x = 0
        particles[index].vel.x = 0

    if particles[index].pos.y < 0 and particles[index].vel.y < 0:
        # particles[index].pos.y = 0
        particles[index].vel.y = 0

    if (particles[index].pos - ti.Vector([0.5, 0.5])).norm() < block_radius and particles[index].vel.dot(ti.Vector([0.5, 0.5]) - particles[index].pos) > 0:
        # pre_pos = particles[index].pos - particles[index].vel * dt
        # dir = ti.math.normalize(particles[index].vel * dt)
        # line_to_circle = ti.Vector([0.5, 0.5]) - pre_pos
        # projection_length = line_to_circle.dot(dir)
        # perpendicular_length = ti.math.sqrt(line_to_circle.norm() ** 2 - projection_length ** 2)
        # intersection_distance = projection_length - ti.math.sqrt(block_radius ** 2 - perpendicular_length ** 2 )
        # intersection_point = pre_pos + dir * intersection_distance
        # particles[index].pos = intersection_point
        #
        # n = ti.math.normalize(intersection_point - ti.Vector([0.5, 0.5]))
        # v_norm = particles[index].vel.norm() * (-v_refect)
        # particles[index].vel = (-dir.dot(n) * 2 * n + dir) * v_norm

        disp = particles[index].pos - ti.Vector([0.5, 0.5])
        particles[index].vel -= particles[index].vel.dot(disp) * disp / disp.norm_sqr()
        # particles[index].vel =ti.Vector([-n.y, n.x]) * v_norm

    particles.pos[index] += particles.vel[index] * dt
    if not auto_diff:
        particles[index].force = ti.math.vec2([0.0, 0.0])
    else:
        pos1[index] = particles[index].pos

@ti.kernel
def particles_init():

    for i in range(particle_cnt):
        particles[i].pos = ti_vertices[i] + center
        particles[i].ref_pos = particles[i].pos
        particles[i].mass = mass
        pos1[i] = particles[i].pos

@ti.kernel
def mesh_init():
    for i in range(mesh_cnt):
        meshs[i].p0 = ti_faces[i].x
        meshs[i].p1 = ti_faces[i].y
        meshs[i].p2 = ti_faces[i].z


@ti.kernel
def fem():
    for i in range(mesh_cnt):
        mesh = meshs[i]
        p0 = particles[mesh.p0]
        p1 = particles[mesh.p1]
        p2 = particles[mesh.p2]
        pos = particles[mesh.p0].pos
        ref_pos = p0.ref_pos
        x1_pos = particles[mesh.p1].pos
        x1_ref_pos = p1.ref_pos
        x2_pos = particles[mesh.p2].pos
        x2_ref_pos = p2.ref_pos
        x10 = x1_pos - pos
        x20 = x2_pos - pos
        r10 = x1_ref_pos - ref_pos
        r20 = x2_ref_pos - ref_pos
        I = ti.math.mat2([1, 0, 0, 1])
        X = ti.math.mat2([x10.x, x20.x, x10.y, x20.y])
        R = ti.math.mat2([r10.x, r20.x, r10.y, r20.y])
        R_inv = ti.math.inverse(R)
        F = X @ R_inv
        G = 0.5 * (F.transpose() @ F - I)
        S = 2 * mu * G + s_lambda * G.trace() * I
        force = - A * F @ S @ R_inv.transpose()
        f1 = ti.math.vec2(force[0, 0], force[1, 0])
        f2 = ti.math.vec2(force[0, 1], force[1, 1])
        f0 = - f1 - f2
        particles[mesh.p0].force += f0
        particles[mesh.p1].force += f1
        particles[mesh.p2].force += f2

        # print(f1, particles[mesh.p1].pos, particles[mesh.p1].force)
        # if ti.math.isnan(f1[0]):
        #     print(F, S)

if __name__ == '__main__':
    gui = ti.GUI('Finite Element Method', (width, height))
    particles_init()
    mesh_init()
    frame_cnt = 0
    while gui.running:
        frame_cnt+=1
        # if frame_cnt == 120:
        #     particles[2].pos = center + ti.Vector([0.1, 0.1])
        for i in range(10):
            if not auto_diff:
                fem()
            else:
                with ti.ad.Tape(loss=U):
                    compute_energy()
            kinematic_mesh()
            U[None] = 0
        # for i in range(particle_cnt):
        #     begin = particles[i].pos
        #     end = particles[(i+1)%particle_cnt].pos
        #     gui.line(begin, end, radius=1, color=0xFF0000)
        # for i in range(mesh_cnt):
        #     p0 = particles[meshs[i].p0].pos
        #     p1 = particles[meshs[i].p1].pos
        #     p2 = particles[meshs[i].p2].pos
        #     gui.line(p0, p1, radius=1, color=0xFF0000)
        #     gui.line(p1, p2, radius=1, color=0xFF0000)
        #     gui.line(p2, p0, radius=1, color=0xFF0000)

        gui.circles(particles.pos.to_numpy(), radius=2, color=0x00FF00)
        gui.circle([0.5, 0.5], color=0xFF0000, radius=block_radius * width)
        gui.show()
