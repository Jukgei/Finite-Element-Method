# coding=utf-8

import taichi as ti
import numpy as np
ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

mu = .5
s_lambda = .5
# center = ti.Vector([0.55, 0.78])
center = ti.Vector([0.55, 0.02])
mass = .001
v_refect = -0.9

side_length = 0.2  #
subdivisions = 1  #

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

Particle = ti.types.struct(
    pos=ti.math.vec2,
    vel=ti.math.vec2,
    acc=ti.math.vec2,
    mass=ti.f32,
    force=ti.math.vec2,
    ref_pos=ti.math.vec2,
    flag=ti.i32
)

Mesh = ti.types.struct(
    p0=ti.i32,
    p1=ti.i32,
    p2=ti.i32
)

particle_cnt = vertices.shape[0]
mesh_cnt = faces.shape[0]
delta_time = 1e-3
meshs = Mesh.field(shape=mesh_cnt)
particles = Particle.field(shape=particle_cnt)

@ti.kernel
def kinematic_mesh():
    for i in range(mesh_cnt):
        p0 = particles[meshs[i].p0]
        p1 = particles[meshs[i].p1]
        p2 = particles[meshs[i].p2]
        if p0.flag == 0:
            kinematic(p0, delta_time)
            particles[meshs[i].p0].pos = p0.pos
            particles[meshs[i].p0].vel = p0.vel
            particles[meshs[i].p0].flag = 1
            particles[meshs[i].p0].force = ti.math.vec2([0.0, 0.0])
        if p1.flag == 0:
            kinematic(p1, delta_time)
            particles[meshs[i].p1].pos = p1.pos
            particles[meshs[i].p1].vel = p1.vel
            particles[meshs[i].p1].flag = 1
            particles[meshs[i].p1].force = ti.math.vec2([0.0, 0.0])
        if p2.flag == 0:
            kinematic(p2, delta_time)
            particles[meshs[i].p2].pos = p2.pos
            particles[meshs[i].p2].vel = p2.vel
            particles[meshs[i].p2].flag = 1
            particles[meshs[i].p2].force = ti.math.vec2([0.0, 0.0])

@ti.func
def kinematic(particle: ti.template(), dt: ti.f32):
    particle.vel += (particle.acc + 9.8 * ti.Vector([0, -1]) + particle.force / particle.mass) * dt
    particle.pos += particle.vel * dt

    if particle.pos.x > 1:
        particle.pos.x = 1
        particle.vel.x *= v_refect

    if particle.pos.y > 1:
        particle.pos.y = 1
        particle.vel.y *= v_refect

    if particle.pos.x < 0:
        particle.pos.x = 0
        particle.vel.x *= v_refect

    if particle.pos.y < 0:
        particle.pos.y = 0
        particle.vel.y *= v_refect

    # if (particle.pos - ti.Vector([0.5, 0.5])).norm() < 0.25:
    #     pre_pos = particle.pos - particle.vel * dt
    #     dir = ti.math.normalize(particle.vel * dt)
    #     line_to_circle = ti.Vector([0.5, 0.5]) - pre_pos
    #     projection_length = line_to_circle.dot(dir)
    #     perpendicular_length = ti.math.sqrt(line_to_circle.norm() ** 2 - projection_length ** 2)
    #     intersection_distance = projection_length - ti.math.sqrt(0.25 ** 2 - perpendicular_length ** 2 )
    #     intersection_point = pre_pos + dir * intersection_distance
    #     particle.pos = intersection_point
    #
    #     n = ti.math.normalize(intersection_point - ti.Vector([0.5, 0.5]))
    #     v_norm = particle.vel.norm() * (-v_refect)
    #     particle.vel = (-dir.dot(n) * 2 * n + dir) * v_norm


    particle.force = ti.math.vec2([0.0, 0.0])

@ti.kernel
def particles_init():
    # particles[0].pos = center + ti.Vector([-0.1, -0.1])
    # particles[1].pos = center + ti.Vector([0.1, -0.1])
    # particles[2].pos = center + ti.Vector([0.1, 0.1])
    # particles[3].pos = center + ti.Vector([-0.1, 0.1])

    for i in range(particle_cnt):
        particles[i].pos = ti_vertices[i] + center
        particles[i].ref_pos = particles[i].pos
        particles[i].mass = mass

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
        pos = p0.pos
        ref_pos = p0.ref_pos
        x1_pos = p1.pos
        x1_ref_pos = p1.ref_pos
        x2_pos = p2.pos
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
        force = - 0.02 * F @ S @ R_inv.transpose()
        f1 = ti.math.vec2(force[0, 0], force[1, 0])
        f2 = ti.math.vec2(force[0, 1], force[1, 1])
        f0 = - f1 - f2
        particles[mesh.p0].force += f0
        particles[mesh.p1].force += f1
        particles[mesh.p2].force += f2
        particles[mesh.p0].flag = 0
        particles[mesh.p1].flag = 0
        particles[mesh.p2].flag = 0
        print(f1, particles[mesh.p1].pos, particles[mesh.p1].force)

if __name__ == '__main__':
    gui = ti.GUI('Finite Element Method', (640, 640))
    particles_init()
    mesh_init()
    frame_cnt = 0
    while gui.running:
        frame_cnt+=1
        # if frame_cnt == 120:
        #     particles[2].pos = center + ti.Vector([0.1, 0.1])
        fem()
        kinematic_mesh()
        # for i in range(particle_cnt):
        #     begin = particles[i].pos
        #     end = particles[(i+1)%particle_cnt].pos
        #     gui.line(begin, end, radius=1, color=0xFF0000)
        for i in range(mesh_cnt):
            p0 = particles[meshs[i].p0].pos
            p1 = particles[meshs[i].p1].pos
            p2 = particles[meshs[i].p2].pos
            gui.line(p0, p1, radius=1, color=0xFF0000)
            gui.line(p1, p2, radius=1, color=0xFF0000)
            gui.line(p2, p0, radius=1, color=0xFF0000)

        # gui.circles(particles.pos.to_numpy(), radius=2, color=0x00FF00)
        # gui.circle([0.5,0.5], color=0xFF0000, radius=160)
        gui.show()
