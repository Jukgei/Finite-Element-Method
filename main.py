# coding=utf-8

import taichi as ti
import numpy as np
import trimesh as tm
from trimesh.interfaces import gmsh
import meshio as mio
ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

dim = 3
if dim == 2:
    vec = ti.math.vec2
    mat = ti.math.mat2
else: # 3d
    vec = ti.math.vec3
    mat = ti.math.mat3

width = 640
height = 640


if dim == 2:
    block_center = [0.5, 0.5]
    center = ti.Vector([0.72, 0.8])
    g_dir = [0, -1]
    block_radius = 0.33
else:
    block_center = [0.5, 0.5, 0.5]
    center = ti.Vector([2, 1, 2])
    g_dir = [0, -1, 0]
    block_radius = 0.0

E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, s_lambda = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
damping = 5

# center = ti.Vector([0.55, 0.3])
v_refect = -0.3
delta_time = 5e-4
side_length = 0.2  #
subdivisions = 10  #

rho = 1000


auto_diff = True

if dim == 2:
    x = np.linspace(0, side_length, subdivisions + 1)  #
    y = np.linspace(0, side_length, subdivisions + 1)  #
    vertices = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  #

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
    element_indices = faces
    A = ((side_length / subdivisions) ** 2) / 2
    mass = rho * A
else:
    # mm = tm.load_mesh('./obj/stanford-bunny.obj')
    obj = tm.load_mesh('./obj/cube2.stl')
    obj.apply_scale(1)
    # mesh = tm.Trimesh(vertices=mm.vertices, faces=mm.faces)
    msh = gmsh.to_volume(obj, './obj/cube2.msh', mesher_id=7)
    mesh4 = mio.read('./obj/cube2.msh')
    vertices = mesh4.points

    # 获取四面体的点的indices
    tetra_indices = mesh4.cells[0].data
    element_indices = tetra_indices
    # 遍历每个四面体
    for tetra in tetra_indices:
        # 获取四面体四个点的索引
        point_indices = tetra

        # 获取四面体四个点的坐标
        tetra_points = vertices[point_indices]

        # 打印四面体的信息
        print("Tetrahedron:")
        print("Point Indices:", point_indices)
        print("Points:")
        for point in tetra_points:
            print(point)
        print()

    faces = []
    for tetra in tetra_indices:
        x, y, z, w = tetra
        faces.append([x, y, z])
        faces.append([x, y, w])
        faces.append([x, z, w])
        faces.append([y, z, w])
    faces = np.array(faces)
    # faces = tetra_indices # Volume
    # vertices = mesh.vertices
    mass = rho / tetra_indices.shape[0]
    # Boundary Box
    box_min = ti.Vector([0, 0, 0])
    box_max = ti.Vector([5, 5, 5])
    box_vert = ti.Vector.field(3, ti.f32, shape=12)
    box_vert[0] = ti.Vector([box_min.x, box_min.y, box_min.z])
    box_vert[1] = ti.Vector([box_min.x, box_max.y, box_min.z])
    box_vert[2] = ti.Vector([box_max.x, box_min.y, box_min.z])
    box_vert[3] = ti.Vector([box_max.x, box_max.y, box_min.z])
    box_vert[4] = ti.Vector([box_min.x, box_min.y, box_max.z])
    box_vert[5] = ti.Vector([box_min.x, box_max.y, box_max.z])
    box_vert[6] = ti.Vector([box_max.x, box_min.y, box_max.z])
    box_vert[7] = ti.Vector([box_max.x, box_max.y, box_max.z])
    box_lines_indices = ti.field(int, shape=(2 * 12))
    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

ti_vertices = ti.Vector.field(dim, ti.f32, shape=vertices.shape[0])
ti_vertices.from_numpy(vertices)
ti_faces = ti.Vector.field(3, ti.i32, shape=faces.shape[0])
ti_faces.from_numpy(faces)
ti_element = ti.Vector.field(4, ti.i32, shape=element_indices.shape[0])
ti_element.from_numpy(element_indices)

phi = ti.field(dtype=ti.f32, shape=faces.shape[0])
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
indices = ti.field(dtype=ti.i32, shape=faces.shape[0] * 3)

Particle = ti.types.struct(
    pos=vec,
    vel=vec,
    acc=vec,
    mass=ti.f32,
    force=vec,
    ref_pos=vec
)

Mesh = ti.types.struct(
    p0=ti.i32,
    p1=ti.i32,
    p2=ti.i32,
    ref=mat
)

Element = ti.types.struct(
    p0=ti.i32,
    p1=ti.i32,
    p2=ti.i32,
    p3=ti.i32,
    ref=mat
)

particle_cnt = vertices.shape[0]
mesh_cnt = faces.shape[0]
element_cnt = element_indices.shape[0]
meshs = Mesh.field(shape=mesh_cnt)
elements = Element.field(shape=element_cnt)
particles = Particle.field(shape=particle_cnt, needs_grad=True)
print('Vertex count: {}'.format(particle_cnt))
print('Mesh count: {}'.format(mesh_cnt))
print('Element count: {}'.format(element_cnt))
print('Element mass: {}'.format(mass))

@ti.kernel
def compute_energy():
    for i in range(element_cnt):
        element = elements[i]
        p0 = particles[element.p0].pos
        p1 = particles[element.p1].pos
        p2 = particles[element.p2].pos
        p3 = particles[element.p3].pos
        p10 = p1 - p0
        p20 = p2 - p0
        p30 = p3 - p0
        # I = mat([1, 0, 0, 1])
        I = mat([1, 0, 0, 0, 1, 0, 0, 0, 1])
        # X = mat([p10.x, p20.x, p10.y, p20.y])
        X = ti.Matrix.cols([p10, p20, p30])

        R_inv = element.ref
        F = X @ R_inv
        # if i == 1:
        #     print(X)
        #     print('F {}'.format(X @ ti.math.inverse(X)))
        # S = ti.abs(p10.cross(p20))
        # S = p10.cross(p20).norm() / 2
        V = (1/6) * ti.abs(p10.dot(p20.cross(p30)))
        # K = G.transpose() @ G
        # U[None] += S * (0.5 * s_lambda * G.trace() **2 + mu * K.trace())

        # Neo-Hookean
        F_i = F
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - dim)
        phi_i -= mu * log_J_i
        phi_i += s_lambda / 2 * log_J_i ** 2
        phi[i] = phi_i * V
        # print(phi[i], log_J_i, F_i.determinant())
        # print(U[None], S * phi_i)
        U[None] += V * phi_i


        # StVK
        # G = 0.5 * (F.transpose() @ F - I)
        # phi_i = (G ** 2).sum() * mu
        # phi_i += s_lambda/2 * G.trace() **2
        # phi[i] = phi_i * S
        # U[None] += S * phi_i


@ti.kernel
def kinematic_mesh():
    for i in range(particle_cnt):
        kinematic(i, delta_time)

@ti.func
def kinematic(index: ti.int32, dt: ti.f32):
    if auto_diff:
        particles[index].vel += (particles[index].acc + 9.8 * ti.Vector(g_dir) - particles.pos.grad[index] /
                                 particles[index].mass) * dt
        # particles[index].vel += (particles[index].acc + 9.8 * ti.Vector(g_dir)) * dt
    else:
        particles[index].vel += (particles[index].acc + 9.8 * ti.Vector(g_dir) + particles[index].force / particles[index].mass) * dt
    #

    particles[index].vel *= ti.exp(-dt * damping)

    if particles[index].pos.x > 1 and particles[index].vel.x > 0:
        particles[index].vel.x = 0

    if particles[index].pos.y > 1 and particles[index].vel.y > 0:
        particles[index].vel.y = 0

    if particles[index].pos.z > 1 and particles[index].vel.z > 0:
        particles[index].vel.z = 0

    if particles[index].pos.x < 0 and particles[index].vel.x < 0:
        particles[index].vel.x = 0

    if particles[index].pos.y < 0 and particles[index].vel.y < 0:
        particles[index].vel.y = 0

    if particles[index].pos.z < 1 and particles[index].vel.z < 0:
        particles[index].vel.z = 0

    if (particles[index].pos - ti.Vector(block_center)).norm() < block_radius and particles[index].vel.dot(ti.Vector(block_center) - particles[index].pos) > 0:
        disp = particles[index].pos - ti.Vector(block_center)
        particles[index].vel -= particles[index].vel.dot(disp) * disp / disp.norm_sqr()
        # particles[index].vel =ti.Vector([-n.y, n.x]) * v_norm

    particles.pos[index] += particles.vel[index] * dt
    # if index == 0:
    #     print(particles.vel[index])
    if not auto_diff:
        particles[index].force = vec(0.0)

@ti.kernel
def particles_init():

    for i in range(particle_cnt):
        particles[i].pos = ti_vertices[i] + center
        particles[i].ref_pos = particles[i].pos
        particles[i].mass = mass

@ti.kernel
def elements_init():
    for i in range(element_cnt):
        elements[i].p0 = ti_element[i][0]
        elements[i].p1 = ti_element[i][1]
        elements[i].p2 = ti_element[i][2]
        elements[i].p3 = ti_element[i][3]

        a, b, c, d = ti_element[i]
        p_a, p_b, p_c, p_d = particles[a].pos, particles[b].pos, particles[c].pos, particles[d].pos
        # if dim == 2:
        #     r = ti.Matrix.cols([p_b - p_a, p_c - p_a])
        # else:
        r = ti.Matrix.cols([p_b - p_a, p_c - p_a, p_d - p_a])

        elements[i].ref = ti.math.inverse(r)


@ti.kernel
def mesh_init():
    for i in range(mesh_cnt):
        meshs[i].p0 = ti_faces[i][0]
        meshs[i].p1 = ti_faces[i][1]
        meshs[i].p2 = ti_faces[i][2]

        indices[i * 3 + 0] = ti_faces[i][0]
        indices[i * 3 + 1] = ti_faces[i][1]
        indices[i * 3 + 2] = ti_faces[i][2]

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
        I = mat([1, 0, 0, 1])
        X = mat([x10.x, x20.x, x10.y, x20.y])
        R = mat([r10.x, r20.x, r10.y, r20.y])
        R_inv = ti.math.inverse(R)
        F = X @ R_inv
        G = 0.5 * (F.transpose() @ F - I)
        S = 2 * mu * G + s_lambda * G.trace() * I
        force = - A * F @ S @ R_inv.transpose()
        f1 = vec(force[:, 0])
        f2 = vec(force[:, 1])
        f0 = - f1 - f2
        particles[mesh.p0].force += f0
        particles[mesh.p1].force += f1
        particles[mesh.p2].force += f2

        # print(f1, particles[mesh.p1].pos, particles[mesh.p1].force)
        # if ti.math.isnan(f1[0]):
        #     print(F, S)

def render2d(gui):
    pos_ = particles.pos.to_numpy()
    phi_ = phi.to_numpy()

    base_ = 0.13
    gui.triangles(a=pos_[meshs.p0.to_numpy()], b=pos_[meshs.p1.to_numpy()], c=pos_[meshs.p2.to_numpy()],
                  color=ti.rgb_to_hex([phi_ + base_, base_, base_]))
    gui.circles(pos_, radius=2, color=0xAAAA00)
    gui.circle(block_center, color=0x343434, radius=block_radius * width)
    gui.show()

def render3d(window, camera):

    canvas = window.get_canvas()
    scene = ti.ui.Scene()

    # Camera & light
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)
    pos_ = particles.pos.to_numpy()

    # scene.particles(particles.pos, color=(1.0, 1.0, 1), radius=.0001)
    scene.mesh(particles.pos, indices, show_wireframe=False)
    canvas.scene(scene)
    window.show()


if __name__ == '__main__':

    if dim == 2:
        gui = ti.GUI('Finite Element Method', (width, height))
        widget = gui
        camera = None
    else:
        window = ti.ui.Window('Window Title', res=(width, height), pos=(150, 150))

        canvas = window.get_canvas()

        scene = ti.ui.Scene()

        camera = ti.ui.Camera()
        camera.position(-1.92, 2.28, -0.14)
        camera.lookat(-1.04, 2.12, 0.30)
        camera.up(0, 1, 0)
        scene.set_camera(camera)

        gui = window.get_gui()
        widget = window

    particles_init()
    mesh_init()
    elements_init()
    frame_cnt = 0
    while widget.running:
        if widget.is_pressed('c'):
            print('Camera position [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_position)))
            print('Camera look at [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_lookat)))
            print('Camera up [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_up)))
        frame_cnt+=1
        # if frame_cnt == 120:
        #     particles[2].pos = center + ti.Vector([0.1, 0.1])
        # U[None] = 0
        for i in range(50):
            if not auto_diff:
                fem()
            else:
                pass
                # if frame_cnt == 1:
                with ti.ad.Tape(loss=U):
                    compute_energy()
            kinematic_mesh()

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

        if dim == 2:
            render2d(widget)
        else:
            # canvas = window.get_canvas()
            # scene = ti.ui.Scene()

            # # Camera & light
            # camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
            # scene.set_camera(camera)
            # scene.ambient_light((0.8, 0.8, 0.8))
            # scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            #
            # scene.particles(particles.pos, color=(1.0, 1.0, 1), radius=1)
            # # print('1')
            # canvas.scene(scene)
            # window.show()
            render3d(widget, camera)
            pass
            # render3d(widget)
