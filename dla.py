import typing
import numpy as np
import numpy.typing
import scipy.ndimage


def calc_2d(
        scene_initial: np.ndarray,
        num_iterations: int = 100,
        num_particles: int = 100,
        num_iterations_per_frame = 100
) -> typing.Tuple[np.ndarray, np.ndarray]:

    num_frames = num_iterations // num_iterations_per_frame

    frames = np.zeros((num_frames, ) + scene_initial.shape)
    frames[0] = scene_initial
    scene = scene_initial

    particles = np.zeros_like(frames)

    # scene = initial_scene.copy()
    num_x = scene_initial.shape[~1]
    num_y = scene_initial.shape[~0]

    particle_x = np.random.randint(
        low=0,
        high=num_x - 1,
        size=(num_particles,),
    )
    particle_y = np.random.randint(
        low=0,
        high=num_y - 1,
        size=(num_particles,),
    )

    # particles_previous = np.zeros((num_x, num_y))
    # particles_previous[particle_x, particle_y] = 1
    # particles[0] = particles_previous

    particles[0, particle_x, particle_y] = 1

    for i in range(1, num_iterations):

        x9 = np.clip(particle_x - 1, 0, num_x - 1)
        x0 = particle_x
        x1 = np.clip(particle_x + 1, 0, num_x - 1)

        y9 = np.clip(particle_y - 1, 0, num_y - 1)
        y0 = particle_y
        y1 = np.clip(particle_y + 1, 0, num_y - 1)

        mask = False
        mask |= scene[x9, y9] != 0
        mask |= scene[x0, y9] != 0
        mask |= scene[x1, y9] != 0
        mask |= scene[x9, y0] != 0
        mask |= scene[x0, y0] != 0
        mask |= scene[x1, y0] != 0
        mask |= scene[x9, y1] != 0
        mask |= scene[x0, y1] != 0
        mask |= scene[x1, y1] != 0

        # frames[i] = frames[i - 1]
        scene[particle_x[mask], particle_y[mask]] = i + 1



        num_bound = np.count_nonzero(mask)
        particle_x[mask] = np.random.randint(
            low=0,
            high=num_x - 1,
            size=num_bound,
        )
        particle_y[mask] = np.random.randint(
            low=0,
            high=num_y - 1,
            size=num_bound,
        )

        is_step_x = np.random.randint(2, size=num_particles).astype(bool)
        is_step_positive = np.random.randint(2, size=num_particles).astype(bool)

        # print('particle_x.shape', particle_x.shape)
        # print('is_step_x', is_step_x)
        # print('is_step_postive', is_step_positive)

        particle_x[is_step_x & is_step_positive] += 1
        particle_x[is_step_x & ~is_step_positive] -= 1
        particle_y[~is_step_x & is_step_positive] += 1
        particle_y[~is_step_x & ~is_step_positive] -= 1

        particle_x = np.clip(particle_x, 0, num_x - 1)
        particle_y = np.clip(particle_y, 0, num_y - 1)

        if i % num_iterations_per_frame == 0:
            frames[i // num_iterations_per_frame] = scene
            particles[i // num_iterations_per_frame, particle_x, particle_y] = i + 1

    return frames, particles



