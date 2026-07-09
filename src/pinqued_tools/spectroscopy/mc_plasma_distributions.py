#%%
import numpy as np

from numpy.typing import NDArray

from abc import ABC
from numba import njit, prange, set_num_threads

# Restrict Numba to 7 cores before executing the function
set_num_threads(7)

from scipy.constants import e, epsilon_0, k
from matplotlib import pyplot as plt

#%%
class MCSimulator(ABC):
    def __init__(self,
                 n_samples: int = 1000,
                 seed: int | None = None):
        self._n_samples = n_samples
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value


# ==========================
# MC Plasma Distribution
# ==========================


class MCPlasmaMicrofieldsSampler(MCSimulator):
    """
    Monte Carlo sampler of the plasma microfield at the origin due to
    `n_particles` charges drawn from a homogeneous distribution of the given
    number `density`.

    NOTE on `linear_size`: this used to be a free parameter, independently
    settable alongside `density`, which meant the *actual* simulated density
    (n_particles / linear_size**3) could silently disagree with the
    `density` you asked for -- e.g. if n_particles was later overridden
    without touching linear_size. It is no longer a constructor argument.
    Instead the box side length is *derived* from density and n_particles
    as L = (n_particles / density) ** (1/3), the unique box size for which
    a uniform distribution of n_particles matches the requested density
    exactly. This makes linear_size a purely internal bookkeeping quantity:
    changing n_particles (for statistical convergence) automatically
    rescales the box so density is preserved, and the resulting field
    distribution depends only on `density` (up to Monte Carlo noise that
    shrinks as n_particles grows), not on any independently chosen box size.
    """

    def __init__(self,
                 seed: int | None = None,
                 temperature: float = 300.0,  # in eV
                 density: float = 1e16,  # in m^-3
                 external_field: float = 0.0,  # in V/m
                 mass: float = 1.0,
                 n_particles: int = 500):
        super().__init__(n_particles, seed)
        self._temperature = temperature
        self._density = density
        self._external_field = external_field
        self._mass = mass
        self._points = self._generate_points()

    @property
    def temperature(self):
        return self._temperature

    @property
    def density(self):
        return self._density

    @property
    def mass(self):
        return self._mass

    @property
    def linear_size(self) -> float:
        # Derived, read-only: the box side length implied by the current
        # density and particle count. No longer an independent input.
        return (self._n_samples / self._density) ** (1.0 / 3.0)

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @density.setter
    def density(self, value):
        self._density = value
        self._points = self._generate_points()

    @mass.setter
    def mass(self, value):
        self._mass = value

    @MCSimulator.n_samples.setter
    def n_samples(self, value):
        # Overridden so that changing the particle count also regenerates
        # points at the box size that keeps density consistent.
        self._n_samples = value
        self._points = self._generate_points()

    def _generate_points(self) -> NDArray:
        # Uniformly distribute n_particles inside a cube whose side length
        # is derived from density (see class docstring), so the *actual*
        # simulated density always matches self._density exactly, no matter
        # how n_particles is chosen.
        half_side = 0.5 * self.linear_size
        points = np.random.uniform(low=-half_side,
                                    high=half_side,
                                    size=(self._n_samples, 3))
        return points

    def _compute_microfields(self, points: NDArray) -> NDArray:
        # All quantities are SI here (density in m^-3, field in V/m).
        distances = np.linalg.norm(points, axis=1)

        safe = distances > 1e-10
        magnitudes = np.zeros_like(distances)
        magnitudes[safe] = e / (4 * np.pi * epsilon_0 * distances[safe] ** 2)

        if self._external_field == 0.0:
            return magnitudes

        unit_vectors = np.zeros_like(points)
        unit_vectors[safe] = points[safe] / distances[safe, np.newaxis]

        microfield_vectors = unit_vectors * magnitudes[:, np.newaxis]
        external_field_vector = self._external_field * np.array([0.0, 0.0, 1.0])
        total_field_vectors = microfield_vectors + external_field_vector[np.newaxis, :]
        return np.linalg.norm(total_field_vectors, axis=1)

    def sample(self) -> NDArray:
        return self._compute_microfields(self._points)


@njit(parallel=True, fastmath=True)
def _numba_sampler(n_particles, n_samples, density, external_field, k_factor):
    """
    Numba-jitted function to sample microfields in parallel.

    The box side length is derived from `density` and `n_particles` as
    L = (n_particles / density) ** (1/3) -- the unique box size for which
    n_particles placed uniformly at random reproduce the requested density
    exactly. There is no independent "linear_size" input: it is computed
    fresh every call from the two physically meaningful quantities, so the
    output distribution depends only on `density` (n_particles controls
    Monte Carlo convergence, not the underlying physics).

    Parameters
    ----------
    n_particles : int
        Number of particles to generate for each microfield sample.
    n_samples : int
        Number of microfield samples to generate.
    density : float
        Number density of particles (in m^-3).
    external_field : float
        Magnitude of the external electric field (in V/m), assumed along z.
    k_factor : float
        The pre-calculated Coulomb constant factor `e / (4 * pi * epsilon_0)`.

    Returns
    -------
    NDArray
        An array of `n_samples` microfield magnitudes (in V/m).
    """
    linear_size = (n_particles / density) ** (1.0 / 3.0)

    total_field_magnitudes = np.zeros(n_samples, dtype=np.float64)
    external_field_vector = np.zeros(3, dtype=np.float64)
    external_field_vector[2] = external_field

    for i in prange(n_samples):
        points = np.random.uniform(low=-0.5 * linear_size,
                                    high=0.5 * linear_size,
                                    size=(n_particles, 3))

        total_microfield_vector = np.zeros(3, dtype=np.float64)

        for j in range(n_particles):
            dist_sq = points[j, 0]**2 + points[j, 1]**2 + points[j, 2]**2
            if dist_sq > 1e-20:
                microfield_mag = k_factor / dist_sq
                dist = np.sqrt(dist_sq)
                total_microfield_vector += (points[j] / dist) * microfield_mag

        total_field_vector = total_microfield_vector + external_field_vector
        total_field_magnitudes[i] = np.sqrt(total_field_vector[0]**2 +
                                             total_field_vector[1]**2 +
                                             total_field_vector[2]**2)

    return total_field_magnitudes


class MCPlasmaMicrofieldsSampler_numba(MCPlasmaMicrofieldsSampler):
    def sample(self, n_samples_to_draw: int = 10000) -> NDArray:
        n_samples_to_draw = int(n_samples_to_draw)
        k_factor = e / (4.0 * np.pi * epsilon_0)
        return _numba_sampler(self.n_samples, n_samples_to_draw, self.density,
                               self._external_field, k_factor)

    def _generate_points(self) -> NDArray:
        # Unused by this subclass: sample() regenerates points fresh inside
        # the JIT loop every call, so there's no point precomputing/storing
        # a self._points array here.
        return np.empty((0, 3))


#%%
if __name__ == "__main__":
    import time

    # --- Parameters for the simulation ---
    DENSITY = 1e15  # m^-3
    EXTERNAL_FIELD = 10.0 * 1e2# V/m
    N_PARTICLES_PER_SAMPLE = 500  # controls MC convergence, not physics
    N_SAMPLES_TO_DRAW = 1000000  # number of independent field realizations

    print(f"Density: {DENSITY:e} m^-3, External Field: {EXTERNAL_FIELD} V/m")
    print(f"Particles per sample: {N_PARTICLES_PER_SAMPLE}")
    print(f"Simulating {N_SAMPLES_TO_DRAW} microfield samples with {N_PARTICLES_PER_SAMPLE} particles each.")
    print("-" * 40)

    print("Running Numba-accelerated sampler...")
    mc_plasma_numba = MCPlasmaMicrofieldsSampler_numba(
        seed=42,
        density=DENSITY,
        external_field=EXTERNAL_FIELD,
        n_particles=N_PARTICLES_PER_SAMPLE,
    )

    start_time = time.perf_counter()
    microfield_samples_numba = mc_plasma_numba.sample(n_samples_to_draw=N_SAMPLES_TO_DRAW) / 100  # V/m -> V/cm
    end_time = time.perf_counter()
    print(f"Numba version took: {end_time - start_time:.4f} seconds")
    print("-" * 40)

    fig, ax = plt.subplots()
    microfield_samples_numba = microfield_samples_numba[microfield_samples_numba < 50]
    ax.hist(microfield_samples_numba, bins=150, density=True, label=f'Numba ({end_time - start_time:.2f}s)')
    ax.set_xlabel('Microfield (V/cm)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Monte Carlo Plasma Microfield Distribution')
    ax.set_yscale('log')
    ax.legend()

# %%