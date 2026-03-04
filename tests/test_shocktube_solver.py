import numpy as np
from da4hiflow.systems.euler1d_shocktube import Euler1DShockTube


def test_shocktube_step_basic():
    sys = Euler1DShockTube(nx=50, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), gamma=1.4, cfl=0.5)
    u0 = sys.get_initial_state()
    assert u0.shape == (sys.nx * 3,)

    u1 = sys.step(u0)
    u2 = sys.step(u1)

    # No NaNs, finite values
    assert np.all(np.isfinite(u1))
    assert np.all(np.isfinite(u2))

    # density (first component in primitive) should remain positive after converting
    prim = sys.cons_to_prim(u2.reshape(sys.nx, 3))
    rho = prim[:, 0]
    assert np.all(rho > 0)
