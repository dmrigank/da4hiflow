"""1D Euler shock tube solver (Sod) with HLLE flux and RK2 time stepping.

State is conservative variables U on grid of nx cells: U shape (nx, 3)
flattened as 1D array for compatibility with Runner.
"""

import numpy as np
from typing import Tuple

from da4hiflow.core.system_base import SystemBase


class Euler1DShockTube(SystemBase):
    """Finite-volume 1D Euler solver configured for Sod shock tube."""

    def __init__(
        self,
        nx: int = 100,
        x0: float = 0.0,
        x1: float = 1.0,
        gamma: float = 1.4,
        left: Tuple[float, float, float] = (1.0, 0.0, 1.0),
        right: Tuple[float, float, float] = (0.125, 0.0, 0.1),
        cfl: float = 0.5,
        seed: int = 0,
    ):
        self.nx = int(nx)
        self.x0 = x0
        self.x1 = x1
        self.gamma = float(gamma)
        self.x = np.linspace(x0, x1, nx)
        self.dx = (x1 - x0) / float(nx)
        self.cfl = float(cfl)
        self.left = tuple(left)
        self.right = tuple(right)

    # primitive <-> conservative helpers
    def prim_to_cons(self, prim: np.ndarray) -> np.ndarray:
        # prim shape (nx, 3): rho, u, p
        rho = prim[:, 0]
        u = prim[:, 1]
        p = prim[:, 2]
        E = p / (self.gamma - 1.0) + 0.5 * rho * u ** 2
        U = np.stack([rho, rho * u, E], axis=1)
        return U

    def cons_to_prim(self, U: np.ndarray) -> np.ndarray:
        rho = U[:, 0]
        u = U[:, 1] / rho
        E = U[:, 2]
        p = (self.gamma - 1.0) * (E - 0.5 * rho * u ** 2)
        prim = np.stack([rho, u, p], axis=1)
        return prim

    def initial_condition(self) -> np.ndarray:
        prim = np.zeros((self.nx, 3))
        # set left/right by x mid
        mid = 0.5 * (self.x0 + self.x1)
        left_mask = self.x <= mid
        right_mask = ~left_mask
        prim[left_mask, 0] = self.left[0]
        prim[left_mask, 1] = self.left[1]
        prim[left_mask, 2] = self.left[2]
        prim[right_mask, 0] = self.right[0]
        prim[right_mask, 1] = self.right[1]
        prim[right_mask, 2] = self.right[2]
        U = self.prim_to_cons(prim)
        return U.flatten()

    # back-compat wrapper
    def get_initial_state(self) -> np.ndarray:
        return self.initial_condition()

    def max_char_speed(self, U: np.ndarray) -> float:
        prim = self.cons_to_prim(U.reshape(self.nx, 3))
        rho = prim[:, 0]
        u = prim[:, 1]
        p = prim[:, 2]
        c = np.sqrt(np.maximum(0.0, self.gamma * p / rho))
        return float(np.max(np.abs(u) + c))

    def flux(self, U: np.ndarray) -> np.ndarray:
        # compute flux for each cell
        prim = self.cons_to_prim(U.reshape(self.nx, 3))
        rho = prim[:, 0]
        u = prim[:, 1]
        p = prim[:, 2]
        E = U.reshape(self.nx, 3)[:, 2]
        F = np.empty_like(U.reshape(self.nx, 3))
        F[:, 0] = rho * u
        F[:, 1] = rho * u ** 2 + p
        F[:, 2] = u * (E + p)
        return F

    def hlle_flux(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        # UL, UR are conservative vectors shape (3,) or arrays (n,3)
        # convert to prim
        UL = UL.reshape(-1, 3)
        UR = UR.reshape(-1, 3)
        primL = self.cons_to_prim(UL)
        primR = self.cons_to_prim(UR)
        rhoL, uL, pL = primL[:, 0], primL[:, 1], primL[:, 2]
        rhoR, uR, pR = primR[:, 0], primR[:, 1], primR[:, 2]
        cL = np.sqrt(np.maximum(0.0, self.gamma * pL / rhoL))
        cR = np.sqrt(np.maximum(0.0, self.gamma * pR / rhoR))

        # fluxes
        FL = np.empty_like(UL)
        FR = np.empty_like(UR)
        EL = UL[:, 2]
        ER = UR[:, 2]
        FL[:, 0] = rhoL * uL
        FL[:, 1] = rhoL * uL ** 2 + pL
        FL[:, 2] = uL * (EL + pL)
        FR[:, 0] = rhoR * uR
        FR[:, 1] = rhoR * uR ** 2 + pR
        FR[:, 2] = uR * (ER + pR)

        sL = np.minimum(uL - cL, uR - cR)
        sR = np.maximum(uL + cL, uR + cR)

        # HLLE formula
        # avoid division by zero
        denom = (sR - sL)
        denom[denom == 0.0] = 1e-12
        num = sR[:, None] * FL - sL[:, None] * FR + (sR * sL)[:, None] * (UR - UL)
        FH = num / denom[:, None]
        # when sL>=0 -> FL, when sR<=0 -> FR
        out = FH.copy()
        maskL = sL >= 0.0
        maskR = sR <= 0.0
        out[maskL] = FL[maskL]
        out[maskR] = FR[maskR]
        return out

    def apply_bcs(self, U: np.ndarray) -> np.ndarray:
        # simple transmissive (zero-gradient) BCs implemented by extending edges
        arr = U.reshape(self.nx, 3)
        U_ext = np.vstack([arr[0:1, :], arr, arr[-1:, :]])
        return U_ext

    def L(self, U_flat: np.ndarray) -> np.ndarray:
        # compute spatial derivative (finite-volume) returning dU/dt
        U_ext = self.apply_bcs(U_flat)
        # compute HLLE fluxes at interfaces between cells i and i+1
        UL = U_ext[:-1]
        UR = U_ext[1:]
        FH = self.hlle_flux(UL, UR)  # shape (nx+1,3)
        # interior flux differences
        res = -(FH[1:] - FH[:-1]) / self.dx
        return res.reshape(-1)

    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance one time step using CFL to compute dt and RK2 (Heun).

        The function computes stable dt from CFL and uses it to advance.
        Returns flattened state.
        """
        U = state.copy().reshape(self.nx, 3)
        maxspeed = self.max_char_speed(state)
        if maxspeed <= 0.0:
            dt = 1e-6
        else:
            dt = self.cfl * self.dx / maxspeed

        # RK2 (Heun)
        k1 = self.L(U.flatten())
        U1 = (U.flatten() + dt * k1).reshape(self.nx, 3)
        k2 = self.L(U1.flatten())
        Unew = (U.flatten() + 0.5 * dt * (k1 + k2)).reshape(self.nx, 3)

        # ensure positive density and pressure floor
        prim = self.cons_to_prim(Unew)
        prim[:, 0] = np.maximum(prim[:, 0], 1e-8)
        prim[:, 2] = np.maximum(prim[:, 2], 1e-8)
        Unew = self.prim_to_cons(prim)
        return Unew.flatten()
