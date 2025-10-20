import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

import cv2

class FastRBFInterpolator2D:
    """
    Approximate RBF interpolation from irregular (X, Y, Z) data
    onto a regular grid using local neighbor-based interpolation.

    Works efficiently with large datasets (e.g. 10 million points).
    """

    def __init__(self, grid_size=(1024, 1024), neighbors=64, epsilon=0.2, device="mps"):
        """
        Parameters
        ----------
        grid_size : tuple
            (nx, ny) number of grid points in x and y directions.
        neighbors : int
            Number of nearest neighbors for local RBF interpolation.
        epsilon : float
            RBF kernel width parameter.
        device : str or None
            'mps', 'cuda', or 'cpu'. If None, auto-detects MPS (Apple GPU) if available.

        TODO: NN on GPU? !!!!!!!!!!!!!!!!!!!!!!

        """
        self.grid_size = grid_size
        self.neighbors = neighbors
        self.epsilon = epsilon
        self.device = torch.device(
            device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        )

    def fit_transform(self, X, Y, Z):
        """
        Interpolates scattered points (X, Y, Z) to a regular 2D grid.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            1D arrays of same length representing irregular sample positions and values.

        Returns
        -------
        Z_grid : np.ndarray
            2D numpy array of shape grid_size containing interpolated values.
        """

        # 1. Build regular grid
        nx, ny = self.grid_size
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xg, Yg = np.meshgrid(xi, yi)
        grid_points = np.column_stack((Xg.ravel(), Yg.ravel()))

        # 2. Find K nearest neighbors for each grid point (on CPU) ==> implement GPU!!!!
        nbrs = NearestNeighbors(n_neighbors=self.neighbors, algorithm='kd_tree').fit(
            np.column_stack((X, Y))
        )
        dists, idxs = nbrs.kneighbors(grid_points)

        # 3. Move data to GPU (MPS)
        dists_t = torch.tensor(dists, dtype=torch.float32, device=self.device)
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
        values_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        # 4. Gaussian RBF weights
        eps = self.epsilon
        weights = torch.exp(-(dists_t / eps) ** 2)

        # 5. Gather neighbor values
        local_vals = values_t[idxs_t]

        # 6. Weighted interpolation
        Z_interp = (weights * local_vals).sum(dim=1) / weights.sum(dim=1)

        # 7. Reshape back to 2D grid (numpy)
        Z_grid = Z_interp.cpu().numpy().reshape(self.grid_size).astype(np.float32)
        return Z_grid

if __name__ == '__main__':

    X = np.load("X.npy")
    Y = np.load("X.npy")
    Z = np.load("X.npy")

    RBF  = FastRBFInterpolator2D(grid_size = (4096, 4096),
                                 neighbors = 6,
                                 epsilon   = 1,
                                 device    = "cpu")
    contact_image = RBF.fit_transform(X, Y, Z)

    plt.figure(figsize=(5,4))
    plt.imshow(contact_image,
               extent = [X.min(), X.max(), Y.min(), Y.max()],
               cmap   = "magma_r",
               vmax   = contact_image.max()*0.95)
    plt.colorbar(shrink=0.5,label=r"Z [$\mu m$]",anchor=(-0.2, 0.0))
    plt.title(r"$Z(x,y) = argmax_Z (\Delta F)$",loc="left")
    plt.xlabel("$X [\mu m]$")
    plt.ylabel("$Y [\mu m]$")
    plt.savefig("topo_RBF.pdf",transparent=True,bbox_inches="tight")
    plt.show()
