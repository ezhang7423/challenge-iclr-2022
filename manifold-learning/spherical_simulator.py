#! /usr/bin/env python

import os
from urllib import request
import numpy as np
from scipy.stats import norm
import logging
from exceptions import IntractableLikelihoodError, DatasetNotAvailableError
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)


class NumpyDataset(Dataset):
    """Dataset for numpy arrays with explicit memmap support"""

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        memmap_threshold = kwargs.get("memmap_threshold", None)

        for array in arrays:
            if isinstance(array, str):
                array = self._load_array_from_file(array, memmap_threshold)

            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n

    @staticmethod
    def _load_array_from_file(filename, memmap_threshold_gb=None):
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_threshold_gb is None or filesize_gb <= memmap_threshold_gb:
            data = np.load(filename)
        else:
            data = np.load(filename, mmap_mode="c")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return data


def download_file_from_google_drive(id, dest):
    URL = "https://docs.google.com/uc?export=download"

    session = request.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768

    with open(dest, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks.
                f.write(chunk)


class SphericalGaussianSimulator:
    def __init__(
        self,
        latent_dim=8,
        data_dim=9,
        phases=0.5 * np.pi,
        widths=0.25 * np.pi,
        epsilon=0.01,
    ):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._phases = (
            phases * np.ones(latent_dim) if isinstance(phases, float) else phases
        )
        self._widths = (
            widths * np.ones(latent_dim) if isinstance(widths, float) else widths
        )
        self._epsilon = epsilon

        assert data_dim > latent_dim
        assert epsilon > 0.0

    def load_dataset(
        self,
        train,
        dataset_dir,
        numpy=False,
        limit_samplesize=None,
        true_param_id=0,
        joint_score=False,
        ood=False,
        paramscan=False,
        run=0,
    ):
        if joint_score:
            raise NotImplementedError(
                "SCANDAL training not implemented for this dataset"
            )
        if ood and not os.path.exists("{}/x_ood.npy".format(dataset_dir)):
            raise DatasetNotAvailableError

        # Download missing data
        self._download(dataset_dir)

        tag = (
            "train" if train else "ood" if ood else "paramscan" if paramscan else "test"
        )
        param_label = ""
        if not train and true_param_id and true_param_id > 0:
            param_label = true_param_id
        run_label = "_run{}".format(run) if run > 0 else ""

        x = np.load("{}/x_{}{}{}.npy".format(dataset_dir, tag, param_label, run_label))
        if self.parameter_dim() is not None:
            params = np.load(
                "{}/theta_{}{}{}.npy".format(dataset_dir, tag, param_label, run_label)
            )
        else:
            params = np.ones(x.shape[0])

        if limit_samplesize is not None:
            logger.info(
                "Only using %s of %s available samples", limit_samplesize, x.shape[0]
            )
            x = x[:limit_samplesize]
            params = params[:limit_samplesize]

        if numpy:
            return x, params
        else:
            return NumpyDataset(x, params)

    def sample_with_noise(self, n, noise, parameters=None):
        x = self.sample(n, parameters)
        x = x + np.random.normal(loc=0.0, scale=noise, size=(n, self.data_dim()))
        return x

    def default_parameters(self, true_param_id=0):
        return np.zeros(self.parameter_dim())

    def eval_parameter_grid(self, resolution=11):
        if self.parameter_dim() is None or self.parameter_dim() < 1:
            raise NotImplementedError

        each = np.linspace(-1.0, 1.0, resolution)
        each_grid = np.meshgrid(
            *[each for _ in range(self.parameter_dim())], indexing="ij"
        )
        each_grid = [x.flatten() for x in each_grid]
        grid = np.vstack(each_grid).T
        return grid

    def _download(self, dataset_dir):
        if self.gdrive_file_ids is None:
            return

        os.makedirs(dataset_dir, exist_ok=True)

        for tag, file_id in self.gdrive_file_ids.items():
            filename = "{}/{}.npy".format(dataset_dir, tag)
            if not os.path.isfile(filename):
                logger.info("Downloading {}.npy".format(tag))
                download_file_from_google_drive(file_id, filename)

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    def log_density(self, x, parameters=None, precise=False):
        z_phi, z_eps = self._transform_x_to_z(x)
        logp = self._log_density(z_phi, z_eps, precise=precise)
        return logp

    def sample(self, n, parameters=None):
        z_phi, z_eps = self._draw_z(n)
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def sample_ood(self, n, parameters=None):
        z_phi, _ = self._draw_z(n)
        z_eps = np.random.uniform(
            -3.0 * self._epsilon, 0.0, size=(n, self._data_dim - self._latent_dim)
        )
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def distance_from_manifold(self, x):
        z_phi, z_eps = self._transform_x_to_z(x)
        return np.sum(z_eps ** 2, axis=1) ** 0.5

    def _draw_z(self, n):
        # Spherical coordinates
        phases_ = np.empty((n, self._latent_dim))
        phases_[:] = self._phases
        widths_ = np.empty((n, self._latent_dim))
        widths_[:] = self._widths
        z_phi = np.random.normal(phases_, widths_, size=(n, self._latent_dim))
        z_phi = np.mod(z_phi, 2.0 * np.pi)

        # Fuzzy coordinates
        z_eps = np.random.normal(
            0.0, self._epsilon, size=(n, self._data_dim - self._latent_dim)
        )
        return z_phi, z_eps

    def _transform_z_to_x(self, z_phi, z_eps):
        r = 1.0 + z_eps[:, 0]
        a = np.concatenate(
            (2 * np.pi * np.ones((z_phi.shape[0], 1)), z_phi), axis=1
        )  # n entries, each (2 pi, z_sub)
        sins = np.sin(a)
        sins[:, 0] = 1
        sins = np.cumprod(
            sins, axis=1
        )  # n entries, each (1, sin(z0), sin(z1), ..., sin(zk))
        coss = np.cos(a)
        coss = np.roll(
            coss, -1, axis=1
        )  # n entries, each (cos(z0), cos(z1), ..., cos(zk), 1)
        exact_sphere = sins * coss  # (n, k+1)
        fuzzy_sphere = exact_sphere * r[:, np.newaxis]
        x = np.concatenate((fuzzy_sphere, z_eps[:, 1:]), axis=1)
        return x

    def _transform_x_to_z(self, x):
        z_phi = np.zeros((x.shape[0], self._latent_dim))
        for i in range(self._latent_dim):
            z_phi[:, i] = np.arccos(
                x[:, i] / np.sum(x[:, i : self._latent_dim + 1] ** 2, axis=1) ** 0.5
            )
        # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        z_phi[:, self._latent_dim - 1] = np.where(
            x[:, self._latent_dim] < 0.0,
            2.0 * np.pi - z_phi[:, self._latent_dim - 1],
            z_phi[:, self._latent_dim - 1],
        )

        r = np.sum(x[:, : self._latent_dim + 1] ** 2, axis=1) ** 0.5
        z_eps = np.copy(x[:, self._latent_dim :])
        z_eps[:, 0] = r - 1
        return z_phi, z_eps

    def _log_density(self, z_phi, z_eps, precise=False):
        r = 1.0 + z_eps[:, 0]
        phases_ = np.empty((z_phi.shape[0], self._latent_dim))
        phases_[:] = self._phases
        widths_ = np.empty((z_phi.shape[0], self._latent_dim))
        widths_[:] = self._widths

        p_sub = 0.0
        for shifted_z_phi in self._generate_equivalent_coordinates(z_phi, precise):
            p_sub += norm(loc=phases_, scale=widths_).pdf(shifted_z_phi)

        logp_sub = np.log(p_sub)
        logp_eps = np.log(norm(loc=0.0, scale=self._epsilon).pdf(z_eps))

        log_det = self._latent_dim * np.log(np.abs(r))
        log_det += np.sum(
            np.arange(self._latent_dim - 1, -1, -1)[np.newaxis, :]
            * np.log(np.abs(np.sin(z_phi))),
            axis=1,
        )

        logp = np.sum(logp_sub, axis=1) + np.sum(logp_eps, axis=1) + log_det
        return logp

    def _generate_equivalent_coordinates(self, z_phi, precise):
        # Restrict z to canonical range: [0, pi) for polar angles, and [0., 2pi) for azimuthal angle
        z_phi = z_phi % (2.0 * np.pi)
        z_phi[:, :-1] = np.where(
            z_phi[:, :-1] > np.pi, 2.0 * np.pi - z_phi[:, :-1], z_phi[:, :-1]
        )

        # Yield this one
        yield z_phi

        # Variations of polar angles
        for dim in range(self._latent_dim - 1):
            z = np.copy(z_phi)
            z[:, dim] = -z_phi[:, dim]
            yield z

            z = np.copy(z_phi)
            z[:, dim] = 2.0 * np.pi - z_phi[:, dim]
            yield z

        # Variations of aximuthal angle
        z = np.copy(z_phi)
        z[:, -1] = -2.0 * np.pi + z_phi[:, -1]
        yield z

        z = np.copy(z_phi)
        z[:, -1] = 2.0 * np.pi + z_phi[:, -1]
        yield z
