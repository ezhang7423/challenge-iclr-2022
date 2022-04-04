# Reimplementation of the core manifold learning ideas from M-flows
import torch
import logging
from exceptions import NoMeanException
from torch import nn
import numpy as np

def product(x):
    try:
        prod = 1
        for factor in x:
            prod *= factor
        return prod
    except:
        return x

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not num_batch_dims >= 0:
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)



def is_positive_int(x):
    return isinstance(x, int) and x > 0


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = merge_leading_dims(samples, num_dims=2)
            context = repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(self, context=None):
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def _mean(self, context):
        raise NoMeanException()


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape)
            return split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


logger = logging.getLogger(__name__)


class Flow:
    """Ambient normalizing flow (AF)"""

    def __init__(self, data_dim, transform):
        super(Flow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)

        self.latent_distribution = StandardNormal((self.total_latent_dim,))
        self.transform = transform

        self._report_model_parameters()

    def project(self, x, context=None):
        return self.decode(self.encode(x, context), context)

    def forward(self, x, context=None):
        """Transforms data point to latent space, evaluates log likelihood"""

        # Encode
        u, log_det = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return x, log_prob, u

    def encode(self, x, context=None):
        """Encodes data point to latent space"""

        u, _ = self._encode(x, context=context)
        return u

    def decode(self, u, context=None):
        """Encodes data point to latent space"""

        x, _ = self.transform.inverse(u, context=context)
        return x

    def log_prob(self, x, context=None):
        """Evaluates log likelihood"""

        # Encode
        u, log_det = self._encode(x, context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return log_prob

    def sample(self, u=None, n=1, context=None):
        """Generates samples from model"""

        if u is None:
            u = self.latent_distribution.sample(n, context=None)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, context=None):
        u, log_det = self.transform(x, context=context)
        return u, log_det

    def _report_model_parameters(self):
        """ Reports the model size """

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.info("Model has %.1f M parameters (%.1f M trainable) with an estimated size of %.1f MB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e6)
