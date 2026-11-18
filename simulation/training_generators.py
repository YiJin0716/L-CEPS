"""Generators used for the released model's original training datasets."""
import numpy as np

def sample_x_toeplitz(rng, n, p, rho):
	# corr(x_i, x_j) = rho ** |i - j|; always non-negative, decays with distance
	innovations = rng.normal(size=(n, p))
	x = np.empty((n, p))
	x[:, 0] = innovations[:, 0]
	for j in range(1, p):
		x[:, j] = rho * x[:, j - 1] + np.sqrt(1 - rho ** 2) * innovations[:, j]
	return x


def sample_x_factor(rng, n, p, loadings):
	# predictors load on shared latent factors
	k = loadings.shape[1]
	factors = rng.normal(size=(n, k))
	idiosyncratic = rng.normal(size=(n, p))
	return factors @ loadings.T + idiosyncratic


def apply_transform(x_j, transform, bump_scale=None):
	if transform == 'linear':
		return x_j
	elif transform == 'quadratic':
		return x_j ** 2
	elif transform == 'sin':
		return np.sin(2 * x_j)
	elif transform == 'threshold':
		return (x_j > 0).astype(float)
	elif transform == 'abs':
		return np.abs(x_j)
	elif transform == 'gaussian_bump':
		return np.exp(-bump_scale * x_j ** 2)
	elif transform == 'cauchy_bump':
		return 1 / (1 + bump_scale * x_j ** 2)
	elif transform == 'tanh':
		return np.tanh(x_j)
	elif transform == 'softplus':
		return np.maximum(x_j, 0.0) + np.log1p(np.exp(-np.abs(x_j)))  # numerically stable
	elif transform == 'relu':
		return np.maximum(x_j, 0.0)
	elif transform == 'signed_sqrt':
		return np.sign(x_j) * np.sqrt(np.abs(x_j))
	elif transform == 'band':
		return (np.abs(x_j) <= 1).astype(float)
	else:
		raise ValueError(f'unknown transform: {transform}')


def standardize(h):
	std = h.std()
	if std < 1e-8:
		return None
	return (h - h.mean()) / std


def sample_x_block(rng, n, p, block_size, rho):
    """Equicorrelated blocks, independent across blocks (outside training mixture)."""
    import numpy as np
    n_blocks = int(np.ceil(p / block_size))
    block_ids = np.repeat(np.arange(n_blocks), block_size)[:p]
    block_factors = rng.normal(size=(n, n_blocks))
    return np.sqrt(rho) * block_factors[:, block_ids] + np.sqrt(1 - rho) * rng.normal(size=(n, p))


def generate_dataset(rng, parameters, coefficient_range):
    import numpy as np

    n, p, p_true = parameters['n'], parameters['p'], parameters['p_true']
    true_features = np.sort(rng.choice(p, size=p_true, replace=False))
    coefficients = rng.choice([-1.0, 1.0], size=p_true) * rng.uniform(*coefficient_range, size=p_true)
    for _ in range(1000):
        if parameters['covariance'] == 'toeplitz':
            x = sample_x_toeplitz(rng, n, p, parameters['rho'])
        elif parameters['covariance'] == 'factor':
            loadings = rng.normal(size=(p, parameters['factor_count']))
            x = sample_x_factor(rng, n, p, loadings)
        elif parameters['covariance'] == 'block':
            x = sample_x_block(rng, n, p, parameters['block_size'], parameters['rho'])
        else:
            raise ValueError(f'unknown covariance: {parameters["covariance"]}')
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        components = [standardize(apply_transform(x[:, feature], parameters['transform'], parameters['bump_scale']))
                      for feature in true_features]
        if all(component is not None for component in components):
            signal = sum(coef * component for coef, component in zip(coefficients, components))
            break
    else:
        raise RuntimeError('could not draw a dataset with no constant true-feature component')
    noise_std = float(np.sqrt(np.var(signal) / parameters['snr']))
    y = signal + rng.normal(scale=noise_std, size=n)
    return x, y - y.mean(), true_features

