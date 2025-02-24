from prototype_phd.utils import *
from prototype_phd.model_utils import *
from prototype_phd.types import *

import collections
import typing
from typing import Union
from warnings import warn

import numpy as np
from scipy.linalg import schur, eigvals
from scipy.sparse import csr_matrix, csc_matrix

def fermi_learning(fitnessA: np.typing.NDArray,  # fitness of strategy A
                   fitnessB: np.typing.NDArray,  # fitness of strategy B
                   β: np.typing.NDArray,  # learning rate
                   ) -> np.typing.NDArray:
    """Compute the likelihood that a player with strategy A adopts strategy B using the fermi function."""
    return (1 + np.exp(-β*(fitnessB - fitnessA)))**(-1)

T_type = list[np.typing.NDArray[np.typing.Shape["N_models"], typing.Any]]

def fixation_rate(Tplus: T_type,  # A list of NDarrays, one array (of size n_models) for each possible number of mutants in the population; the probability of gaining one mutant
                  # A list of NDarrays, one array (of size n_models) for each possible number of mutants in the population; the probability of losing one mutant
                  Tneg: T_type,
                  ) -> np.typing.NDArray[np.typing.Shape["N_models"], typing.Any]:  # Fixation rates for the given strategy in each model
    """Calculate the likelihood that a mutant invades the population."""
    Z = len(Tplus) + 1
    ρ = (np.sum([np.prod([Tneg[j-1]/Tplus[j-1]
                         for j in range(1, i+1)],
                         axis=0,
                         keepdims=False)
                 for i in range(1, Z)],
                axis=0,
                keepdims=False)
         + 1)**-1
    # The fixation rate may be very close to 0. Innacuracies with floats
    # may mean that we run into issues later on. We assume the fixation rate
    # never drops below 1e-10.
    ρ = np.maximum(ρ, 1e-10)
    return ρ

@multi
def fixation_rate_stable(ΠA: list,  # Average payoffs for the strategy A they consider adopting for each number of mutants following A
                         ΠB: list,  # Average payoffs for the strategy B that the player currently follows for each number of mutants following A
                         β: Array1D,  # learning rate
                         method="cheap", # method to dispatch on
                         ):
       return method

@method(fixation_rate_stable)
def fixation_rate_stable(ΠA: list,  # Average payoffs for the strategy A they consider adopting for each number of mutants following A
                         ΠB: list,  # Average payoffs for the strategy B that the player currently follows for each number of mutants following A
                         β: Array1D,  # learning rate
                         method=None, # method to dispatch on
                         ):
    """Calculate the likelihood that a mutant B invades population A
    using a numerically stable method."""
    assert len(ΠA) == len(ΠB)
    Z = len(ΠA) + 1
    ρ = (np.sum([np.exp(np.clip(np.sum([-β*(ΠB[j-1] - ΠA[j-1])
                                        for j in range(1, i+1)],
                                       axis=0,
                                       keepdims=False),
                                -500,
                                500))  # avoid underflow/overflow warnings
                 for i in range(1, Z)],
                axis=0,
                keepdims=False)
         + 1)**-1
    # The fixation rate may be very close to 0. Innacuracies with floats
    # may mean that we run into issues later on. We assume the fixation rate
    # never drops below 1e-10.
# ρ = np.maximum(ρ, 1e-15)
    return ρ

@method(fixation_rate_stable, "cheap")
def fixation_rate_stable(ΠA: list,  # Average payoffs for the strategy A they consider adopting for each number of mutants following A
                         ΠB: list,  # Average payoffs for the strategy B that the player currently follows for each number of mutants following A
                         β: Array1D,  # learning rate
                         method=None, # method to dispatch on
                         ):
    """Calculate the likelihood that a mutant B invades population A
    using a numerically stable method."""
    assert len(ΠA) == len(ΠB)
    Z = len(ΠA) + 1
    # avoid underflow/overflow warnings
    ρ = (np.sum(np.exp(np.clip(np.cumsum([-β*(ΠB[j-1] - ΠA[j-1])
                                          for j in range(1, Z)],
                                         axis=0),
                               -500,
                               500)),  
                axis=0,
                keepdims=False)
         + 1)**-1
    # The fixation rate may be very close to 0. Innacuracies with floats
    # may mean that we run into issues later on. We assume the fixation rate
    # never drops below 1e-10.
#     ρ = np.maximum(ρ, 1e-15)
    return ρ

@method(fixation_rate_stable, "cheap2")
def fixation_rate_stable(ΠA: list,  # Average payoffs for the strategy A they consider adopting for each number of mutants following A
                         ΠB: list,  # Average payoffs for the strategy B that the player currently follows for each number of mutants following A
                         β: Array1D,  # learning rate
                         method=None, # method to dispatch on
                         ):
    """Calculate the likelihood that a mutant B invades population A
    using a numerically stable method."""
    assert len(ΠA) == len(ΠB)
    Z = len(ΠA) + 1
    # avoid underflow/overflow warnings
    ρ = (np.sum(2**(np.clip(np.cumsum([-β*(ΠB[j-1] - ΠA[j-1])
                                          for j in range(1, Z)],
                                         axis=0),
                               -500,
                               500)),  
                axis=0,
                keepdims=False)
         + 1)**-1
    # The fixation rate may be very close to 0. Innacuracies with floats
    # may mean that we run into issues later on. We assume the fixation rate
    # never drops below 1e-10.
#     ρ = np.maximum(ρ, 1e-15)
    return ρ

class ModelTypeEGT():
    """This is the schema for an Evolutionary Game Theory model.

    Note: This schema is not enforced and is here purely for documentation
    purposes."""

    def __init__(self,
                 Z: int,  # the size of the population
                 strategy_set: list[str],  # the set of strategies in the model
                 β: Array1D,  # the learning rate
                 payoffs: Array3D,  # the payoffs of the game
                 transition_matrix: Array3D = None,  # the model's transition matrix
                 ergodic: Array2D = None,  # ergodic distribution of the model's markov chain
                 ):
        pass

@multi
def build_transition_matrix(models: dict  # A dictionary that contains the parameters in `ModelTypeEGT`
                            ):
    """Build a transition matrix between all monomorphic states using the
    fermi social learning rule."""
    return models.get('dispatch-type')

@method(build_transition_matrix)
def build_transition_matrix(models: dict  # A dictionary that contains the parameters in `ModelTypeEGT`
                            ):
    """Build a transition matrix between all monomorphic states
    using the fermi social learning rule for each model.    
    """

    Z, S, β = [models[k] for k in ['Z', 'strategy_set', 'β']]
    π = models['payoffs']
    n_models = π.shape[0]
    M = np.zeros((n_models, len(S), len(S)))
    for row_ind, s in enumerate(S):
        for col_ind, sₒ in enumerate(S):
            if row_ind == col_ind:
                M[:, row_ind, row_ind] += 1
                # We calibrate these entries later so rows add up to 1
                continue
            πAA = π[:, row_ind, row_ind]
            πAB = π[:, row_ind, col_ind]
            πBA = π[:, col_ind, row_ind]
            πBB = π[:, col_ind, col_ind]
            ΠA = [πAA*(Z-k-1)/(Z-1) + πAB*k/(Z-1)
                  for k in range(1, Z)]
            ΠB = [πBA*(Z-k)/(Z-1) + πBB*(k-1)/(Z-1)
                  for k in range(1, Z)]
            # We use a numerically stable method to find the fixation rate, ρ.
            # ρ is the probability that mutant B successfully invades A
            ρ = fixation_rate_stable(ΠA, ΠB, β)
            M[:, row_ind, col_ind] = ρ / max(1, len(S)-1)
            M[:, row_ind, row_ind] -= ρ / max(1, len(S)-1)
    return {**models, "transition_matrix": M}

@method(build_transition_matrix, 'unstable')
def build_transition_matrix(models: dict  # A dictionary that contains the parameters in `ModelTypeEGT`
                            ):
    """Build a transition matrix using a numerically unstable method."""

    Z, S, β = [models[k] for k in ['Z', 'strategy_set', 'β']]
    π = models['payoffs']
    n_models = π.shape[0]
    M = np.zeros((n_models, len(S), len(S)))
    for row_ind, s in enumerate(S):
        for col_ind, sₒ in enumerate(S):
            if row_ind == col_ind:
                M[:, row_ind, row_ind] += 1
                # We calibrate these entries later so rows add up to 1
                continue
            πAA = π[:, row_ind, row_ind]
            πAB = π[:, row_ind, col_ind]
            πBA = π[:, col_ind, row_ind]
            πBB = π[:, col_ind, col_ind]
            ΠA = [πAA*(Z-k-1)/(Z-1) + πAB*k/(Z-1)
                  for k in range(1, Z)]
            ΠB = [πBA*(Z-k)/(Z-1) + πBB*(k-1)/(Z-1)
                  for k in range(1, Z)]
            Tneg = [fermi_learning(ΠB[k-1], ΠA[k-1], β)
                    for k in range(1, Z)]
            Tplus = [fermi_learning(ΠA[k-1], ΠB[k-1], β)
                     for k in range(1, Z)]
            ρ = fixation_rate(Tplus, Tneg)
            M[:, row_ind, col_ind] = ρ / max(1, len(S)-1)
            M[:, row_ind, row_ind] -= ρ / max(1, len(S)-1)
    return {**models, "transition_matrix": M}

def find_ergodic_distribution(models: dict  # A dictionary that contains the parameters in `ModelTypeEGT`
                              ):
    """Find the ergodic distribution of a markov chain with the
    given transition matrix."""
    M = models["transition_matrix"]
    # find unit eigenvector of markov chain
    Λ, V = np.linalg.eig(M.transpose(0, 2, 1))
    V = np.real_if_close(V)
    x = np.isclose(Λ, 1)
    # if multiple unit eigenvalues then choose the first
    y = np.zeros_like(x, dtype=bool)
    idx = np.arange(len(x)), x.argmax(axis=1)
    y[idx] = x[idx]
    ergodic = np.array(V.transpose(0, 2, 1)[y], dtype=float)
    # ensure ergodic frequencies are positive and sum to 1
    ergodic = np.abs(ergodic) / np.sum(np.abs(ergodic), axis=1)[:, None]
    return {**models, 'ergodic': ergodic}

@multi
def calculate_stationary_distribution(transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix],
                                      method=None):
    """A multimethod for calculating the stationary distribution of different
    types of matrices."""
    return method

@method(calculate_stationary_distribution)
def calculate_stationary_distribution(transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix], # A single 2D transition matrix or a 3D array containing a stack of transition matrices
                                      method=None # The method to use to find the statonary distribution, the default approach relies on using `numpy.linalg.eig` which is not recommended for non-hermitian matrices. Use "shcur" if matrix is non-hermitian.
                                      ) -> np.ndarray:
    """
    Calculates stationary distribution from a transition matrix of Markov chain.

    The use of this function is not recommended if the matrix is non-Hermitian. Please use
    calculate_stationary_distribution_non_hermitian instead in this case.

    The stationary distribution is the normalized eigenvector associated with the eigenvalue 1

    Parameters
    ----------
    transition_matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        A 2 dimensional transition matrix

    Returns
    -------
    numpy.ndarray
        A 1-dimensional vector containing the stationary distribution

    See Also
    --------
    egttools.utils.calculate_stationary_distribution_non_hermitian

    """
    if (type(transition_matrix) == csr_matrix) or (type(transition_matrix) == csc_matrix):
        tmp = transition_matrix.toarray()
    else:
        tmp = transition_matrix
        
    if np.ndim(transition_matrix)==2:
        tmp = transition_matrix[None, ...]
    
    # Check if there is any transition with value 1 - this would mean that the game is degenerate
    if np.isclose(tmp, 1., atol=1e-11).any():
        warn(
            "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
            "This could result in more than one eigenvalue of magnitute 1 "
            "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)
        
    # `numpy.linalg.eig` returns the right-handed eigenvectors so we need to tranpose our transition matrices first.
    tmp = tmp.transpose(0, 2, 1)

    # calculate stationary distributions using eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(tmp)
    
    # look for the first element closest to 1 in the list of eigenvalues
    index_stationary = (np.arange(len(eigenvalues)),
                        np.argmin(np.abs(eigenvalues - 1.0), axis=-1))
    mask_stationary = np.zeros_like(eigenvalues, dtype=bool)
    mask_stationary[index_stationary] = True
    sd = np.abs(eigenvectors.transpose(0, 2, 1)[mask_stationary].real)
    return sd / np.sum(sd, axis=-1)[:, None]  # normalize

@method(calculate_stationary_distribution, "schur")
def calculate_stationary_distribution(transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix], # A single 2D transition matrix or a 3D array containing a stack of transition matrices
                                      method=None # The method to use to find the statonary distribution, the default approach relies on using `numpy.linalg.eig` which is not recommended for non-hermitian matrices. Use "shcur" if matrix is non-hermitian.
                                      ) -> np.ndarray:
    """
    Calculates stationary distribution from a transition matrix of Markov chain.

    The stationary distribution is the normalized eigenvector associated with the eigenvalue 1

    Parameters
    ----------
    transition_matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        A 2 dimensional transition matrix

    Returns
    -------
    numpy.ndarray
        A 1-dimensional vector containing the stationary distribution

    See Also
    --------
    egttools.utils.calculate_stationary_distribution_non_hermitian

    """
    if (type(transition_matrix) == csr_matrix) or (type(transition_matrix) == csc_matrix):
        tmp = transition_matrix.toarray()
    else:
        tmp = transition_matrix
        
    if np.ndim(transition_matrix)==2:
        tmp = transition_matrix[None, ...]
    
    # Check if there is any transition with value 1 - this would mean that the game is degenerate
    if np.isclose(tmp, 1., atol=1e-11).any():
        warn(
            "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
            "This could result in more than one eigenvalue of magnitute 1 "
            "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)

    # calculate stationary distributions using eigenvalues and eigenvectors
    schur_results = [schur(m) for m in tmp]
    eigenvectors = np.array([r[1] for r in schur_results]).real
    eigenvalues = np.array([eigvals(r[0]) for r in schur_results]).real
    # look for the first element closest to 1 in the list of eigenvalues
    index_stationary = (np.arange(len(eigenvalues)),
                        np.argmin(np.abs(eigenvalues - 1.0), axis=-1))
    mask_stationary = np.zeros_like(eigenvalues, dtype=bool)
    mask_stationary[index_stationary] = True
    sd = np.abs(eigenvectors[mask_stationary].real)
    return sd / np.sum(sd, axis=-1)[:, None]  # normalize

def gth_solve(A, overwrite=False):
    r"""
    This routine computes the stationary distribution of an irreducible
    Markov transition matrix (stochastic matrix) or transition rate
    matrix (generator matrix) `A`.
    More generally, given a Metzler matrix (square matrix whose
    off-diagonal entries are all nonnegative) `A`, this routine solves
    for a nonzero solution `x` to `x (A - D) = 0`, where `D` is the
    diagonal matrix for which the rows of `A - D` sum to zero (i.e.,
    :math:`D_{ii} = \sum_j A_{ij}` for all :math:`i`). One (and only
    one, up to normalization) nonzero solution exists corresponding to
    each reccurent class of `A`, and in particular, if `A` is
    irreducible, there is a unique solution; when there are more than
    one solution, the routine returns the solution that contains in its
    support the first index `i` such that no path connects `i` to any
    index larger than `i`. The solution is normalized so that its 1-norm
    equals one. This routine implements the Grassmann-Taksar-Heyman
    (GTH) algorithm [1]_, a numerically stable variant of Gaussian
    elimination, where only the off-diagonal entries of `A` are used as
    the input data. For a nice exposition of the algorithm, see Stewart
    [2]_, Chapter 10.
    Parameters
    ----------
    A : array_like(float, ndim=2)
        Stochastic matrix or generator matrix. Must be of shape n x n.
    Returns
    -------
    x : numpy.ndarray(float, ndim=1)
        Stationary distribution of `A`.
    overwrite : bool, optional(default=False)
        Whether to overwrite `A`.
    References
    ----------
    .. [1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
       Analysis and Steady State Distributions for Markov Chains,"
       Operations Research (1985), 1107-1116.
    .. [2] W. J. Stewart, Probability, Markov Chains, Queues, and
       Simulation, Princeton University Press, 2009.
    """
    A1 = np.array(A, dtype=float, copy=not overwrite, order='C')
    # `order='C'` is for use with Numba <= 0.18.2
    # See issue github.com/numba/numba/issues/1103

    if len(A1.shape) != 2 or A1.shape[0] != A1.shape[1]:
        raise ValueError('matrix must be square')

    n = A1.shape[0]
    x = np.zeros(n)

    # === Reduction === #
    for k in range(n-1):
        scale = np.sum(A1[k, k+1:n])
        if scale <= 0:
            # There is one (and only one) recurrent class contained in
            # {0, ..., k};
            # compute the solution associated with that recurrent class.
            n = k+1
            break
        A1[k+1:n, k] /= scale

        A1[k+1:n, k+1:n] += np.dot(A1[k+1:n, k:k+1], A1[k:k+1, k+1:n])

    # === Backward substitution === #
    x[n-1] = 1
    for k in range(n-2, -1, -1):
        x[k] = np.dot(x[k+1:n], A1[k+1:n, k])

    # === Normalization === #
    x /= np.sum(x)

    return x

@method(calculate_stationary_distribution, "quantecon")
def calculate_stationary_distribution(transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix], # A single 2D transition matrix or a 3D array containing a stack of transition matrices
                                      method=None # The method to use to find the statonary distribution, the default approach relies on using `numpy.linalg.eig` which is not recommended for non-hermitian matrices. Use "shcur" if matrix is non-hermitian.
                                      ) -> np.ndarray:
    """
    Calculates stationary distribution from a transition matrix of Markov chain.

    The stationary distribution is the normalized eigenvector associated with the eigenvalue 1

    Parameters
    ----------
    transition_matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        A 2 dimensional transition matrix

    Returns
    -------
    numpy.ndarray
        A 1-dimensional vector containing the stationary distribution

    See Also
    --------
    egttools.utils.calculate_stationary_distribution_non_hermitian

    """
    if (type(transition_matrix) == csr_matrix) or (type(transition_matrix) == csc_matrix):
        tmp = transition_matrix.toarray()
    else:
        tmp = transition_matrix
        
    if np.ndim(transition_matrix)==2:
        tmp = transition_matrix[None, ...]
    
    # Check if there is any transition with value 1 - this would mean that the game is degenerate
    if np.isclose(tmp, 1., atol=1e-11).any():
        warn(
            "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
            "This could result in more than one eigenvalue of magnitute 1 "
            "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)
    return np.array([gth_solve(p) for p in tmp])

def calculate_sd_helper(models):
    P =  models['transition_matrix']
    sd = calculate_stationary_distribution(P, method=models.get('sd-method'))
    return {**models, "ergodic": sd }

def markov_chain(models: dict  # A dictionary that contains the parameters in `ModelTypeEGT`
                 ):
    """Find the ergodic distribution of the evolutionary
    game given by each model in models."""
    return thread_macro(models,
                        build_transition_matrix,
                        find_ergodic_distribution)

@multi
def sample_profile(models):
    return models.get('sample_profile_key')

@method(sample_profile)
def sample_profile(models):
    """We sample players from their allowed sectors as per the sector weights."""
    sector_strategies = models['sector_strategies']
    allowed_sectors = models['allowed_sectors']
    profile = models['profile']
    chosen_player = models['chosen_player']
    chosen_strategy = int(models['chosen_strategy'])
    current_strategy = int(models['current_strategy'])
    mutant_strategy = int(models['mutant_strategy'])
    affected_sector = models['affected_sector']
    n_mutants = models['n_mutants']
    Z = models['Z']
    sector_weights = models.get('sector_weights', {})
    assert n_mutants >= 1
    assert n_mutants <= Z[affected_sector] - 1
    assert current_strategy != mutant_strategy
    assert chosen_strategy in [current_strategy, mutant_strategy]
    assert affected_sector in sector_strategies.keys()
    assert chosen_player in allowed_sectors.keys()

    profile_tuple = list(map(int, profile.split("-")))
    assert chosen_strategy in profile_tuple

    # TODO: does it make sense for chosen_player_likelihood to take into account
    # any possible position our chosen_player could have been in, no matter
    # which strategies each player actually plays in the profile?
    above = sector_weights.get(chosen_player, {}).get(affected_sector, 1)
    below = 0
    for player, sectors in allowed_sectors.items():
        if affected_sector in sectors:
            below += sector_weights.get(player, {}).get(affected_sector, 1)
    if below == 0:
        raise ValueError("""affected_sector is never allowed in the game, 
                         double check allowed_sectors""")
    # print("chosen_player_likelihood: ", above / below)
    chosen_player_likelihood = above / below

    likelihood = chosen_player_likelihood

    n_sampled_from_affected_sector = 1
    n_mutants_sampled = 1 if chosen_strategy == mutant_strategy else 0
    for i, strategy in enumerate(profile_tuple[::-1]):
        valid_strategy = False
        for sector in allowed_sectors[f"P{i+1}"]:
            if strategy in map(int, sector_strategies[sector]):
                valid_strategy = True
        if not valid_strategy:
            raise ValueError(f"""Profile, {profile}, implies a player plays a
                             strategy from a sector they do not belong to.""")
        if f"P{i+1}" == chosen_player:
            continue
        if strategy in map(int, sector_strategies[affected_sector]):
            if strategy == current_strategy:
                likelihood *= ((Z[affected_sector]
                                - (n_sampled_from_affected_sector
                                   - n_mutants_sampled)
                                - n_mutants)
                               / (Z[affected_sector]
                                  - n_sampled_from_affected_sector))
                # print("current-lk: ",
                #       (Z[affected_sector]
                #        - (n_sampled_from_affected_sector - n_mutants_sampled)
                #        - n_mutants),
                #       "/",
                #       (Z[affected_sector] - n_sampled_from_affected_sector))
                n_sampled_from_affected_sector += 1
            elif strategy == mutant_strategy:
                likelihood *= ((n_mutants
                                - n_mutants_sampled)
                               / (Z[affected_sector]
                                  - n_sampled_from_affected_sector))
                # print("mutant-lk: ",
                #       (n_mutants - n_mutants_sampled),
                #       "/",
                #       (Z[affected_sector] - n_sampled_from_affected_sector))
                n_mutants_sampled += 1
                n_sampled_from_affected_sector += 1
            else:
                raise ValueError("""At least one profile implies the copresence
                                 of 3 strategies for one sector. This default
                                 profile likelihood method is not meant to be
                                 used for such situations. Make sure you use
                                 profile filters to prevent passing such
                                 strategy profiles to this sampling rule.""")
        relevant_sector = [sector
                           for sector in sector_strategies.keys()
                           if strategy in map(int, sector_strategies[sector])]
        if len(relevant_sector) > 1:
            raise ValueError("Each sector must have unique strategy codes")
        elif len(relevant_sector) == 0:
            raise ValueError("Strategy does not belong to any sector")
        else:
            relevant_sector = relevant_sector[0]
        above = sector_weights.get("P{i+1}", {}).get(relevant_sector, 1)
        below = sum(sector_weights.get("P{i+1}", {}).get(sector, 1)
                    for sector in allowed_sectors[f"P{i+1}"])
        likelihood *= above / below
        # print("relevant_sector_likelihood: ", above / below)
    return likelihood

def create_recurrent_states(models):
    """Create all recurrent-states for the set of models."""
    sector_strategies = models['sector_strategies']
    n_states = np.prod([len(S) for S in sector_strategies.values()])
    sorted_keys = sorted(sector_strategies, reverse=True)
    strategy_axes = [sector_strategies[k] for k in sorted_keys]
    grid = build_grid_from_axes(strategy_axes)
    states = ["-".join(map(str, row)) for row in grid]
    assert len(states) == n_states
    return states

def valid_transition(ind1: str,  # The index of the current state, expressed in the form "{strategy_code}-{strategy_code}-{strategy_code}"
                     ind2: str,  # The index of the next state, expressed in the same form as `ind1`
                     ) -> bool:  # True if the transition is valid, false otherwise
    """Check if the transition from ind1->ind2 is valid
    i.e. that only one population undergoes a change in strategy."""
    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    valid = sum(differ) == 1
    return valid

@multi
def compute_profile_dist(models):
    """Compute the probability distribution of the relevant profiles."""
    return models.get('profile_dist_rule')

@method(compute_profile_dist)
def compute_profile_dist(models):
    """Compute the probability distribution of the relevant profiles - default."""
    chosen_strategy = models['chosen_strategy']
    profiles = models['profiles_filtered']
    profile_distribution = {}
    for profile in profiles:
        profile_tuple = list(map(int, profile.split("-")))
        possible_players = [f"P{i+1}"
                            for i, strategy in enumerate(profile_tuple[::-1])
                            if strategy == chosen_strategy]
        profile_distribution[profile] = {}
        for chosen_player in possible_players:
            likelihood = sample_profile({**models,
                                         "profile": profile,
                                         "chosen_player": chosen_player})
            profile_distribution[profile][chosen_player] = likelihood
    return profile_distribution

@method(compute_profile_dist, 'multi-player-symmetric')
def compute_profile_dist(models):
    """Compute the probability distribution of the relevant profiles - we have
    one profile per combination of players and only compute the likelihood for
    the relevant player type."""
    chosen_strategy = models['chosen_strategy']
    profiles = models['profiles_filtered']
    profile_distribution = {}
    counter = collections.Counter()
    visited = collections.defaultdict()
    for profile in profiles:
        profile_tuple = list(map(int, profile.split("-")))
        unique, counts = np.unique(profile_tuple, return_counts=True)
        counter_key = "-".join([f"{u}:{c}" for u, c in zip(unique, counts)])
        counter[counter_key] += 1
        visited[counter_key] = False
    for profile in profiles:
        profile_tuple = list(map(int, profile.split("-")))
        unique, counts = np.unique(profile_tuple, return_counts=True)
        counter_key = "-".join([f"{u}:{c}" for u, c in zip(unique, counts)])
        if visited[counter_key]:
            # If we have seen a profile with the same strategy counts, skip it
            continue
        else:
            visited[counter_key] = True
            possible_players = [f"P{i+1}"
                                for i, strategy in enumerate(profile_tuple[::-1])
                                if strategy == chosen_strategy]
            profile_distribution[profile] = {}
            if len(possible_players) > 0:
                # Player order does not matter
                chosen_player = possible_players[0]
                likelihood = sample_profile({**models,
                                             "profile": profile,
                                             "chosen_player": chosen_player})
                # We must multiply the above likelihood by the number of ways
                # this combination of players can be permuted.
                likelihood *= counter[counter_key] * len(possible_players)
                profile_distribution[profile][chosen_player] = likelihood
    return profile_distribution

@multi
def compute_success(models):
    """Compute the success of the two strategies under consideration."""
    return models.get('compute_success_rule', "cheap")

@method(compute_success)
def compute_success(models):
    """Compute the success of the two strategies under consideration for each
    number of k mutants implied by the transition."""
    models = apply_profile_filters(models)
    ind1, ind2 = models['transition_indices']
    sector_strategies = models['sector_strategies']
    Z = models['Z']
    payoffs = models['payoffs']

    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
    current_strategy = ind1_tuple[np.argmax(differ)]
    mutant_strategy = ind2_tuple[np.argmax(differ)]

    ΠA = []
    ΠB = []
    for n_mutants in range(1, Z[affected_sector]):
        dist1 = compute_profile_dist({**models,
                                      'chosen_strategy': current_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        dist2 = compute_profile_dist({**models,
                                      'chosen_strategy': mutant_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        success_A = 0
        for profile, player_map in dist1.items():
            for player, likelihood in player_map.items():
                success_A += payoffs[profile][player] * likelihood
        ΠA.append(success_A)
        success_B = 0
        for profile, player_map in dist2.items():
            for player, likelihood in player_map.items():
                success_B += payoffs[profile][player] * likelihood
        ΠB.append(success_B)
    return ΠA, ΠB

@method(compute_success, "cheap")
def compute_success(models):
    """Compute the success of the two strategies under consideration for each
    number of k mutants implied by the transition."""
    models = apply_profile_filters(models)
    ind1, ind2 = models['transition_indices']
    sector_strategies = models['sector_strategies']
    Z = models['Z']
    payoffs = models['payoffs']

    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
    current_strategy = ind1_tuple[np.argmax(differ)]
    mutant_strategy = ind2_tuple[np.argmax(differ)]

    ΠA = []
    ΠB = []
    for n_mutants in range(1, Z[affected_sector]):
        dist1 = compute_profile_dist({**models,
                                      'chosen_strategy': current_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        dist2 = compute_profile_dist({**models,
                                      'chosen_strategy': mutant_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        payoffsA = []
        likelihoodsA = []
        for profile, player_map in dist1.items():
            for player, likelihood in player_map.items():
                payoffsA.append(payoffs[profile][player])
                likelihoodsA.append(likelihood)
        ΠA.append(np.dot(np.array(payoffsA).T, likelihoodsA))
        payoffsB = []
        likelihoodsB = []
        for profile, player_map in dist2.items():
            for player, likelihood in player_map.items():
                payoffsB.append(payoffs[profile][player]) 
                likelihoodsB.append(likelihood)
        ΠB.append(np.dot(np.array(payoffsB).T, likelihoodsB))
    return ΠA, ΠB


@method(compute_success, "functional")
def compute_success(models):
    """Compute the success of the two strategies under consideration for each
    number of k mutants implied by the transition."""
    models = apply_profile_filters(models)
    ind1, ind2 = models['transition_indices']
    sector_strategies = models['sector_strategies']
    Z = models['Z']
    payoffs_function = models['payoffs_function']

    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
    current_strategy = ind1_tuple[np.argmax(differ)]
    mutant_strategy = ind2_tuple[np.argmax(differ)]

    ΠA = []
    ΠB = []
    for n_mutants in range(1, Z[affected_sector]):
        dist1 = compute_profile_dist({**models,
                                      'chosen_strategy': current_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        dist2 = compute_profile_dist({**models,
                                      'chosen_strategy': mutant_strategy,
                                      'current_strategy': current_strategy,
                                      'mutant_strategy': mutant_strategy,
                                      'affected_sector': affected_sector,
                                      'n_mutants': n_mutants})
        # Record the strategy counts in the population implied by the number of
        # mutants so that our payoff function can make use of this information
        strategy_counts = {str(strategy): Z[sector]
                           for strategy in ind1_tuple
                           for sector in sector_strategies.keys()
                           if strategy in sector_strategies[sector]}
        strategy_counts = {**strategy_counts,
                           str(current_strategy): Z[affected_sector] - n_mutants,
                           str(mutant_strategy): n_mutants}
        population_state = models.get("population_state", {})
        population_state = {**population_state,
                            "strategy_counts": strategy_counts}
        # Compute the payoffs for each strategy in each possible profile
        # and the likelihood of that profile occuring.
        payoffsA = []
        likelihoodsA = []
        for profile, player_map in dist1.items():
            for player, likelihood in player_map.items():
                payoffsA.append(payoffs_function({**models,
                                                  "population_state": population_state,
                                                  "profile": profile,
                                                  "player": player}))
                likelihoodsA.append(likelihood)
        ΠA.append(np.dot(np.array(payoffsA).T, likelihoodsA))
        payoffsB = []
        likelihoodsB = []
        for profile, player_map in dist2.items():
            for player, likelihood in player_map.items():
                payoffsB.append(payoffs_function({**models,
                                                  "population_state": population_state,
                                                  "profile": profile,
                                                  "player": player}))
                likelihoodsB.append(likelihood)
        ΠB.append(np.dot(np.array(payoffsB).T, likelihoodsB))
    return ΠA, ΠB

def vals(d: dict):
    "Return the values of a dictionary."
    return d.values()

def infer_n_models(models):
    "Infer the number of models from the model payoffs."
    try:
        payoffs = models.get('payoffs')
        payoff_vector = thread_macro(payoffs,
                                     vals,
                                     iter,
                                     next,
                                     vals,
                                     iter,
                                     next)
        n_models = (1
                    if isinstance(payoff_vector, (float, int))
                    else len(payoff_vector))
    except:
        raise ValueError("""Unable to infer `n_models`.
                         `payoffs` is structured incorrectly""")
    return n_models

@method(build_transition_matrix, 'multiple-populations')
def build_transition_matrix(models: dict  # A dictionary that contains the parameters in `ModelTypeEGTMultiple`
                            ):
    """Build a transition matrix between all monomorphic states
    when there are multiple populations.    
    """
    β = models['β']
    n_models = models.get('n_models',
                          infer_n_models(models))
    S = models.get('recurrent_states',
                   create_recurrent_states(models))
    M = np.zeros((n_models, len(S), len(S)))
    for row_ind in range(M.shape[-1]):
        M[:, row_ind, row_ind] += 1
    transition_inds = [(i, j)
                       for i in range(len(S))
                       for j in range(len(S))]
    for row_ind, col_ind in transition_inds:
        current_state, new_state = S[row_ind], S[col_ind]
        if current_state == new_state:
            continue
        if not valid_transition(current_state, new_state):
            continue
        ΠA, ΠB = compute_success(assoc(models,
                                       "transition_indices",
                                       [current_state, new_state]))
        # TODO: Clean up assymetric beta code below
        ind1_tuple = list(map(int, current_state.split("-")))
        ind2_tuple = list(map(int, new_state.split("-")))
        differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
        affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
        if isinstance(β, dict):
            ρ = fixation_rate_stable(ΠA, ΠB, β[affected_sector])
        else:
            ρ = fixation_rate_stable(ΠA, ΠB, β)
        n_mutations = sum(valid_transition(current_state, s_alt)
                          for s_alt in S)
        M[:, row_ind, col_ind] = ρ / n_mutations
        M[:, row_ind, row_ind] -= ρ / n_mutations
    return {**models, 'transition_matrix': M,
            'recurrent_states': S,
            'n_models': n_models}

@multi
def compute_success_analytical(models):
    return models.get('success_analytical_derivation')

@method(compute_success_analytical, '2sector2strategy2player')
def compute_success_analytical(models):
    """Compute the success of each strategy involved in a transition for
    a game with two players who can each be from one of two sectors. Each
    sector uses two strategies."""

    # 50% chance of facing fixed strategy no matter if we look at mutant or
    # current strategy
    # Otherwise, we have a 0.5 * (n_mutants / z_s1) chance of facing strategy 2
    # and a 0.5 * ((z_s1 - n_mutants) / z_s1) chance of facing strategy 1.
    # For each of theses there is a 50% of being player 1 or player 2.
    transition_indices = models['transition_indices']
    Z = models['Z']
    payoffs = models['payoffs']
    ind1, ind2 = transition_indices
    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
    current_strategy = ind1_tuple[np.argmax(differ)]
    mutant_strategy = ind2_tuple[np.argmax(differ)]
    # We only support two sectors here, so we know the other value must be the
    # fixed sector.
    fs = ind1_tuple[np.argmin(differ)]
    cs = current_strategy
    ms = mutant_strategy
    SA, SB = [], []
    z = Z[affected_sector]
    for k in range(1, z):
        successA = ((0.5
                    * k / (z - 1)
                    * (payoffs[f"{ms}-{cs}"]['P1']
                        + payoffs[f"{cs}-{ms}"]['P2']) / 2
                     )
                    + (0.5
                        * (z - 1 - k) / (z - 1)
                        * (payoffs[f"{cs}-{cs}"]['P1']
                           + payoffs[f"{cs}-{cs}"]['P2']) / 2
                       )
                    + (0.5
                        * (payoffs[f"{fs}-{cs}"]['P1']
                           + payoffs[f"{cs}-{fs}"]['P2']) / 2
                       )
                    )
        SA.append(successA)
        successB = ((0.5
                     * (k - 1) / (z - 1)
                     * (payoffs[f"{ms}-{ms}"]['P1']
                         + payoffs[f"{ms}-{ms}"]['P2']) / 2
                     )
                    + (0.5
                       * (z - k) / (z - 1)
                       * (payoffs[f"{cs}-{ms}"]['P1']
                           + payoffs[f"{ms}-{cs}"]['P2']) / 2
                       )
                    + (0.5
                       * (payoffs[f"{fs}-{ms}"]['P1']
                          + payoffs[f"{ms}-{fs}"]['P2']) / 2
                       )
                    )
        SB.append(successB)
    return SA, SB

@method(compute_success_analytical, '2sector2strategy3player')
def compute_success_analytical(models):
    """Compute the success of each strategy involved in a transition for
    a game with three players where two players are from one sector and the
    third is from another. Each sector uses two strategies."""

    # 100% chance of third player playing either their chosen strategy if they
    # are the chosen agent, or otherwise 100% chance of third player playing
    # their current strategy.
    # If chosen agent in first sector, we have a (n_mutants / z_s1) chance of
    # facing strategy 2 and a ((z_s1 - n_mutants) / z_s1) chance of facing
    # strategy 1.
    # For each of these, there is a 50% chance of being player 1 or player 2.
    # If chosen player is the third player, then they have a 100% of facing
    # against the fixed strategy played by the other two players.
    transition_indices = models['transition_indices']
    Z = models['Z']
    payoffs = models['payoffs']
    allowed_sectors = models['allowed_sectors']
    ind1, ind2 = transition_indices
    ind1_tuple = list(map(int, ind1.split("-")))
    ind2_tuple = list(map(int, ind2.split("-")))
    differ = [i1 != i2 for i1, i2 in zip(ind1_tuple, ind2_tuple)]
    affected_sector = f"S{np.argmax(differ[::-1]) + 1}"
    current_strategy = ind1_tuple[np.argmax(differ)]
    mutant_strategy = ind2_tuple[np.argmax(differ)]
    # We only support two sectors here, so we know the other value must be the
    # fixed sector.
    fs = ind1_tuple[np.argmin(differ)]
    cs = current_strategy
    ms = mutant_strategy
    SA, SB = [], []
    z = Z[affected_sector]
    relevant_players = [p
                        for p in allowed_sectors.keys()
                        if affected_sector in allowed_sectors[p]]
    if len(relevant_players) == 1:
        for k in range(1, z):
            SA.append(payoffs[f"{cs}-{fs}-{fs}"]['P3'])
            SB.append(payoffs[f"{ms}-{fs}-{fs}"]['P3'])
    elif len(relevant_players) == 2:
        for k in range(1, z):
            successA = ((k / (z - 1)
                        * (payoffs[f"{fs}-{ms}-{cs}"]['P1']
                            + payoffs[f"{fs}-{cs}-{ms}"]['P2']) / 2
                         )
                        + ((z - 1 - k) / (z - 1)
                            * (payoffs[f"{fs}-{cs}-{cs}"]['P1']
                               + payoffs[f"{fs}-{cs}-{cs}"]['P2']) / 2
                           )
                        )
            SA.append(successA)
            successB = (((k - 1) / (z - 1)
                        * (payoffs[f"{fs}-{ms}-{ms}"]['P1']
                            + payoffs[f"{fs}-{ms}-{ms}"]['P2']) / 2
                         )
                        + ((z - k) / (z - 1)
                        * (payoffs[f"{fs}-{cs}-{ms}"]['P1']
                            + payoffs[f"{fs}-{ms}-{cs}"]['P2']) / 2
                           )
                        )
            SB.append(successB)
    return SA, SB
