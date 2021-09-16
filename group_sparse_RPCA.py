import numpy as np
from numpy import linalg as LA
from utilities import maxWithIdx, multiMatmul
import time


def block_shrinkage_operator(G, blocks_by_frame, lambdas_by_frame, mu, large_val=1e7):
    """
    G[pixel, frame] - input matrix
    blocks_by_frame[frame, group, mask_pixel] - binary column masks
    lambdas_by_frame[frame, group] - lambda for each block
    mu - number
    """
    result = np.full_like(G, large_val)  # arbitrarily large value for anything outside of the groups
    for frame_idx in range(len(blocks_by_frame)):  # run over frames
        blocks = blocks_by_frame[frame_idx]  # get the blocks of that frame
        lambdas = lambdas_by_frame[frame_idx]

        for block_idx in range(len(blocks)):  # run over blocks
            # assuming n overlap between groups
            block = blocks[block_idx]
            epsilon = lambdas[block_idx] / mu

            block_in_G = G[block, frame_idx]
            result[block, frame_idx] = np.maximum(1 - epsilon / LA.norm(block_in_G, ord=2), 0) * block_in_G

    return result


def inexact_alm_group_sparse_RPCA(D0):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    m, n = D.shape
    d = np.min(D.shape)

    delta = 10
    lambda_param = (np.sqrt(np.max((m, n))) * delta) ** (-1)

    # initialize
    Y = D
    norm_two = LA.norm(Y, ord=2)
    norm_inf = LA.norm(Y, ord=np.inf) / lambda_param
    dual_norm = np.max((norm_two, norm_inf))
    Y = Y / dual_norm

    mu = 1.25 / norm_two  # can be tuned
    rho = 1.6
    tol_out = 1e-7

    # L = np.zeros(D.shape, order='F') if L0 is None else L0
    S = np.zeros(D.shape, order='F')

    converged = False
    iter_out = 0
    sv = 10  # sv0

    while not converged:  # Algorithm line 2
        iter_out += 1

        # SOLVE FOR L
        G_L = D - S + Y / mu  # Algorithm line 4

        # matlab algorithm add another condition here (choosvd)
        u, s, vh = LA.svd(G_L, full_matrices=False)
        s = s[0:sv]

        # soft-thresholding
        last_nonzero_sv_idx = np.max(np.nonzero(s - 1 / mu > 0))
        svp = last_nonzero_sv_idx + 1  # svp = # of s.v that are bigger than 1/mu

        # predicting the number of s.v bigger than 1/mu
        ratio = s[:-1] / s[1:]
        max_ratio, max_idx = maxWithIdx(ratio)
        svn = svp if max_ratio <= 2 else min(svp, max_idx + 1)

        # matlab used round(0.05 * d) instead of 10
        sv = svn + 1 if svn < sv else min(svn + 10, d)

        L = multiMatmul(u[:, :svp], np.diag(s[:svp] - 1 / mu, 0), vh[:svp, :], order='F')  # Algorithm line 5

        # SOLVE FOR S
        G_S = D - L + Y / mu  # Algorithm line 7
        S = prox(G_S, lambda_param / mu, graph)  # Algorithm line 8

        # UPDATE Y, mu
        Z = D - L - S
        Y = Y + mu * Z  # Algorithm line 9
        mu = min(mu * rho, mu * 1e7)  # Algorithm line 10 (+limit max mu)

        # check error and convergence
        err = LA.norm(Z, ord='fro') / LA.norm(D, ord='fro')

        # print iteration info
        print(f'Iteration: {iter_out:3d} rank(L): {svp:2d} ||S||_0: {LA.norm(S.flat, ord=0):.2E} err: {err:.3E}')

        if err < tol_out:
            print('CONVERGED')
            converged = True

    return L, S, iter_out


def check_BS_operator():
    frame_size, frames = (5, 3)
    G = np.array(range(frame_size * frames)).reshape((frame_size, frames), order='F')
    np.random.shuffle(G)
    print(G)

    blocks_by_frame = [
        [[0, 1], [3, 4]],
        [[0]],
        [[1], [2, 3], [4]]
    ]
    print(blocks_by_frame)

    lambdas_by_frame = [
        [1, 2],
        [3],
        [4, 5, 6]
    ]
    print(lambdas_by_frame)

    mu = 1000

    block_shrinkage_result = block_shrinkage_operator(G, blocks_by_frame, lambdas_by_frame, mu, large_val=100)
    print(block_shrinkage_result)

def main():
    check_BS_operator()


if __name__ == '__main__':
    print('START')
    start = time.time()
    main()
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end-start):.3f} seconds')