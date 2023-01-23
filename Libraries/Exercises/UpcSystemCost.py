#!/usr/bin/python

import numpy as np

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]

    return (path[::-1], cost_mat)


def get_cost_pose_user(df_trainer_coords, results_frame, n_pose):
    results_array = []
    for i in range(0, len(results_frame.pose_landmarks.landmark)):
        results_array.append(results_frame.pose_landmarks.landmark[i].x)
        results_array.append(results_frame.pose_landmarks.landmark[i].y)
        results_array.append(results_frame.pose_landmarks.landmark[i].z)
        results_array.append(results_frame.pose_landmarks.landmark[i].visibility)
    
    user_array = results_array
    results_costs = []
    trainer_array = []

    for i in df_trainer_coords.columns:
        #ct = datetime.datetime.now()
        #print(str(ct) + " Evaluating the position: " + str(n_pose))
        trainer_array.append(df_trainer_coords[i][n_pose])

    x = np.array(user_array) 
    y = np.array(trainer_array)

    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])
        # DTW
    path, cost_mat = dp(dist_mat)
    x_path, y_path = zip(*path)
    results_costs.append([n_pose,cost_mat[N - 1, M - 1],cost_mat[N - 1, M - 1]/(N + M)])

    return round(results_costs[0][1], 4)
