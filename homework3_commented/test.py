import numpy as np
def _init_utility_policy_storage(depth):
    M, N = (6, 8)
    utility_grids = np.zeros((M, N, depth))
    policy_grids = np.zeros_like(utility_grids)
    return utility_grids, policy_grids
utility_grids, policy_grids =_init_utility_policy_storage(10)
print(utility_grids)
print('***\n')
print(policy_grids)
