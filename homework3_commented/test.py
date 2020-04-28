import numpy as np
# print(np.arange(10))
# _direction_deltas = [
    # (-1, 0),
    # (0, 1),
    # (1, 0),
    # (0, -1),
# ]
# _num_actions = len(_direction_deltas)
# print(_num_actions)
# for action in range(_num_actions):
    # print(action)
# action_probabilities=[
        # (-1, 0.1),
        # (0, 0.8),
        # (1, 0.1),
# ]

# for offset, P in action_probabilities:
    # print(offset)
    # print('**')
    # print(P)
    # print('**')
# print(-1%4)
# out: -1
# **
# 0.1
# **
# 0
# **
# 0.8
# **
# 1
# **
# 0.1
# **
# 3
# if __name__ == '__main__':
    # shape = (6, 8)
    # goal = (5, -1)
    # trap1 = (1, -1)
    # trap2 = (4, 1)
    # trap3 = (4, 2)
    # trap4 = (4, 3)
    # trap5 = (4, 4)
    # trap6 = (4, 5)
    # trap7 = (4, 6)
    # obstacle1 = (1, 1)
    # obstacle2 = (0, 5)
    # obstacle3 = (2, 3)
    # obstacle4 = (3, 5)
    # start = (2, 0)
    # default_reward = -0.1
    # goal_reward = 1
    # trap_reward = -1

    # reward_grid = np.zeros(shape) + default_reward
    # reward_grid[goal] = goal_reward
    # reward_grid[trap1] = trap_reward
    # reward_grid[trap2] = trap_reward
    # reward_grid[trap3] = trap_reward
    # reward_grid[trap4] = trap_reward
    # reward_grid[trap5] = trap_reward
    # reward_grid[trap6] = trap_reward
    # reward_grid[trap7] = trap_reward
    # reward_grid[obstacle1] = 0
    # reward_grid[obstacle2] = 0
    # reward_grid[obstacle3] = 0
    # reward_grid[obstacle4] = 0
    # M, N = reward_grid.shape
    # print(M)
    # print(N)

indices = np.arange(48)
r0, c0 = np.unravel_index(indices, (6, 8))
dr = 0
dc = -1
T = np.zeros((6, 8, 4, 6, 8))
r1 = np.clip(r0 + dr, 0, 5)
c1 = np.clip(c0 + dc, 0, 7)
print(r1)
print('***\n')
print(c1)
# print(r0)
# print('***\n')
# print(c0)
# print('***\n')
# print(T[r0, c0, :, r0, c0])
# T[r0, c0, :, r0, c0] += 1
# print(T[r0, c0, :, r0, c0])

