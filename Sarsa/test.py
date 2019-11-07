from gridworld import GridWorldEnv
from gym import spaces

env = GridWorldEnv(n_width=12, # 水平方向格子数量
                    n_height=4, # 垂直方向格子数量
                    u_size=60, #格子的尺寸
                    default_reward=-1, #默认的即时奖励
                    default_type=0) #默认的格子是可进入的
env.action_space = spaces.Discrete(4) # 设置action的数量
# 格子世界环境默认使用0表示左，1右，2上，3下，4567斜向
# 格子世界的观测空间自动计算
env.start = (0,0)
env.ends=[(11,0)]
for i in range(10):
    env.rewards.append((i+1,0,-100))
    env.ends.append((i+1,0))
env.types = [(5,1,1),(5,2,1)]
env.refresh_setting()
env.reset()
env.render()
input("输入任意键继续........")