'''
#################################################################################################
# author wudong
# date 20190805
# 功能 
#   利用迭代法实现动态规划的
#   policy evaluation algorithm

    示例问题——方格世界 david silver的RL课程中leature3的small gird world example
    https://zhuanlan.zhihu.com/p/28084955
    已知：
    状态空间S：如图。S1 - S14非终止状态，ST终止状态，下图灰色方格所示两个位置；
    行为空间A：{n, e, s, w} 对于任何非终止状态可以有东南西北移动四个行为；
    转移概率P：任何试图离开方格世界的动作其位置将不会发生改变，其余条件下将100%地转移到动作指向的状态；
    即时奖励R：任何在非终止状态间的转移得到的即时奖励均为-1，进入终止状态即时奖励为0；
    衰减系数γ：1；
    当前策略π：Agent采用随机行动策略，在任何一个非终止状态下有均等的几率采取任一移动方向这个行为，
        即π(n|•) = π(e|•) = π(s|•) = π(w|•) = 1/4。
    问题：评估在这个方格世界里给定的策略。
    该问题等同于：求解该方格世界在给定策略下的（状态）价值函数，
        也就是求解在给定策略下，该方格世界里每一个状态的价值。
#################################################################################################
'''

# 声明states
states = [i for i in range(16)]
# 初始化states的values
values = [0 for _ in range(16)]
# 声明action space
actions = ["n", "e", "s", "w"]
# 使用字典，方便计算动作后的states
ds_actions = {"n":-4,"e":1,"s":4,"w":-1}
# 衰减因子
gamma = 1.00

# 根据当前的state和action计算下一个状态id以及即时奖励R
def nextState(s,a):
    next_state = s
    # 判断试图离开方格世界的动作
    if(s%4==0 and a=="w") or (s<4 and a=="n") or ((s+1)%4==0 and a=="e") or (s>11 and a=="s"):
        pass
    else:
        ds = ds_actions[a]
        next_state = next_state + ds
    return next_state

# 得到某个state的奖励reward R
def rewardOf(s):
    return 0 if s in [0,15] else -1

# 判断state是否为终止状态
def isTerminateState(s):
    return s in [0,15]

# 获取某一状态可能的后继状态
def getSuccessors(s):
    successors = []
    if isTerminateState(s):
        return successors
    for a in actions:
        next_state = nextState(s,a)
        successors.append(next_state)
    return successors

# 更新state s的value
def updateValue(s):
    sucessors = getSuccessors(s)
    newValue = 0
    num = 4 # 后继状态，1个状态有4个可能的动作
    reward = rewardOf(s) # 得到状态的即时奖励
    for next_state in sucessors:
        newValue += 1.00/num * (reward + gamma * values[next_state]) # Iterative policy evalution
    return newValue

# peform one-step iteration 一次迭代，一个eposide
# 一次迭代中的整个eposide完成后，才会整体更新value
def performOneIteration():
    newValues = [0 for _ in range(16)]
    for s in states:
        newValues[s] = updateValue(s)
    global values
    values = newValues
    printValue(values)

# 打印网格的value
def printValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]),end = " ")
        if(i+1)%4 == 0:
            print("")
    print()

def main():
    max_iterate_times = 160
    cur_iterate_times = 0
    while cur_iterate_times <= max_iterate_times: #迭代
        print("Iterate No.%a"% cur_iterate_times)
        performOneIteration()
        cur_iterate_times = cur_iterate_times +1
    printValue(values)

if __name__ == "__main__":
    main()