dataset = pd.DataFrame([[2,20,0,0],
                        [2,20,0,10],
                        [10,20,10,10],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [2,20,10,5],
                        [1,50,5,2],
                        [1,50,5,2],
                        [1,50,5,2],
                        [1,50,5,2],
                        [20,10,5,2], #step change
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [20,40,5,2],
                        [26,16,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [25,15,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [20,10,5,2],
                        [25,15,5,2],
                        [20,10,5,2],
                        [20,10,5,2]
                       ] 
                      )


tau = 10
n_arms = 2
last_choices = np.full(tau,-1)
last_rewards = np.zeros(tau)
alpha = 1

choices = []
rewards = dataset.iloc[:,0:2]
last_rewards[0] = rewards.iloc[0,0]
last_rewards[1] = rewards.iloc[1,1]
last_choices[0] = 0 #arm 0 first
choices.append(0)
last_choices[1] = 1 #arm 1 second
choices.append(1)
t = 2

def compute_index(arm, last_rewards, last_choices, alpha, t, tau):
    last_pulls_of_this_arm = np.count_nonzero(last_choices == arm)
    if last_pulls_of_this_arm < 1:
        return float('+inf')
    else:
        return (np.sum(last_rewards[last_choices == arm]) / last_pulls_of_this_arm) + np.sqrt((alpha * np.log(min(t, tau))) / last_pulls_of_this_arm)
    
  
for t in range(2, len(rewards)):
    now = t % tau
    val = 0
    for arm in range(0,n_arms):
        val_c = compute_index(arm,last_rewards, last_choices,1,t,tau)
        if(val_c > val):
            arm_choice = arm
            val = val_c
    last_rewards[now] = rewards.iloc[t, arm_choice]
    last_choices[now] = arm_choice
    choices.append(arm_choice)

fig_rewards = rewards.plot()
fig_rewards.set_title('Test Data Non-Stationary Rewards')
fig_rewards.set_xlabel('Time')
fig_rewards.set_ylabel('Reward')

aa = rewards.iloc[:,1] > rewards.iloc[:,0]
fig1 = aa.astype(float).plot()
fig1.set_title('Ideal Arm Choice to mimize regret')
fig1.set_xlabel('Time')
fig1.set_ylabel('Arm')

choices_df = pd.DataFrame(choices)
fig_ucb = choices_df.plot()
fig_ucb.set_title('UCB Arm Choice')
fig_ucb.set_xlabel('Time')
fig_ucb.set_ylabel('Arm')




