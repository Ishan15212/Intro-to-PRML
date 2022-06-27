
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500


Q_reward = -100000*numpy.zeros((500,6))
print(Q_reward.shape)

action = random.randint(0,3)

    
for episodesCount in range(0,num_of_episodes):
    stateBeforeAction = env.reset();
    env.render();
    # action = random.randint(0,3);
    actionLoop = numpy.argmax(Q_reward[stateBeforeAction,:])
    for stateLoop in range(0,num_of_steps):
        
        # stateBeforeAction, rewardBeforeAction, doneBeforeAction, infoBeforeAction = env.step(action);
        # for actionLoop in range(0,5):
        getOldQ=(1-alpha)*Q_reward[stateBeforeAction,actionLoop];
        
        stateAfterAction, rewardAfterAction, doneAfterAction, infoAfterAction = env.step(actionLoop);
        
        env.render();
        
        getLearnedValue=alpha*(rewardAfterAction+(gamma*(numpy.max(Q_reward[stateAfterAction,:]))))
        
        getNewQvalue=getOldQ+getLearnedValue;
        
        print("old value"+str(Q_reward[stateBeforeAction,actionLoop]))
        Q_reward[stateBeforeAction,actionLoop]=getNewQvalue;
        
        
        print("stateBeforeAction"+str(stateBeforeAction))
        print("stateAfterAction"+str(stateAfterAction))
        print("actionLoop"+str(actionLoop))
        print("getNewQvalue"+str(getNewQvalue))
        print("Q_rewardSpecific"+str(Q_reward[stateBeforeAction,actionLoop]))
        print("rewardAfterAction"+str(rewardAfterAction))
        actionLoop = numpy.argmax(Q_reward[stateAfterAction,:])
        stateBeforeAction=stateAfterAction
        
        print("---------------------------- new step "+str(stateLoop)+" in episode "+str(episodesCount)+" -----------------------------")
    print("************************* new episode "+str(episodesCount)+" *******************************")
    print("************************* new episode "+str(episodesCount)+" *******************************")
    print("************************* new episode "+str(episodesCount)+" *******************************")
    print("************************* new episode "+str(episodesCount)+" *******************************")

print(Q_reward);

# Testing

state = env.reset()
tot_reward = 0
for t in range(50):  # default 50
    action = numpy.argmax(Q_reward[state,:])

    print(action)
    state, reward, done, info = env.step(action)
    print(state)
    tot_reward += reward
    print(tot_reward)
    env.render()
    time.sleep(1)
    if done:
        print("Total reward %d" %tot_reward)
        break
