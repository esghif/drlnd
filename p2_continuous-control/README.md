# Navigation

### Environment Details

The goal of the Reacher environment is to move and keep a double-jointed arm
in a target location. The environment provides a reward of +0.1 for each time step 
that the arm is in the target position.
I've chose to work on the parallel environment that conatins 20 arms 
operating in paralllel.
The environment is considered solved when the average reward over all arms
and over the last 100 episodes is higher than 30.


The state of an arm is given by 33 variables, while the action is a 4 variable
taking values between -1 and +1. Fro the whole environment the state variable is 
a 20x33 matrix, while the action is a 20x4 matrix.


### Installation Details

Clone the repository:

> git clone https://github.com/esghif/drlnd.git

Go to the project folder:

> cd drlnd/p2_continuous-control

Download the simulator from [Reacher_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) 
to the `p2_navigation` folder and unzip locally:

> unzip Reacher_Linux.zip


### How to Run

Start the `conda` environment provided in the classroom:

> source activate drlnd

Start `jupyter` notebook:

> jupyter notebook



### Learning Algorithm 



The learning algorithm employed is DDPG (Deep Deterministic Policy Gradient).
The algorithm is similar to A2C in that it uses a network to train
the policy (actor) and a separate netowrk to learn the Q-function (critic).
The policy function is learned using the formula for deterministic policies 
found by D. Silver et.al in "Deterministic Policy Gradient Algorithms".

In another respect the algorithm is similar to DQN that a experience
replay buffer and target network are used to stabilize learning.
A single actor and critic network is trained using samples from a replay buffer
that is filled with samples from all parallel agents.

The successful training of tha agent is dependent on a few extra factors.
One is gradient clipping. I used gradient clipping with a threshold value of 0.1
for both the actor and the critic networks as shown below:


```
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.critic_grad_threshold)
        self.critic_optimizer.step()
        ...
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.actor_grad_threshold)
        self.actor_optimizer.step()        
```

The second ingredient for the successful training is the gradual decrease of the number
of learning steps taken as the policy improves as shown below.

```
    def step(self, states, actions, rewards, next_states, dones, eps=0):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if eps < 10:
            self.mod = 1
            self.repeat = 20
        elif eps < 20:
            self.mod = 1
            self.repeat = 10
        elif eps < 100:
            self.mod = 1
            self.repeat = 2
        else:
            self.mod = 10
            self.repeat = 2  
```


### Plot Of Rewards

![Plot of Rewards](p2_scores.png)

### Ideas for Future Development

The DDPG algorithm seems to be very well suited to learn a policy that solves 
the Reacher environment. It was found, though, that learning is sensitive to 
the random initialization seed.



Prioritized experience replay has not really worked for me so the first thing I would continue to 
investigate whether there is not still a programming error. 


I attempted to solve the learning-from-pixels problem. I ran into the problem, though, that 
there seems to be a memory leak in the simulator and learning gets really difficult with time.
Of course, there may be a problem in my code, also.