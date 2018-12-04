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

Open the `p2_continuous-control` folder and in it start the `p2_continuous-control`
Jupyter notebook. Run the first 4 blocks of the notebook to train the policy and 
critic network and display the results. The resulsts are in genereal sensitive to the
initialization seed, if the simulation fails to reach comparative performance, try again 
with a different seed.

```
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)        
```



