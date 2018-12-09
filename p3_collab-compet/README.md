# Multi-Agent Learning on the Tennis Environment

### Environment Details

The Tennis environment consists of 2 agents controlling 2 rackets that hit a ball
over a net. Each agent receives a reward of +0.1 for hitting the ball over the
net and a penalty of -0.1 for letting the ball hit the ground of hitting the ball out of bounds.

Each agent receives a score per episode by summing all obtained rewards without discounting.
The higher score obtained by either of the agents is considered the score of the episode. 
The environment is solved when an average score of +0.5 is obtained over the
past 100 episodes. 

The state returned by the environment at each call of the `step` function is a 2x24 variable, 
i.e. a 24 dimensional vector for each agent. The 24 dimensional variable consists of a temporal sequence
of 3 consecutive 8 dimensional time snapshots. Each 8 dimensional time snapshot
gives the position and velocity of the racket and the ball.
The position of the racket is given in a coordinate system relative to the agent allowing training of
the agent without the need to transform the state.

The action is a 2 dimensional vector taking values between -1 and +1. The first component
controls the horizontal movement of the agent, while the second component controls the jumping 
of the racket (i.e. a value higher than +0.5 sends the jump command).


### Installation Details

Clone the repository:

> git clone https://github.com/esghif/drlnd.git

Go to the project folder:

> cd drlnd/p3_collab-compet

Download the simulator from [Tennis_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) 
to the `p3_collab-compet` folder and unzip locally:

> unzip Reacher_Linux.zip


### How to Run

Start the `conda` environment provided in the classroom:

> source activate drlnd

Start `jupyter` notebook:

> jupyter notebook

Open the `p3_collab-compet` folder and in it start the `Tennis_Solution`
Jupyter notebook. Run the first 4 blocks of the notebook to train the policy and 
critic network and display the results. 




