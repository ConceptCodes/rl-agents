# Snake RL Agent

This is a reinforcement learning agent for playing Snake using the Gymnasium library.


### Observations 
put a table here

### Actions
The agent can perform the following actions:

### RL Algorithm
The agent is trained using a Q-learning algorithm (DQN). We're using epsilon-greedy policy for action selection. As well as building a q table to store the Q-values for each state-action pair.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To train the agent, run:

```bash
python src/snake/train.py
```

To evaluate the agent, run:

```bash
python src/snake/evaluate.py
```
