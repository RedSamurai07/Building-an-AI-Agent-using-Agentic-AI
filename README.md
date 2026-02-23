# Building an AI Agent using Agentic AI

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Data Analysis](#data-analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

### Executive Summary


### Goal  
### Data structure and initial checks
[Dataset]()

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
### Tools

### Analysis
``` python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yfinance as yf
from collections import deque
import warnings
warnings.filterwarnings("ignore")
```
Loading the data
``` python
# define stock symbol and time period
symbol = "AMZN" # Microsoft stock price
start_date = "2020-01-01"
end_date = "2025-01-01"

# download historical data
data = yf.download(symbol, start=start_date, end=end_date)
```
Data preprocessing
``` python
# feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Returns'] = data['Close'].pct_change()
```
Handling missing values
``` python
# drop NaN values and reset index
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
```
Setting actions for the stocks to sell or buy
``` python
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}
```
``` python
# get state function
def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])
```
Trading environment
``` python
# trading environment
class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return get_state(self.data, self.index)

    def step(self, action):
        price = float(self.data.loc[self.index, 'Close'])
        reward = 0

        if action == 1 and self.balance >= price:  # BUY
            self.holdings = self.balance // price
            self.balance -= self.holdings * price
        elif action == 2 and self.holdings > 0:  # SELL
            self.balance += self.holdings * price
            self.holdings = 0

        self.index += 1
        done = self.index >= len(self.data) - 1

        if done:
            reward = self.balance - self.initial_balance

        next_state = get_state(self.data, self.index) if not done else None
        return next_state, reward, done, {}
```
Deep Q network
``` python
# deep q-network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```
``` python
# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_tensor = self.model(state_tensor).clone().detach()
            target_tensor[0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```
Training the agen
``` python
# train the agent
env = TradingEnvironment(data)
agent = DQNAgent(state_size=4, action_size=3)
batch_size = 32
episodes = 500
total_rewards = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay(batch_size)
    total_rewards.append(total_reward)
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

print("Training Complete!")
```
Environment for the output
``` python
# create a fresh environment instance for testing
test_env = TradingEnvironment(data)
state = test_env.reset()
done = False

# simulate a trading session using the trained agent
while not done:
    # always choose the best action (exploitation)
    action = agent.act(state)
    next_state, reward, done, _ = test_env.step(action)
    state = next_state if next_state is not None else state

final_balance = test_env.balance
profit = final_balance - test_env.initial_balance
print(f"Final Balance after testing: ${final_balance:.2f}")
print(f"Total Profit: ${profit:.2f}")
```

### Insights

### Recommendations

