# Building an AI Agent using Agentic AI

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

### Executive Summary
The notebook outlines an end-to-end machine learning pipeline, including data acquisition via yfinance, feature engineering with technical indicators, and the construction of a custom Reinforcement Learning environment. While the technical framework is soundly built using PyTorch, the initial training results indicate significant challenges. In the provided test run, the agent ended with a 99% loss (Final Balance: $102.35 from an initial $10,000), highlighting the difficulty of training financial agents without highly tuned reward signals and risk management.

### Goal  

- Automated Strategy: Develop an agent capable of making autonomous "Buy," "Sell," or "Hold" decisions based on market states.

- RL Environment Construction: Create a custom OpenAI Gym-style environment that simulates trading logic, including balance tracking and asset holdings.

- Predictive Modeling: Use a Deep Neural Network to approximate Q-values (the expected future rewards) for specific market conditions.

- Performance Evaluation: Test the agent's ability to generalize its learned strategy on historical data to see if it can outperform a simple "buy and hold" approach.

### Data structure and initial checks
[Dataset](The dataset was captured from yahoo finance of Amazon stocks)

### Tools
Python: Data manipulation, Data preprocessing, Neural networks, Model training & evaluation

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
symbol = "AMZN" # Amazon stock price
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
Training the agent
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
<img width="829" height="592" alt="image" src="https://github.com/user-attachments/assets/5c72eff9-42dd-47a2-b1ff-0eaf5d6716ec" />
<img width="424" height="560" alt="image" src="https://github.com/user-attachments/assets/d8b7f583-982d-4f80-be82-8b6f2532c0e3" />
<img width="403" height="555" alt="image" src="https://github.com/user-attachments/assets/597b66d6-1d1b-4d13-b3a6-3a2d16090695" />
<img width="415" height="594" alt="image" src="https://github.com/user-attachments/assets/1e3c6a4b-6708-4b5e-8156-5f786d06a1fa" />
<img width="403" height="580" alt="image" src="https://github.com/user-attachments/assets/26233b3d-2a08-4330-8dc4-c73432223295" />
<img width="400" height="595" alt="image" src="https://github.com/user-attachments/assets/bde2b676-d865-4ed9-acbe-dda0a10e5b74" />
<img width="402" height="580" alt="image" src="https://github.com/user-attachments/assets/bad8e075-35df-40a6-8a68-f2fba7467d3d" />
<img width="438" height="601" alt="image" src="https://github.com/user-attachments/assets/8bfe2af2-3f76-44be-b904-cd0cd68c7b9c" />
<img width="419" height="594" alt="image" src="https://github.com/user-attachments/assets/1c664357-d5dd-4569-9b99-b6fee9b4f45c" />
<img width="433" height="587" alt="image" src="https://github.com/user-attachments/assets/d8cb47de-7dc9-492f-af29-31fcb16527d1" />
<img width="424" height="593" alt="image" src="https://github.com/user-attachments/assets/44857a59-1c9e-4cba-b708-817b9e46d736" />
<img width="414" height="591" alt="image" src="https://github.com/user-attachments/assets/876c0485-1c21-4cae-b6b7-ece9b79eb478" />

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
<img width="343" height="63" alt="image" src="https://github.com/user-attachments/assets/db0f6464-75c4-4257-a413-3fa01cd7a422" />

### Insights

- Sparse Reward Problem: The current environment only provides a reward at the very end of an episode (if done: reward = self.balance - self.initial_balance). This makes it extremely difficult for the agent to understand which specific daily actions led to success or failure.

- Aggressive Trading Logic: The step function is "all-in." When it buys, it uses the entire balance to buy as many shares as possible. This lack of position sizing makes the agent highly vulnerable to market volatility.

- Convergence Issues: The training logs show "Total Reward" values hovering around -9,800 consistently. This suggests the agent is failing to learn a profitable pattern and is likely stuck in a loop of losing its capital early in the simulation.

- Feature Limitations: The agent currently only looks at the Close Price, SMA_5, SMA_20, and Returns. While these are standard, they may not provide enough context for the model to differentiate between a "dip" and a "crash."

### Recommendations
To turn this from a loss-making model into a viable trading agent, I recommend the following adjustments:

1. Refine the Reward Function
Instead of waiting until the end of the 5-year period to give a reward, implement Step-wise Rewards. Reward the agent daily based on the percentage change in its "Net Account Value" (Balance + Market Value of Holdings).

2. Implement a Target Network
The current DQNAgent uses a single model for both prediction and target calculation. Standard DQN practice involves using a Target Network (a copy of the model updated every few hundred steps) to stabilize training and prevent the "moving target" problem.

3. Add Risk Management
Position Sizing: Modify the BUY action to only use a fraction of the available cash (e.g., 10%).

Stop-Loss: Add logic to automatically sell if the price drops a certain percentage below the purchase price.

4. Expand the Feature Set
Introduce more sophisticated technical indicators to give the agent a better "view" of the market:

- RSI (Relative Strength Index) to detect overbought/oversold conditions.

- MACD for momentum.

- Volume to confirm the strength of price movements.
