import numpy as np
from dezero import Model, layers, functions, optimizers
from matplotlib import pyplot as plt

from common.gridworld import GridWorld


def one_hot(state):
    HIGHT, WIDTH = 3, 4
    vec = np.zeros(HIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Linear(100)
        self.l2 = layers.Linear(4)

    def forward(self, x):
        y = functions.relu(self.l1(x))
        y = self.l2(y)
        return y


class QLearningAgent:

    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.forward(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        next_qs = self.qnet.forward(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()

        target = reward + (1 - done) * self.gamma * next_q
        qs = self.qnet.forward(state)
        q = qs[:, action]
        loss = functions.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)

plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# visualize
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)
