import gym
import numpy as np
from dezero import Model, layers, functions, optimizers


class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = layers.Linear(128)
        self.l2 = layers.Linear(action_size)

    def forward(self, x):
        y = functions.relu(self.l1(x))
        y = functions.softmax(self.l2(y))
        return y


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Linear(128)
        self.l2 = layers.Linear(1)

    def forward(self, x):
        y = functions.relu(self.l1(x))
        y = self.l2(y)
        return y


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action],

    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # 価値関数の更新
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = functions.mean_squared_error(v, target)

        # 方策の更新
        delta = target - v
        delta.unchain()
        loss_pi = -functions.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


episodes = 3000
env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, prob, reward, next_state, done)
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

# plot
from common.utils import plot_total_reward

plot_total_reward(reward_history)


