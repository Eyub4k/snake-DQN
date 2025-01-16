import pygame
import numpy as np
import random
from collections import deque
import json
import os

class SnakeGame:
    def __init__(self, width=400, height=400, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.reset()

    def _place_food(self):
        while True:
            food = (random.randint(0, (self.width-self.grid_size)//self.grid_size) * self.grid_size,
                   random.randint(0, (self.height-self.grid_size)//self.grid_size) * self.grid_size)
            if food not in self.snake:
                return food

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = random.choice([(0, -self.grid_size), (0, self.grid_size), 
                                      (-self.grid_size, 0), (self.grid_size, 0)])
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _get_state(self):
        head = self.snake[0]
        
        danger = [
            self._is_danger(head[0], head[1] - self.grid_size),  # haut
            self._is_danger(head[0] + self.grid_size, head[1]),  # droite
            self._is_danger(head[0], head[1] + self.grid_size),  # bas
            self._is_danger(head[0] - self.grid_size, head[1])   # gauche
        ]
        
        dir_u = self.direction == (0, -self.grid_size)
        dir_r = self.direction == (self.grid_size, 0)
        dir_d = self.direction == (0, self.grid_size)
        dir_l = self.direction == (-self.grid_size, 0)
        
        food_u = self.food[1] < head[1]
        food_r = self.food[0] > head[0]
        food_d = self.food[1] > head[1]
        food_l = self.food[0] < head[0]
        
        return np.array(danger + [dir_u, dir_r, dir_d, dir_l, food_u, food_r, food_d, food_l], dtype=int)

    def _is_danger(self, x, y):
        return (x < 0 or x >= self.width or 
                y < 0 or y >= self.height or 
                (x, y) in self.snake[:-1])

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return False
        return True

    def step(self, action):
        if action == 1:  # gauche
            self.direction = self._turn_left(self.direction)
        elif action == 2:  # droite
            self.direction = self._turn_right(self.direction)
            
        new_head = (self.snake[0][0] + self.direction[0], 
                   self.snake[0][1] + self.direction[1])
        
        reward = 0
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self._place_food()
            else:
                self.snake.pop()
                reward = -0.1
                
        return self._get_state(), reward, self.game_over

    def _turn_left(self, direction):
        if direction == (0, -self.grid_size): return (-self.grid_size, 0)
        if direction == (-self.grid_size, 0): return (0, self.grid_size)
        if direction == (0, self.grid_size): return (self.grid_size, 0)
        return (0, -self.grid_size)

    def _turn_right(self, direction):
        if direction == (0, -self.grid_size): return (self.grid_size, 0)
        if direction == (self.grid_size, 0): return (0, self.grid_size)
        if direction == (0, self.grid_size): return (-self.grid_size, 0)
        return (0, -self.grid_size)

    def render(self):
        if not self._handle_events():
            return False
        
        self.screen.fill((0, 0, 0))
        
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        (self.food[0], self.food[1], self.grid_size, self.grid_size))
        
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (segment[0], segment[1], self.grid_size, self.grid_size))
            
        pygame.display.flip()
        self.clock.tick(10)
        return True

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = np.zeros((2**self.state_size, self.action_size))
        return model

    def save_model(self, filename='snake_model.json'):
        model_data = {
            'model': self.model.tolist(),
            'epsilon': self.epsilon
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
        print(f"Modèle sauvegardé dans {filename}")

    def load_model(self, filename='snake_model.json'):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                model_data = json.load(f)
                self.model = np.array(model_data['model'])
                self.epsilon = model_data['epsilon']
            print(f"Modèle chargé depuis {filename}")
            return True
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state_idx = int(''.join(map(str, state)), 2)
        return np.argmax(self.model[state_idx])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_idx = int(''.join(map(str, state)), 2)
            next_state_idx = int(''.join(map(str, next_state)), 2)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model[next_state_idx])
            
            self.model[state_idx][action] = (1-self.learning_rate) * self.model[state_idx][action] + \
                                          self.learning_rate * target

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(continue_training=False):
    env = SnakeGame()
    state_size = 12
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    if continue_training:
        agent.load_model()
    
    episodes = 1000
    batch_size = 32
    save_frequency = 100

    running = True
    for e in range(episodes):
        if not running:
            break
            
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"episode: {e+1}/{episodes}, score: {env.score}, epsilon: {agent.epsilon:.2f}")
                break
            
            if not env.render():
                running = False
                break
            
        if running:
            agent.replay(batch_size)
            
            if (e + 1) % save_frequency == 0:
                agent.save_model()

    if running:
        agent.save_model()
    pygame.quit()

if __name__ == "__main__":
    train(continue_training=True)