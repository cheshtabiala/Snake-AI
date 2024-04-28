import pygame
import random
import os
import numpy as np
from enum import Enum
from collections import namedtuple
from collections import deque
from pygame import mixer 
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import imageio.v3 as iio 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
w=800
h=800
BLOCK_SIZE = 20
SPEED = 20

pygame.init()

screen=pygame.display.set_mode((w,h))

food_image = pygame.image.load('frog.png')
food_image = pygame.transform.scale(food_image, (BLOCK_SIZE, BLOCK_SIZE))
font = pygame.font.SysFont('arial', 25)

'''This creates a Python Enum (enumerator) called Direction
which assigns integer values to the different directions that the snake can move in.'''
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 255, 0)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class Hungry_Snake:
    
    def _init_(self):
        self.w = w
        self.h = h
        background_music = mixer.music.load('Snake Game - Theme Song.mp3')
        mixer.music.play(-1)
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Hungry Snake')
        icon = pygame.image.load('icon.jpg') #to change logo 
        pygame.display.set_icon(icon)
            
        self.clock = pygame.time.Clock()
        self.reset()
        # isko dekhna hai clock ko

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0  # ek tarah se ye refersh rate hai

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food() #recursive call
            

    def play_step(self, action):
            self.frame_iteration += 1
            # 1. collect user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                                
            # 2. move
            self._move(action)  # update the head
            self.snake.insert(0, self.head)

            # 3. check if game over
            reward = 0
            game_over = False
            if self.is_collision() or self.frame_iteration > 100*len(self.snake):
                game_over = True
                reward = -10
                return reward, game_over, self.score
            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                reward = 10 
            else:
                self.snake.pop()

            # 5. update ui and clock
            self._update_ui()
            self.clock.tick(SPEED)
            # 6. return game over and score
            return reward, game_over, self.score

    def is_collision(self, pt=None):
            if pt is None:
                pt = self.head
            # hits boundary
            if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
                return True
            # hits itself
            if pt in self.snake[1:]:
                return True

            return False

    def _update_ui(self):
            self.display.fill(BLACK)
            
            # '''To change the background image'''
            # background = pygame.image.load('background.jpg').convert() # Load the image and convert it to a surface obj
            # background = pygame.transform.scale(background, (self.w, self.h)) # Resize the image to fit the screen size
            # self.display.blit(background,(0,0)) #draw image on screen
            
            for pt in self.snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2,pygame.Rect(pt.x+4, pt.y+4, 12, 12))

            # Rectangle food
            # pygame.draw.rect(self.display, RED, pygame.Rect(
            #     self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            
            #circular food
            # center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
            # pygame.draw.circle(self.display, RED, center, BLOCK_SIZE // 2)
            
            self.display.blit(food_image, (self.food.x, self.food.y))

            text = font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])
            pygame.display.flip()

    def _move(self, action):
            # [staright, right, left]

            clock_wise = [Direction.RIGHT, Direction.DOWN,
                Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx+1) % 4
                new_dir = clock_wise[next_idx]  # right turn r->d->l->up
            else:  # [0,0,1]
                next_idx = (idx-1) % 4
                new_dir = clock_wise[next_idx]  # left turn r->up->l->down

            self.direction = new_dir

            x = self.head.x
            y = self.head.y
            if self.direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction == Direction.UP:
                y -= BLOCK_SIZE

            self.head = Point(x, y)


plt.ion()

def plot(scores, mean_score):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_score)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_score)-1, mean_score[-1], str(mean_score[-1]))
    plt.show(block=False)
    plt.pause(.1)


class Linear_QNet(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super()._init_()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def _init_(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
            # (n, x)

        if len(state.shape) == 1:
                # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

            # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * \
                    torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Agent:
        
    def _init_(self):
            self.n_games = 0
            self.epsilon = 0 # randomness
            self.gamma = 0.9 # discount rate
            self.memory = deque(maxlen=MAX_MEMORY) # popleft()
            self.model = Linear_QNet(11, 256, 3)
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
            head = game.snake[0]
            point_l = Point(head.x - 20, head.y)
            point_r = Point(head.x + 20, head.y)
            point_u = Point(head.x, head.y - 20)
            point_d = Point(head.x, head.y + 20)
            
            dir_l = game.direction == Direction.LEFT
            dir_r = game.direction == Direction.RIGHT
            dir_u = game.direction == Direction.UP
            dir_d = game.direction == Direction.DOWN

            state = [
                # Danger straight
                (dir_r and game.is_collision(point_r)) or 
                (dir_l and game.is_collision(point_l)) or 
                (dir_u and game.is_collision(point_u)) or 
                (dir_d and game.is_collision(point_d)),

                # Danger right
                (dir_u and game.is_collision(point_r)) or 
                (dir_d and game.is_collision(point_l)) or 
                (dir_l and game.is_collision(point_u)) or 
                (dir_r and game.is_collision(point_d)),

                # Danger left
                (dir_d and game.is_collision(point_r)) or 
                (dir_u and game.is_collision(point_l)) or 
                (dir_r and game.is_collision(point_u)) or 
                (dir_l and game.is_collision(point_d)),
                
                # Move direction
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                
                # Food location 
                game.food.x < game.head.x,  # food left
                game.food.x > game.head.x,  # food right
                game.food.y < game.head.y,  # food up
                game.food.y > game.head.y  # food down
                ]

            return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)
            # for state, action, reward, nexrt_state, done in mini_sample:
            #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
            # random moves: tradeoff exploration / exploitation
            self.epsilon = 80 - self.n_games
            final_move = [0,0,0]
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

            return final_move


def train():
        plot_scores = []
        plot_mean_score = []
        total_score = 0
        record = 0
        agent = Agent()
        game = Hungry_Snake()
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_score.append(mean_score)
                plot(plot_scores, plot_mean_score)


if _name_ == '_main_':
    train()
