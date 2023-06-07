import pandas as pd
import pygame as pg
from random import choice

df1 = pd.read_csv('file1h.csv')
df2 = pd.read_csv('file2v.csv')
new_data = []
for r1, r2 in zip(df1['predicted_label'], df2['predicted_label']):  # right up
    new_data.append(r1)
    new_data.append(r2)
df = pd.DataFrame(new_data, columns=["predicted_label"])

print(df)

window = 500
Tile_size = 40
RANGE = (Tile_size // 2, window - Tile_size // 2, Tile_size)
food_positions = [(window // 2, window // 2 - Tile_size), (309, 309), (269, 269)]
get_random_position = lambda: choice(food_positions)

snake = pg.rect.Rect([window // 2, window // 2, Tile_size - 2, Tile_size - 2])
time, time_step = 0, 110
food = snake.copy()
food.center = (window // 2, window // 2 - Tile_size)  # Set the initial food position above the center

length = 1
segments = [snake.copy()]
direction = (Tile_size, 0)  # Set the initial direction to move right
screen = pg.display.set_mode([window] * 2)
clock = pg.time.Clock()

def move_snake(label):
    global direction, segments, time_step, length
    if label == 'Right':
        direction = (Tile_size, 0)
    elif label == 'Left':
        direction = (-Tile_size, 0)
    elif label == 'Up':
        direction = (0, -Tile_size)
    elif label == 'Down':
        direction = (0, Tile_size)
    elif label == 'Blink':
        direction = (Tile_size, 0)  # Move right (faster)
        time_step = 80  # Decrease time step for faster movement

    snake.move_ip(direction)
    segments.append(snake.copy())
    if len(segments) > length:
        segments = segments[-length:]

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exit()

    screen.fill('black')
    selfeating = any(segment.colliderect(snake) for segment in segments[:-1])
    if (
        snake.left < 0
        or snake.right > window
        or snake.top < 0
        or snake.bottom > window
        or selfeating
    ):
        snake.center, food.center = (window // 2, window // 2), get_random_position()
        length, direction = 1, (Tile_size, 0)  # Reset the direction to move right
        segments = [snake.copy()]
        time_step = 110  # Reset time step to default
    if snake.colliderect(food):
        food.center = get_random_position()
        length += 1

    if not df.empty:
        label = df.iloc[0]['predicted_label']
        print("Snake coordinates:", snake.center)  # Print the snake's coordinates
        move_snake(label)
        df = df.iloc[1:]

    pg.draw.rect(screen, 'red', food)
    [pg.draw.rect(screen, 'green', segment) for segment in segments]

    pg.display.flip()
    clock.tick(1)  # Adjust the frame rate to control the snake's movement speed
