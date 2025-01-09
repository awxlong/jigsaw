from PIL import Image
import numpy as np
import random
import pygame
import sys

# Constants
GRID_SIZE = 3
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def split_image(image_path, grid_size):
    img = Image.open(image_path)
    img = img.resize((SCREEN_WIDTH, SCREEN_HEIGHT))  # Resize the image to fit the screen
    img_width, img_height = img.size
    piece_width = img_width // grid_size
    piece_height = img_height // grid_size

    pieces = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * piece_width
            top = i * piece_height
            right = (j + 1) * piece_width
            bottom = (i + 1) * piece_height
            piece = img.crop((left, top, right, bottom))
            pieces.append(piece)
    return pieces, piece_width, piece_height

def shuffle_pieces(pieces):
    shuffled_pieces = pieces.copy()
    random.shuffle(shuffled_pieces)
    return shuffled_pieces

def create_puzzle_grid(pieces, grid_size, piece_width, piece_height):
    puzzle_image = Image.new('RGB', (piece_width * grid_size, piece_height * grid_size))
    for index, piece in enumerate(pieces):
        row = index // grid_size
        col = index % grid_size
        puzzle_image.paste(piece, (col * piece_width, row * piece_height))
    return puzzle_image

def is_solved(grid, grid_size):
    # Check if the grid is in the correct order
    return grid == list(range(grid_size * grid_size))

def draw_puzzle(screen, pieces, piece_width, piece_height, grid, empty_pos):
    screen.fill(WHITE)
    for index, piece_index in enumerate(grid):
        if piece_index == empty_pos:
            continue  # Skip drawing the empty piece
        row = index // GRID_SIZE
        col = index % GRID_SIZE
        x = col * piece_width
        y = row * piece_height
        piece_surface = pygame.image.fromstring(pieces[piece_index].tobytes(), pieces[piece_index].size, pieces[piece_index].mode)
        screen.blit(piece_surface, (x, y))

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, BLACK, (0, i * piece_height), (SCREEN_WIDTH, i * piece_height), 2)
        pygame.draw.line(screen, BLACK, (i * piece_width, 0), (i * piece_width, SCREEN_HEIGHT), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jigsaw Puzzle")
    clock = pygame.time.Clock()

    image_path = 'lima.jpg'
    pieces, piece_width, piece_height = split_image(image_path, GRID_SIZE)
    shuffled_pieces = shuffle_pieces(pieces)

    # Initialize the grid with shuffled pieces
    grid = list(range(GRID_SIZE * GRID_SIZE))
    random.shuffle(grid)
    empty_pos = grid.index(GRID_SIZE * GRID_SIZE - 1)  # Assume the last piece is empty

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the position of the mouse click
                x, y = pygame.mouse.get_pos()
                col = x // piece_width
                row = y // piece_height
                clicked_index = row * GRID_SIZE + col

                # Check if the clicked piece is adjacent to the empty space
                if abs(clicked_index - empty_pos) == 1 or abs(clicked_index - empty_pos) == GRID_SIZE:
                    # Swap the clicked piece with the empty space
                    grid[empty_pos], grid[clicked_index] = grid[clicked_index], grid[empty_pos]
                    empty_pos = clicked_index

        # Draw the puzzle
        draw_puzzle(screen, shuffled_pieces, piece_width, piece_height, grid, empty_pos)

        # Check if the puzzle is solved
        if is_solved(grid, GRID_SIZE):
            font = pygame.font.Font(None, 74)
            text = font.render("Congratulations!", True, RED)
            screen.blit(text, (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 50))
            pygame.display.flip()
            pygame.time.wait(3000)  # Display the message for 3 seconds
            pygame.quit()
            sys.exit()

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()