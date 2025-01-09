import pygame
import sys
from q_learning_agent import RLPuzzleAgent  # Import the RL agent
from PIL import Image
import random

# Constants
GRID_SIZE = 2
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

def is_solved(grid, grid_size):
    """Check if the puzzle is solved (excluding the empty space)."""
    # The solved state is [0, 1, 2, ..., grid_size * grid_size - 1]
    solved_state = list(range(grid_size * grid_size))
    return grid == solved_state

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

def train_agent(image_path, episodes=1000):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RL Puzzle Solver")
    clock = pygame.time.Clock()

    # Split the image into pieces
    pieces, piece_width, piece_height = split_image(image_path, GRID_SIZE)
    agent = RLPuzzleAgent(GRID_SIZE)

    for episode in range(episodes):
        # Initialize the grid with shuffled pieces
        grid = list(range(GRID_SIZE * GRID_SIZE))
        random.shuffle(grid)
        empty_pos = grid.index(GRID_SIZE * GRID_SIZE - 1)  # Assume the last piece is empty

        # Create a shuffled_pieces list that matches the grid
        shuffled_pieces = [pieces[i] for i in grid]

        episode_solved = False  # Track if the episode is solved

        while not episode_solved:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Agent chooses an action
            possible_actions = agent.get_possible_actions(grid)
            action_index = agent.choose_action(grid, possible_actions)
            action = possible_actions[action_index]  # Convert action index to grid index

            # Perform the action
            new_grid = grid.copy()
            new_grid[empty_pos], new_grid[action] = new_grid[action], new_grid[empty_pos]

            # Reward: +1 if the puzzle is solved, 0 otherwise
            reward = 1 if is_solved(new_grid, GRID_SIZE) else 0

            # Update Q-table
            agent.update_q_table(grid, action_index, reward, new_grid)

            # Update state
            grid = new_grid
            empty_pos = action

            # Update shuffled_pieces to match the new grid
            shuffled_pieces = [pieces[i] for i in grid]

            # Draw the puzzle
            draw_puzzle(screen, shuffled_pieces, piece_width, piece_height, grid, empty_pos)
            pygame.display.flip()
            clock.tick(FPS)

            # Check if the puzzle is solved
            if is_solved(grid, GRID_SIZE):
                # Display "Congratulations" message
                font = pygame.font.Font(None, 74)
                text = font.render("Congratulations!", True, RED)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                screen.blit(text, text_rect)
                pygame.display.flip()
                pygame.time.wait(2000)  # Display the message for 2 seconds
                episode_solved = True  # Mark the episode as solved

        # Decay exploration rate
        agent.decay_exploration_rate()

        print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {agent.exploration_rate:.2f}")

    pygame.quit()

if __name__ == "__main__":
    train_agent('lima.jpg', episodes=2)