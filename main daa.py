import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PuzzleState:
    def __init__(self, board, empty_pos, g, parent=None):
        self.board = board
        self.empty_pos = empty_pos
        self.g = g  # Cost from start to this state
        self.parent = parent
        self.h = self.calculate_heuristic()
        self.f = self.g + self.h  # f = g + h (A*)
        self.h_greedy = self.h  # h (Greedy Best-First Search)
    
    def calculate_heuristic(self):
        """Calculate the Manhattan distance."""
        distance = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] != 0:
                    target_x = (self.board[i][j] - 1) // 4
                    target_y = (self.board[i][j] - 1) % 4
                    distance += abs(i - target_x) + abs(j - target_y)
        return distance

    def get_neighbors(self):
        """Get neighboring states by sliding a tile into the empty space."""
        neighbors = []
        x, y = self.empty_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(PuzzleState(new_board, (nx, ny), self.g + 1, self))
        return neighbors

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(str(self.board))

def reconstruct_path(state):
    path = []
    while state:
        path.append(state)
        state = state.parent
    return path[::-1]


def print_board(board):
    for row in board:
        print(' '.join(str(x).rjust(2, ' ') for x in row))
    print()

def solve_puzzle_astar(initial_board):
    empty_pos = None
    for i in range(4):
        for j in range(4):
            if initial_board[i][j] == 0:
                empty_pos = (i, j)
                break
        if empty_pos:
            break

    initial_state = PuzzleState(initial_board, empty_pos, 0)
    open_set = []
    heapq.heappush(open_set, initial_state)
    closed_set = set()

    while open_set:
        current_state = heapq.heappop(open_set)
        
        if current_state.h == 0:
            solution_path = reconstruct_path(current_state)
            return solution_path, len(solution_path) - 1
        
        closed_set.add(current_state)

        for neighbor in current_state.get_neighbors():
            if neighbor in closed_set:
                continue
            if neighbor not in open_set:
                heapq.heappush(open_set, neighbor)

    return None, 0

def draw_board(board, ax, time_taken, algorithm, moves_no):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], minor=True)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                ax.text(j, i, str(board[i][j]), ha='center', va='center', fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Display the time taken and algorithm name in the top left corner
    ax.text(0.7, 0.95, f'Time taken: {time_taken:.4f} seconds', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.02, 0.95, f'Algorithm : {algorithm}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.4, 0.95, f'No. of moves : {moves_no}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
def animate_solution_astar(solution_path, time_taken, moves):
    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size here
    ani = animation.FuncAnimation(fig, draw_board, frames=[state.board for state in solution_path], fargs=(ax, time_taken, "A*", moves), interval=500, repeat=False)
    plt.show()

def animate_solution_greedy(solution_path, time_taken, moves):
    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size here
    ani = animation.FuncAnimation(fig, draw_board, frames=[state.board for state in solution_path], fargs=(ax, time_taken, "Greedy", moves), interval=500, repeat=False)
    plt.show()

class GreedyPuzzleState:
    def __init__(self, board, empty_pos, g, parent=None):
        self.board = board
        self.empty_pos = empty_pos
        self.g = g  # Cost from start to this state
        self.parent = parent
        self.h = self.calculate_heuristic()
        self.f = self.h  # f = h (Greedy Best-First Search)

    def calculate_heuristic(self):
        """Calculate the Manhattan distance."""
        distance = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] != 0:
                    target_x = (self.board[i][j] - 1) // 4
                    target_y = (self.board[i][j] - 1) % 4
                    distance += abs(i - target_x) + abs(j - target_y)
        return distance

    def get_neighbors(self):
        """Get neighboring states by sliding a tile into the empty space."""
        neighbors = []
        x, y = self.empty_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(GreedyPuzzleState(new_board, (nx, ny), self.g + 1, self))
        return neighbors

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(str(self.board))

def greedy_solve_puzzle(initial_board):
    empty_pos = None
    for i in range(4):
        for j in range(4):
            if initial_board[i][j] == 0:
                empty_pos = (i, j)
                break
        if empty_pos:
            break

    initial_state = GreedyPuzzleState(initial_board, empty_pos, 0)
    open_set = []
    heapq.heappush(open_set, initial_state)
    closed_set = set()

    while open_set:
        current_state = heapq.heappop(open_set)

        if current_state.h == 0:
            solution_path = reconstruct_path(current_state)
            return solution_path, len(solution_path) - 1

        closed_set.add(current_state)

        for neighbor in current_state.get_neighbors():
            if neighbor in closed_set:
                continue
            if neighbor not in open_set:
                heapq.heappush(open_set, neighbor)

    return None, 0


if __name__ == "__main__":
    initial_board = [
        [5, 1, 2, 3],
        [4, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 0, 15]
    ]
    
    # Solving with A*
    start_time_astar = time.time()
    solution_path_astar, num_moves_astar = solve_puzzle_astar(initial_board)
    end_time_astar = time.time()
    time_taken_astar = end_time_astar - start_time_astar
    
    if solution_path_astar:
        print("Solution found with A*:")
        print(f"Number of moves: {num_moves_astar}")
        print(f"Time taken: {time_taken_astar:.4f} seconds")
        animate_solution_astar(solution_path_astar, time_taken_astar, num_moves_astar)
    else:
        print("No solution found with A*.")
    
    print("\n")
    start_time_greedy = time.time()
    solution_path_greedy, num_moves_greedy = greedy_solve_puzzle(initial_board)
    end_time_greedy = time.time()
    time_taken_greedy = end_time_greedy - start_time_greedy

    if solution_path_greedy:
        print("Solution found with Greedy Best-First Search:")
        print(f"Number of moves: {num_moves_greedy}")
        print(f"Time taken: {time_taken_greedy:.4f} seconds")
        animate_solution_greedy(solution_path_greedy, time_taken_greedy, num_moves_greedy)
    else:
        print("No solution found with Greedy Best-First Search.")
