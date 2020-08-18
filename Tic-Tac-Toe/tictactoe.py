
"""
Tic Tac Toe Player
"""

import math
import copy


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    
    
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x_count = 0
    o_count = 0

    for rows in board:
        for columns in rows:
            if (columns == X):
                x_count += 1
            elif (columns == O):
                o_count += 1
    if (x_count <= o_count):
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if (board[i][j] == EMPTY):
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    copyd = copy.deepcopy(board)
    if (copyd[action[0]][action[1]] != EMPTY):
        return Exception
    else:
        copyd[action[0]][action[1]] = player(board)

    return copyd
    

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if (board[i][0] == board[i][1] == board[i][2] == X):
            return X
        elif (board[i][0] == board[i][1] == board[i][2] == O):
            return O
    for i in range(3):
        if (board[0][i] == board[1][i] == board[2][i] == X):
            return X
        elif (board[0][i] == board[1][i] == board[2][i] == O):
            return O

    if (board[0][0] == board[1][1] == board[2][2] == X):
        return X
    if (board[0][0] == board[1][1] == board[2][2] == O):
        return O
    if (board[0][2] == board[1][1] == board[2][0] == X):
        return X
    if (board[0][2] == board[1][1] == board[2][0] == O):
        return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if (winner(board) !=None):
        
        return True
    
    
    else:
        for i in range(3):
            for j in range(3):
                if (board[i][j] == EMPTY):
                    return False
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if (winner(board) == X):
        return 1
    elif (winner(board) == O):
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the move player on the board.
    """
    if (terminal(board)):
        return None

    if (player(board) == X):
        v = -math.inf
        best = None

        for action in actions(board):
            move = min_value(result(board, action))
            if move > v:
                v = move
                best = action
        return best

    elif (player(board) == O):
        v = math.inf
        best = None
        for action in actions(board):
            move = max_value(result(board, action))
            if move < v:
                v = move
                best = action
        return best


def max_value(board):
    if (terminal(board)):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))

    return v


def min_value(board):
    if (terminal(board)):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v