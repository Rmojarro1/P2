from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 100 # 1000 default. 
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """

    while True:
        assert node, "Node is none in traverse_nodes"

        #return node if untried actions exist
        if node.untried_actions:
            return node, state
        
        #return node if the game is over
        if board.is_ended(state):
            return node, state
        
        #otherwise, use UCB for the next node
        best_child, best_value = None, -float('inf')
        for action, child in node.child_nodes.items():
            is_opponent = (bot_identity != board.current_player(state))
            child_value = ucb(child, is_opponent)
            if (child_value > best_value):
                best_value = child_value
                best_child = child

        #update node and the state for the next loop
        node = best_child
        state = board.next_state(state, node.parent_action)
    

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    assert node, "Node is none in expand_leaf"

    #if untried actions for the node exist
    if node.untried_actions:
        action = node.untried_actions.pop()
        next_state = board.next_state(state, action)
        child_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(next_state))
        node.child_nodes[action] = child_node
        return child_node, next_state
    else:
        return node, state

def rollout(board: Board, state, bot_identity: int):
    """
    Simulates the game to completion using a heuristic to prioritize winning and blocking moves.

    Args:
        board: The game setup.
        state: The current state of the game.
        bot_identity: The identity of the bot (1 or 2).

    Returns:
        state: The terminal state of the game.
    """
    opponent_identity = 3 - bot_identity
    while not board.is_ended(state):  # Continue until the game ends
        winning_move = None
        blocking_move = None
        subbox_winning_move = None
        block_subbox = None 

        current_ownership = list(board.owned_boxes(state).values())
        for action in board.legal_actions(state):
            next_state = board.next_state(state, action)
            next_ownership = list(board.owned_boxes(next_state).values()) 
            # Check for a winning move
            if board.points_values(next_state):
                if is_win(board, next_state, bot_identity):
                    winning_move = action
                    break
                if is_win(board, next_state, opponent_identity):
                    blocking_move = action
                    break
                # Check for sub-box win (logic probably the issue)
                if sum(next_ownership) - sum(current_ownership) == bot_identity:
                    subbox_winning_move = action
                if sum(next_ownership) - sum(current_ownership) == opponent_identity:
                    block_subbox = action

        # Prioritize moves
        if winning_move:
            action = winning_move
        elif blocking_move:
            action = blocking_move
        elif subbox_winning_move:
            action = subbox_winning_move
        elif block_subbox:
            action = block_subbox
        else:
            action = choice(board.legal_actions(state))
        state = board.next_state(state, action)
    return state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if not node:
        return
    node.visits += 1
    node.wins += int(won)
    backpropagate(node.parent, won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    assert node, "Node is None in ucb"
    assert node.visits > 0, "Node has not yet been visited in ucb"
    assert node.parent, "Node does not have a parent in ucb"
    win_rate = (node.wins / node.visits)
    if is_opponent:
        win_rate = 1 - win_rate
    explore_value = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    return win_rate + explore_value

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    assert root_node.child_nodes, "Root node has no children"
    return max(root_node.child_nodes.values(), key=lambda x: x.wins / max(x.visits, 1)).parent_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """

    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS - This is all you!
        # ...
        node, state = traverse_nodes(node, board, state, bot_identity)
        node, state = expand_leaf(node, board, state)
        terminal_state = rollout(board, state, bot_identity)
        player_won = is_win(board, terminal_state, bot_identity)
        backpropagate(node, player_won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    print(f"Action chosen: {best_action}")
    # print(f"Action chosen: {best_action}")
    return best_action