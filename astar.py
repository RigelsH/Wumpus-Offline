import argparse
import random
import sys
import json
import networkx as nx
from typing import Iterable
import wumpus as wws
from wumpus import run_episode
from utils import json2graph, Agent, cost_function
import functools
from heuristics import (
    heuristic_manhatten_distance, 
    heuristic_euclidian_distance,
    heuristic_manhatten_distance_cheat,
    heuristic_minmax
)


class AstarPlayer(wws.OfflinePlayer):
    """Offline player demonstrating the use of the start episode method to inspect the world."""

    def start_episode(self, world: wws.WumpusWorld) -> Iterable[wws.Hunter.Actions]:
        """Print the description of the world before starting."""

        world_info = {k: [] for k in ('Hunter', 'Pits', 'Wumpus', 'Gold', 'Exits')}
        world_info['Size'] = (world.size.x, world.size.y)
        world_info['Blocks'] = [(c.x, c.y) for c in world.blocks]

        for obj in world.objects:
            if isinstance(obj, wws.Hunter):
                world_info['Hunter'].append((obj.location.x, obj.location.y))
                all_actions = list(obj.Actions)
            elif isinstance(obj, wws.Pit):
                world_info['Pits'].append((obj.location.x, obj.location.y))
            elif isinstance(obj, wws.Wumpus):
                world_info['Wumpus'].append((obj.location.x, obj.location.y))
                print('this is ',world_info['Wumpus'][0])
            elif isinstance(obj, wws.Exit):
                world_info['Exits'].append((obj.location.x, obj.location.y))
            elif isinstance(obj, wws.Gold):
                world_info['Gold'].append((obj.location.x, obj.location.y))

        print('World details:')
        for k in ('Size', 'Pits', 'Wumpus', 'Gold', 'Exits', 'Blocks'):
            print('  {}: {}'.format(k, world_info.get(k, None)))

        def astar_search():
            """
            Perform astar search.
            """
            # Convert the world map to a graph
            G = json2graph(world.to_dict())

            # Get the locations of the hunter, wumpus, exit, gold and the pits
            hunter_loc = world_info['Hunter'][0]
            wumpus_loc = world_info['Wumpus'][0]
            gold_loc = world_info['Gold'][0]
            exit_loc = world_info['Exits'][0]
            pits_loc = world_info['Pits']

            # If the gold is in a pit, return the action to climb out of the game
            if gold_loc in pits_loc:
                yield all_actions[5]

            # Calculate the cost of the path through the Wumpus
            # -------------------------------------------------
            
            # Check if there is a path to the gold
            if nx.has_path(G, source=hunter_loc, target=gold_loc):

                # Usage of the heuristic 
                heuristic = functools.partial(heuristic_minmax)
                
                # For a star we don't use path reversal, instead we calculate the path twice

                # From Hunters Location (0,0) to the gold location
                wumpus_path_forward = nx.astar_path(G=G, source=hunter_loc, target=gold_loc, heuristic=heuristic)
                
                # From gold's location to the exit point (0,0)
                wumpus_path_backward = nx.astar_path(G=G, source=gold_loc, target=exit_loc, heuristic=heuristic)
                
                # Concatenate
                wumpus_path = wumpus_path_forward + wumpus_path_backward[1:]

                # Calculate the cost
                total_cost_wumpus = cost_function(
                    path=wumpus_path, 
                    has_wumpus=True,
                    hunter_loc=hunter_loc, 
                    gold_loc=gold_loc,
                    wumpus_loc=wumpus_loc
                )
            else:
                # If there is no path to the gold, climb out of the game
                yield all_actions[5]

            # Calculate the cost of the path that avoids the Wumpus
            # -----------------------------------------------------

            # This could be the same path, if the path found above does not include Wumpus in it
            # Wumpus basically becomes a pit (unreachable node as far as the Graph is concerned)

            G_no_wumpus = G.copy()
            if wumpus_loc in G_no_wumpus.nodes():
                G_no_wumpus.remove_node(wumpus_loc)

            # Check if there is a path to the gold
            if nx.has_path(G_no_wumpus, source=hunter_loc, target=gold_loc):

                # Usage of the heuristics
                heuristic = functools.partial(heuristic_minmax)

                # Calculate the path from hunters location (0,0) to the gold location
                no_wumpus_path_forward = nx.astar_path(G=G_no_wumpus, source=hunter_loc, target=gold_loc, heuristic=heuristic)

                # Calculate the path from gold location to the exit location (0,0)
                no_wumpus_path_backward = nx.astar_path(G=G_no_wumpus, source=gold_loc, target=exit_loc, heuristic=heuristic)
                
                # Concatenate
                no_wumpus_path = no_wumpus_path_forward + no_wumpus_path_backward[1:]
                
                # Calculate the cost
                total_cost_no_wumpus = cost_function(
                    path=no_wumpus_path, 
                    has_wumpus=False,
                    hunter_loc=hunter_loc, 
                    gold_loc=gold_loc,
                    wumpus_loc=wumpus_loc
                )
            else:
                no_wumpus_path = wumpus_path
                total_cost_no_wumpus = total_cost_wumpus

            # Choose the path with the lowest cost
            # ------------------------------------

            if total_cost_wumpus > total_cost_no_wumpus:
                final_path = no_wumpus_path
            else:
                final_path = wumpus_path

            stl = Agent(
                hunter_location=hunter_loc, 
                gold_location=gold_loc, 
                wumpus_location=wumpus_loc
            )
            toyld = stl.navigate(path=final_path)
            
             # Yield the actions needed to follow the chosen path

            for element in toyld:
                if element == 'Move' :
                    yield all_actions[0]
                elif element == 'Right' :
                    yield all_actions[1]
                elif element == 'Left':
                    yield all_actions[2]
                elif element == 'Shoot':
                    yield all_actions[3]
                elif element == 'Grab':
                    yield all_actions[4]
                else :
                    yield all_actions[5]
    
        return astar_search()


def play_fixed_informed():
    """Play on a given world described in JSON format."""
    
    with open("data/wumpus_world-kill_to_grab.json") as fd:
        world_dict = json.load(fd)
    world = wws.WumpusWorld.from_JSON(world_dict)
    #Run a player with knowledge about the world
    run_episode(world=world ,player=AstarPlayer()) 


def play_fixed_informed2():
    """Play on a given world described in JSON format."""
    
    with open("data/wumpus_world-opt_kill.json") as fd:
        world_dict2 = json.load(fd)
    world = wws.WumpusWorld.from_JSON(world_dict2)
    #Run a player with knowledge about the world
    run_episode(world=world ,player=AstarPlayer()) 


EXAMPLES=(play_fixed_informed, play_fixed_informed2)


def main(*cargs):
    """Demonstrate the use of the wumpus API on selected worlds"""
    ex_names = {ex.__name__.lower(): ex for ex in EXAMPLES}
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('example', nargs='?', help='select one of the available example', choices=list(ex_names.keys()))
    args = parser.parse_args(cargs)
    
    if args.example:
        ex = ex_names[args.example.lower()]
    else:
        # Randomly play one of the examples
        ex = random.choice(EXAMPLES)

    print('Example {}:'.format(ex.__name__))
    print('  ' + ex.__doc__)
    ex()

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
