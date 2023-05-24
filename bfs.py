import argparse
import random
import sys
import json
import networkx as nx
from typing import Iterable
import wumpus as wws
from wumpus import run_episode
from utils import json2graph, Agent, cost_function


class BfsPlayer(wws.OfflinePlayer):
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
        
        def bfs_search():
            """
            Perform BFS search.
            """
            # Convert the world map to a graph
            G = json2graph(world.to_dict())

            # Get the locations of the hunter, wumpus, and gold, and the pits
            hunter_loc = world_info['Hunter'][0]
            wumpus_loc = world_info['Wumpus'][0]
            gold_loc = world_info['Gold'][0]
            pits = world_info['Pits']

            # If the gold is in a pit, return the action to climb out of the game
            if gold_loc in pits:
                yield all_actions[5]
            
            # Set the weights of the edges based on the direction of the move
            # This tends to make the change of y coordinates less expensive 
            # It slightly improves the nx.shortest_path because it gives an indicaction not to turn
            # (nx does not consider that turning costs 1)

            for u, v in G.edges():
                x1, y1 = u
                x2, y2 = v
                if y1 != y2:
                    G[u][v]['weight'] = 0.5 
                elif x1 != x2:
                    G[u][v]['weight'] = 1

            # Calculate the cost of the path through the Wumpus
            # -------------------------------------------------
            
            # Check if there is a path to the gold
            if nx.has_path(G, source=hunter_loc, target=gold_loc):

                wumpus_path_forward = nx.shortest_path(
                    G, source=hunter_loc, target=gold_loc, weight='weight', method="dijkstra")
                wumpus_path_backward = list(reversed(wumpus_path_forward))
                wumpus_path = wumpus_path_forward + wumpus_path_backward[1:]

                total_cost_wumpus = cost_function(
                    path=wumpus_path, 
                    has_wumpus=True,
                    hunter_loc=hunter_loc, 
                    gold_loc=gold_loc,
                    wumpus_loc=wumpus_loc
                )
            else:
                # If there is no path to the gold, return the action to climb out of the pit
                yield all_actions[5]
            
            # Calculate the cost of the path that avoids the Wumpus
            # -----------------------------------------------------

            # It basically turns Wumpus into a pit (for the purposes of the graph)

            G_no_wumpus = G.copy()
            if wumpus_loc in G_no_wumpus.nodes():
                G_no_wumpus.remove_node(wumpus_loc)
            
            # If there is a path to the gold without the wumpus
            if nx.has_path(G_no_wumpus, source=hunter_loc, target=gold_loc):
                no_wumpus_path_forward = nx.shortest_path(
                    G_no_wumpus, source=hunter_loc, target=gold_loc, weight='weight', method="dijkstra")
                no_wumpus_path_backward = list(reversed(no_wumpus_path_forward))
                no_wumpus_path = no_wumpus_path_forward + no_wumpus_path_backward[1:]

                total_cost_no_wumpus = cost_function(
                    path=no_wumpus_path, 
                    has_wumpus=False,
                    hunter_loc=hunter_loc, 
                    gold_loc=gold_loc,
                    wumpus_loc=wumpus_loc
                )
            else: 
                # If there is no path to the gold without the wumpus
                no_wumpus_path = wumpus_path
                total_cost_no_wumpus = total_cost_wumpus
            
            # Choose the path with the lowest cost
            # ------------------------------------

            if total_cost_wumpus > total_cost_no_wumpus:
                final_path = no_wumpus_path
            else:
                final_path = wumpus_path

            # Yield the actions needed to follow the chosen path
            
            stl = Agent(
                hunter_location=hunter_loc, 
                gold_location=gold_loc, 
                wumpus_location=wumpus_loc
            )
            toyld = stl.navigate(path=final_path)
            
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
    
        return bfs_search()


def play_fixed_informed():
    """Play on a given world described in JSON format."""
    
    with open("data/wumpus_world-kill_to_grab.json") as fd:
        world_dict = json.load(fd)
    world = wws.WumpusWorld.from_JSON(world_dict)
    #Run a player with knowledge about the world
    run_episode(world=world ,player=BfsPlayer()) 


def play_fixed_informed2():
    """Play on a given world described in JSON format."""
    
    with open("data/wumpus_world-opt_kill.json") as fd:
        world_dict2 = json.load(fd)
    world = wws.WumpusWorld.from_JSON(world_dict2)
    #Run a player with knowledge about the world
    run_episode(world=world ,player=BfsPlayer()) 


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
