# Offline search and Wumpus

## Problem description
The problem  is to find the best sequence of actions (if one is available) to make the agent (a hunter) grab the gold ingot in the given environment and eventually climb out of the world.
In this case the world is modeled as a grid map with some pits, where the agent would die if entering them. The agent also has an orientation (either North, East, South or West) and can only move following the direction he is heading to. To make the agent change direction there is a Left and Right action to make the agent rotate. The agent also has one arrow he can use to kill the wumpus, if necessary, in order to shoot the monster and move inside its location without dying.

## Running requirements
To run the code you have to create an anaconda environment with the configuration file found in environment.yml and then activate it to run the code.
The required commands to make it work are the following:
- conda create env -f environment.yml
- jupyter lab

Contributors:  Artem Merinov & Rigels Hita
