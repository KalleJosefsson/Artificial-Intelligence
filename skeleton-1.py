import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
import copy

env: ConnectFourEnv = gym.make("ConnectFour-v0")


SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["ka5532jo-s", "da22test-s2"] # TODO: fill this list with your stil-id's
BIG_NUMBER = 10**9
MAX_DEPTH = 5

def call_server(move):
   res = requests.post(SERVER_ADDRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADDRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))
  

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done


    

#implemented by Kalle aka me ka5532jo-s
def student_move(env):
   #return random.choice([0, 1, 2, 3, 4, 5, 6])
   avmoves = env.available_moves()
   best_action = None
   best_score = -BIG_NUMBER
   
   for action in list(avmoves):
      env_copy = copy.deepcopy(env)
      _,score,done, _ = env_copy.step(action)
      if done:          
         return action
     

      if not done:
         score = min_player(env_copy,1,best_score,BIG_NUMBER)
         if score > best_score:
            best_score = score
            best_action = action
   return best_action      
      

   
   


   """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """

def EVAL(env):
   eval_matrix = [[3,4,5,7,5,4,3], 
                  [4,6,8,10,8,6,4], 
                  [5,8,11,13,11,8,5], 
                  [5,8,11,13,11,8,5], 
                  [4,6,8,10,8,6,4], 
                  [3,4,5,7,5,4,3]]
   eval_score = eval_matrix*env.board
   eval_score = eval_score.sum()
   return eval_score
   
   

def SCORE(env):
    return EVAL(env)


# Create my min_player
def min_player(env,depth,alpha,beta):
   if env.is_win_state():
      return 10000 #SCORE(env)
   if depth == MAX_DEPTH:
      return SCORE(env)
   env.change_player()
   min_score =BIG_NUMBER
   for a in env.available_moves():
      env_copy = copy.deepcopy(env)
      _, score, done, _ = env_copy.step(a)
      if done and score == 1:
         min_score = -10000
      if not done:
         score = max_player(env_copy, depth + 1,alpha,beta)
         min_score = min(min_score, score)
         beta = min(beta,min_score)
         if min_score <= alpha:break
   return min_score  




   
# Create my max_player
def max_player(env,depth,alpha,beta):
   if env.is_win_state():
      return -10000 #SCORE(env)
   if depth == MAX_DEPTH:
      return SCORE(env)
   env.change_player()
   max_score = -BIG_NUMBER
   
   for a in env.available_moves():
      env_copy= copy.deepcopy(env)
      _,score,done,_ = env_copy.step(a)
      if done and score == 1:
         max_score = score*10000
      if not done:
         score = min_player(env_copy,depth+1,alpha,beta)
         max_score = max(max_score,score)
         alpha = max(alpha,max_score)
         if max_score >= beta: break
   return max_score
 
   
      


def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      # reset env to state from the server (if you want to use it to keep track)
      env.reset(board=state)
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(env) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         # reset env to state from the server (if you want to use it to keep track)
         env.reset(board=state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
