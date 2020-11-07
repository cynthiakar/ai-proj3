# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #a for loop that will run through until the iteration is reached 
        for itera in range(self.iterations):
            # inital counter, which is a dict with default 0
            current_count = util.Counter()
            # for loop to run through the current states
            for s in self.mdp.getStates():
                # set maxNum 
                maxNum = float("-inf")
                # loop through the possible actions/path taken in the current state
                #initalized actions to getPossibleactions of the states
                actions = self.mdp.getPossibleActions(s) 
                for path in actions:
                    # setting the Q value
                    valueQ = self.computeQValueFromValues(s,path)
                    # if maxNum is less that Q value then set maxNum to valueQ 
                    if( maxNum < valueQ):
                        maxNum = valueQ
                    # else set the current counter of the state of the maxNum 
                    current_count[s] = maxNum
            # set the current value to current count 
            self.values = current_count

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # initalize total counter 
        total_counter = 0 
        # loop through the possible path/actions
        for st, poss in self.mdp.getTransitionStatesAndProbs(state, action):
            # value for the reward 
            reward = self.mdp.getReward(state, action, st)
            # equation to obtiain the total 
            total_counter = total_counter + (poss * (reward + self.discount * self.values[st]))
        # return total
        return total_counter 

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # initalize best_action to None
        bestAct = None
        # initalize the possibleActions of State to action
        actions = self.mdp.getPossibleActions(state)
        #initalize max value to float("-inf")
        maxNum = float("-inf")
        #loop through the possible paths in the state
        for path in actions:
            #obtain the Q value from computeQValueFromValues 
            valueQ = self.computeQValueFromValues(state,path)
            # if Q value is greater than maxNUm  
            if maxNum < valueQ:
                #then set maxNUm to valueQ 
                maxNum = valueQ
                #and best_action to path
                bestAct = path
        # else return best_action
        return bestAct

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #initalize location to the states
        location = self.mdp.getStates()
        #compute the length og the states/path
        lengthOfPath = len(self.mdp.getStates())
        # for itera in the iterations
        # obtian the position 
        for itera in range(self.iterations):
            #by taking the location value of the mod itera & lengthofpath
            position = location[itera % lengthOfPath]
            # if its a non-terminal:
            # create a values list
            if not self.mdp.isTerminal(position):
                values = []
                # initalize the actions to the the possibleActions(position)
                actions = self.mdp.getPossibleActions(position)
                # loop through the possible actions of the position
                for path in actions:
                    # compute the qValueFromValues of the position and path
                    valueQ = self.computeQValueFromValues(position,path)
                    # append to the qvalue to values list
                    values.append(valueQ)
                # set the max of the values list to the values' position
                self.values[position] = max(values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

