import math
import random


# upperBound: int, int -> float
# the size of the upper confidence bound for ucb1
def upperBound(step, numPlays):
   return math.sqrt(2 * math.log(step + 1) / numPlays)


# ucb1: int, (int, int -> float) -> generator
# perform the ucb1 bandit learning algorithm.  numActions is the number of
# actions, indexed from 0. reward is a function (or callable) accepting as
# input the action and producing as output the reward for that action
def ucb1(numActions, reward):
   payoffSums = [0] * numActions
   numPlays = [1] * numActions
   ucbs = [0] * numActions

   # initialize empirical sums
   for t in range(numActions):
      payoffSums[t] = reward(t,t)
      yield t, payoffSums[t], ucbs

   t = numActions

   while True:
      ucbs = [payoffSums[i] / numPlays[i] + upperBound(t, numPlays[i]) for i in range(numActions)]
      action = max(range(numActions), key=lambda i: ucbs[i])
      theReward = reward(action, t)
      numPlays[action] += 1
      payoffSums[action] += theReward

      yield action, theReward, ucbs
      t = t + 1


# Test UCB1 using stochastic payoffs for 10 actions.
def simpleTest():
   numActions = 10
   numRounds = 1000

   biases = [1.0 / k for k in range(5,5+numActions)]
   means = [0.5 + b for b in biases]
   deltas = [means[0] - x for x in means[1:]]
   deltaSum = sum(deltas)
   invDeltaSum = sum(1/x for x in deltas)

   bestAction = 0
   rewards = lambda choice, t: random.random() + biases[choice]

   cumulativeReward = 0
   bestActionCumulativeReward = 0

   t = numActions
   for (choice, reward, ucbs) in ucb1(numActions, rewards):
      cumulativeReward += reward
      bestActionCumulativeReward += reward if choice == bestAction else rewards(bestAction, t)
      regret = bestActionCumulativeReward - cumulativeReward
      regretBound = 8 * math.log(t + 5) * invDeltaSum + (1 + math.pi*math.pi / 3) * deltaSum

      #print("regret: %d\tregretBound: %.2f" % (regret, regretBound))

      t += 1
      if t >= numRounds:
         break

   return cumulativeReward


if __name__ == "__main__":
   print(simpleTest())
