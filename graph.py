from matplotlib import pyplot as plt
import numpy

def column(A, j):
    return [A[i][j] for i in range(len(A))]

def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]


def regretWeightsGraph(filename, title):
   with open(filename, 'r') as infile:
      lines = infile.readlines()

   lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
   data = transpose(lines)

   regret = numpy.array(data[0])
   regretBound = numpy.array(data[1])
   xs = numpy.array(list(range(len(data[0]))))

   ax1 = plt.subplot(111)
   plt.ylabel('Cumulative Regret')
   ax1.plot(xs, regret)
   ax1.plot(xs, regretBound)
   plt.title(title)

   plt.show()


regretWeightsGraph('first-example.txt', "Regret of UCB1\n10 actions, 1 million rounds")

