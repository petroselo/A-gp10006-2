## Create system

from operator import truediv
from GradDescent import descend
from System_cards import *
import numpy as np

#Assume cards and their positions have been determined and their connections as well.


# Stores the whole setup to be solved.
cards = []

# Vector of 'Variables'
x = []

#r1
r1 = Resistor()
cards.append(r1)
r1.V1.fixed=True
r1.V1.value=0
r1.R.fixed = True
r1.R.value = 5

#r2
r2 = Resistor()
cards.append(r2)
r2.R.fixed = True
r2.R.value = 10


#r3
r3 = Resistor()
cards.append(r3)
r3.R.fixed = True
r3.R.value = 10

#r4
r4 = Resistor()
cards.append(r4)
r4.V1.fixed=True
r4.V2.value=12
r4.R.fixed=True
r4.R.value=20

#j1
j1 = Junction()
cards.append(j1)

#j2
j2 = Junction()
cards.append(j2)


# Define a numpy vector of the independent variables and set initial value.
npx0 = np.zeros(len(x))
for i in range(len(x)):
	npx0[i] = x[i].default



# do  state calculation
def erf(npx):
	for i in range(len(npx)):
		x[i].value = npx[i]

	for card in cards:
		card.reset()

	for card in cards:
		if card.processed == False: 
			card.process()

	error = sum([card.error() for card in cards])
	return error


npx = descend(erf, npx0)





