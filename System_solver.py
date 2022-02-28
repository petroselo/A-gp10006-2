## Create system

from System_cards import *

#Assume cards and their positions have been determined and their connections as well.



cards = []

#r1
r1 = Resistor()

cards.append(r1)

#r2

#r3

#r4

#j1

#j2

# Define a vector of the independent variables.

# do  state calculation
error = 0

for card in cards:
	card.processed = False

for card in cards:
	if card.processed: continue
	# If all variables are defined give an error function term.
	

	# Else go 

# get error then reset over and over.



