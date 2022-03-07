from Cards import *

class Variable:
	def __init__(self, default) -> None:
		self.default = default
		self.value = default
		
		# Whether value has been determined in this solve.
		self.floating = True

		# Whether this should get reset or not.
		self.fixed = False

class Component(): #Card
	def __init__(self):
		self.processed = False
		self.vars = []

		# Return state to unsolved
	def reset(self):
		self.processed = False
		for var in self.vars:
			if var.fixed == False:
				var.floating = True

class Control(Card):
    pass

# Non-card based class carries state like pressure/voltage and also a flow like current or m_dot
class Connector:
	# A reference to another connector.
	conn = None


# -----------------------

class Resistor(Component):
	def __init__(self):
		super().__init__()
		self.V1 = Variable(0)
		self.V2 = Variable(0)
		self.I = Variable(0)
		self.R = Variable(1)
		self.vars = [self.V1, self.V2, self.I, self.R]

	# This is just the voltage error though. And now also current error.
	def error(self):
		#return (self.V1.value - self.V2.value - self.I.value * self.R.value)**2
		return (self.I.value + (self.V1.value - self.V2.value)/self.R.value)**2
	
	# Depends on the law of the card and what inputs it gets in and puts out.
	def process(self):
		# If all variables are defined give an error function term.
		if all( [v.floating == False for v in self.vars ]):
			self.processed == True

		# if one is floating use the formula to calculate it.
		
			

class Junction(Component):
	def __init__(self):
		super().__init__()
		self.V = Variable(0)
		self.I1 = Variable(0)
		self.I2 = Variable(0)
		self.I3 = Variable(1)
		self.vars = [self.V, self.I1, self.I2, self.I3]

	def error(self):
		return (self.I1 + self.I2 + self.I3)**2

class Wire(Connector):

	#Implement V and I as properties where a setter will also set the connection if it is not none.

	# Voltage
	V = None
	# Current (defined positive coming out of component)
	I = None