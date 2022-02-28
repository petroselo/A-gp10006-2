from Cards import *

class Variable:
	def __init__(self, default) -> None:
		self.default = default
		self.value = default
		self.floating = True

class Component(Card):
    pass

class Control(Card):
    pass

# Non card based class carries state like pressure/voltage and also a flow like current or m_dot
class Connector:
	pass


# -----------------------

class Resistor(Component):
	def __init__(self, fid, raw_corners):
		super().__init__(fid, raw_corners)
		self.V1 = Variable(0)
		self.V2 = Variable(0)
		self.I = Variable(0)
		self.R = Variable(1)
		self.vars = [self.V1, self.V2, self.I, self.R]

	# This is just the voltage error though.
	def error(self):
		return self.V1.value - self.V2.value - self.I.value * self.R.value
	

class Junction(Component):
	pass

class Wire(Connector):
	pass