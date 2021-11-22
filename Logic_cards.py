import numpy as np
import constants as C


class Logic_gate:
	def __init__(self, name, n_in, n_out, fn): # plus some kind of function here to define a mapping between inputs and outputs.
		self.n_in = n_in
		self.n_out = n_out
		self.mapping = fn

# 0 -> 0
# 1 -> 1
# 2 -> floating
# 3 -> Unknowable

class Input:
	def __init__(self, pos):
		self.pos = pos
		self.conn = None


class Output:
	def __init__(self, pos, parent):
		self.pos = pos
		self.val = 0
		self.parent_card = parent

ident = lambda a:[C.UNKNOWABLE]

# Lambda takes a list of input values of length n_in and returns a list of output values of length n_out
logic_card_dict = {
	20: Logic_gate('0',    0, 1, lambda a: [0]),
	21: Logic_gate('1',    0, 1, lambda a: [1]),
	22: Logic_gate('NOT',  1, 1, lambda a: [not  a[0] ]),
	23: Logic_gate('AND',  2, 1, lambda a: [int( a[0] and a[1] )]),
	24: Logic_gate('OR',   2, 1, lambda a: [int( a[0] or a[1]  )]),
	25: Logic_gate('XOR',  2, 1, lambda a: [0] if a[0]==a[1] else [1]),
	26: Logic_gate('NAND', 2, 1, lambda a: [int(not( a[0] and a[1] ))]),
	27: Logic_gate('XNOR', 2, 1, lambda a: [1] if a[0]==a[1] else [0])
}


class Logic_card:

	def __init__(self, fid, corners):

		self.gate = logic_card_dict[fid] # ->  n_in, n_out and a function
		self.corners = corners

		self.position = 0.25 * corners.sum( axis=0 )
		self.xvec = 0.25 * (corners[1] + corners[2] - corners[0] - corners[3])
		self.yvec = 0.25 * (corners[0] + corners[1] - corners[2] - corners[3])
		self.scale = (np.linalg.norm(self.xvec) + np.linalg.norm(self.yvec))
		self.snap_distance = self.scale * 2

		# Define inputs and outputs
		inp_rng = np.array([0]) if self.gate.n_in == 1 else np.linspace(-1, 1, num=self.gate.n_in)
		outp_rng = np.array([0]) if self.gate.n_out == 1 else np.linspace(-1, 1, num=self.gate.n_out)
		self.inps, self.outps = [], []
		for i in inp_rng:
			self.inps.append(Input(self.position - 2*self.xvec + i*self.yvec))
		for i in outp_rng:
			self.outps.append(Output(self.position + 2*self.xvec + i*self.yvec, self))

		self.evaluated = False

	def evaluate(self):
		# look at inputs. if any floating evaluate them first
		for i in self.inps:
			if i.conn is not None and i.conn.val == C.FLOATING:
				i.conn.parent_card.evaluate()
		# if any are None or unknowable set all outputs as unknowable
		for i in self.inps:
			if i.conn is None or i.conn.val == C.UNKNOWABLE:
				for o in self.outps: o.val = C.UNKNOWABLE
				self.evaluated = True
				return

		# ?? maybe check if being called in a recursive loop from this function before??
		# Apply mapping function between inputs and ouputs
		outvals = self.gate.mapping([i.conn.val for i in self.inps])

		# Set output values
		for (outp, val) in zip(self.outps, outvals):
			outp.val = val

		self.evaluated = True


		# Input positions


		# N
