import numpy as np
import pandas as pd
import re
import networkx as nx
from data.mol import Mol

"""
Helper functions to extend and format a dataframe with additional information
notably contacts, transmissions curves and adjacency matrix.
"""

class DFKeys:
    """
    Keys used in the dataframe (so that I don't have to remember them/write them wrong)
    """
    CONTACTS = "contacts"
    NB_CONTACTS = "nb_contacts"
    TRANSMISSIONS = "transmissions"
    MIN_TRANS_X = "min_trans_x"
    MAX_TRANS_X = "max_trans_x"
    MIN_TRANS_Y = "min_trans_y"
    MAX_TRANS_Y = "max_trans_y"
    ADJ_MATRIX = "adj_matrix"
    EIGVALS = "eigvals"
    EIGVECS = "eigvecs"

CONTACT_TYPES = {
    "oooooo": 0, 
    "xooooo": 1, 
    "xxoooo": 2, 
    "xoxooo": 3, 
    "xooxoo": 4
}

"""
Steps of the transmission curve abscises
"""
TRANSMISSION_STEP = 0.05
"""
Regular expression allowing for easy splitting of a contact strings
"""
SPLIT_CONTACT_REGEX = re.compile(r"\((?P<first>\d+),\s*(?P<second>\d+)\)")
def split_contacts(ln):
    """
    Slits a string of contacts into a pair tuples[int, int].
    """
    return list(map(lambda match: (int(match[1]), int(match[2])), SPLIT_CONTACT_REGEX.finditer(ln)))
SPLIT_TRANSMISSIONS_REGEX = re.compile(r"\(\s*\[\s*(?P<first>(?:np\.float64\(-?\d+\.\d+(?:e-?\d+)?\))(?:\s*,\s*np\.float64\(-?\d+\.\d+(?:e-?\d+)?\))*)\s*]\s*,\s*\[\s*(?P<second>(?:np\.float64\(-?\d+\.\d+(?:e-?\d+)?\))(?:\s*,\s*np\.float64\(-?\d+\.\d+(?:e-?\d+)?\))*)\s*]\s*\)")
NUMBER_REGEX = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")
def _split_transmission_pair(pair):
    X = np.fromiter(map(lambda match: np.float64(match.group()), NUMBER_REGEX.finditer(pair[0])), dtype=np.float64)
    Y = np.fromiter(map(lambda match: np.float64(match.group()), NUMBER_REGEX.finditer(pair[1])), dtype=np.float64)
    return (Y, min(X), max(X))
def split_transmissions(ln):
    """
    Splits a string of transmissions curves into a list of numpy arrays.
    """
    triplets_str = SPLIT_TRANSMISSIONS_REGEX.finditer(ln)
    pairs = map(lambda match: (match.group("first"), match.group("second")), triplets_str)
    return list(map(_split_transmission_pair, pairs))
SPLIT_ADJ_MATRIX_REGEX = re.compile(r"\n\s*")
def split_adjacency_matrix(ln):
    """
    Splits a string repr an adjacency matrix into a numpy array
    """
    lns = SPLIT_ADJ_MATRIX_REGEX.split(ln[1:-1])
    return np.array([[float(dt) for dt in ln[1:-1].split()] for ln in lns])

def add_contacts_list(df):
	"""
	Format and add the contacts list to the dataframe.
	"""
	df[DFKeys.CONTACTS] = df.loc[:, DFKeys.CONTACTS].apply(split_contacts)
	df[DFKeys.NB_CONTACTS] = df.loc[:, DFKeys.CONTACTS].apply(lambda x: len(x)*2)
    
def add_transmissions_curves(df):
	"""
	Format and add the transmissions curves and the minimum/maximum of said curves to the dataframe.
	"""
	trans = []
	minx = []
	maxx = []
	miny = []
	maxy = []
	for _, row in df.iterrows():
		tmp_trans = []
		tmp_minx = []
		tmp_maxx = []
		tmp_miny = []
		tmp_maxy = []
		for Y, mnx, mxx in split_transmissions(row.loc[DFKeys.TRANSMISSIONS]):
			tmp_trans.append(Y)
			tmp_minx.append(mnx)
			tmp_maxx.append(mxx)
			tmp_miny.append(np.min(Y))
			tmp_maxy.append(np.max(Y))
		trans.append(tmp_trans)
		minx.append(np.min(tmp_minx))
		maxx.append(np.max(tmp_maxx))
		miny.append(np.min(tmp_miny))	
		maxy.append(np.max(tmp_maxy))
	df[DFKeys.TRANSMISSIONS] = trans
	df[DFKeys.MIN_TRANS_X] = minx
	df[DFKeys.MAX_TRANS_X] = maxx
	df[DFKeys.MIN_TRANS_Y] = miny
	df[DFKeys.MAX_TRANS_Y] = maxy
 
def add_adjacency_matrix(df):
    df[DFKeys.ADJ_MATRIX] = df.loc[:, DFKeys.ADJ_MATRIX].apply(split_adjacency_matrix)
    
def extend_df(df):
    """
    Add and format the extra data we need in the dataframe
    """
    add_contacts_list(df)
    add_transmissions_curves(df)
    add_adjacency_matrix(df)
    
def develop_df(df):
	"""
	Develop the dataframe by adding an entry for each pair of contact for each molecule.
	"""
	df  = df.drop([DFKeys.MIN_TRANS_Y, DFKeys.MAX_TRANS_Y, DFKeys.NB_CONTACTS, DFKeys.EIGVALS, DFKeys.EIGVECS], axis=1)
	wdf = pd.DataFrame({colname: [] for colname in df.columns})
	for _, row in df.iterrows():
		for i, contact in enumerate(row.loc[DFKeys.CONTACTS]):
			new_row = row.copy()
			new_row[DFKeys.CONTACTS] = contact
			new_row[DFKeys.TRANSMISSIONS] = np.array(row.loc[DFKeys.TRANSMISSIONS][i])
			wdf = pd.concat([wdf, new_row.to_frame().T], ignore_index=True)
	return wdf

def gen_xyz(adj_matrix):
	"""
	Generate the xyz coordinates for the carbon atoms based on the adjacency matrix.

	Assume every atoms is a carbon atom in a hexagonal matrix with no island and with distance 1 with all of its neighbors
 
	returns: array of coordinates for each atom in the graph and a set of edges
	"""
	graph = nx.Graph(adj_matrix)
	cycles = nx.minimum_cycle_basis(graph)
	#print(f"cycles: {cycles}")
	coords = [None for _ in range(len(graph.nodes()))]
	edges = {}
	first = True
	#atoms = 0
	while len(cycles) > 0:
		if first:
			cycle = cycles.pop(0)
			first = False
		else:
			cycle = find_connected_cycle(cycles, edges)
		#print(f"\tcycle: {cycle}")
		a1 = None
		a2 = None
		horaire = False
		for anext in cycle:
			if coords[anext] is None:
				if a2 is None:# first atom in the cycle, potentially the first atom in the whole graph
				# since we align the cycle so that the first two atoms are an already placed edge, we can assume that if this atom has no coordinates yet, it is the first atom in the graph
					coords[anext] = np.array((0., 0., 0.)) # first atom in graph at origin
					#atoms += 1
					#print(f"{atoms}e atom !: {anext} ({coords[anext]})")
				elif a1 is None:
        			# Similar reasoning as above, but this time we assume that this is the second atom in the graph
					coords[anext] = np.array((1., 0., 0.)) # second atom in graph next to first
					#atoms += 1
					#print(f"{atoms}e atom !!: {anext} ({coords[anext]})")
				else:
					v1 = coords[a1]
					v2 = coords[a2]
					d = v2-v1
					angle = -60 if horaire else 60
					costdx = np.cos(np.deg2rad(angle)) * d[0]
					sintdx = np.sin(np.deg2rad(angle)) * d[0]
					costdy = np.cos(np.deg2rad(angle)) * d[1]
					sintdy = np.sin(np.deg2rad(angle)) * d[1]
					d_ = np.array((costdx - sintdy, sintdx + costdy, 0.))
					coords[anext] = v2 + d_
					#atoms += 1
					#print(f"{atoms}e atom: {anext} ({coords[anext]})")
			elif a1 is None and a2 is not None: # Détecter la direction du cycle
				if (a2, anext) in edges:
					horaire = not edges[(a2, anext)]
				elif (anext, a2) in edges:
					horaire = edges[(anext, a2)]
				else:
					horaire = False
    
			a1 = a2
			a2 = anext
			if a1 is not None and a2 is not None and (a1, a2) not in edges and (a2, a1) not in edges:
				edges[(a1, a2)] = horaire
		# Connect the last atom in the cycle to the first one
		e = (cycle[-1], cycle[0])
		if e not in edges and (cycle[0], cycle[-1]) not in edges:
			edges[e] = horaire
		#print(f"\edges: {edges}")
   
	return coords, edges.keys()

def find_connected_cycle(cycles, edges):
	"""
	Find a cycle that is connected to the edges already found.
	"""
	for i, c in enumerate(cycles):
		for e in range(len(c)):
			n1 = c[e]
			n2 = c[(e + 1) % len(c)]
			if (n1, n2) in edges:
				cycle = cycles.pop(i)
				# rotate the cycle so that the connected edge is at the start
				cycle_ = cycle[e:] + cycle[:e]
				#print(f"\t\t{cycle} -> {cycle_} !")
				return cycle_
			elif (n2, n1) in edges:
				cycle = cycles.pop(i)
				cycle_ = cycle[e:] + cycle[:e]
				cycle_ = list(reversed(cycle_))
				cycle_ = cycle_[-2:] + cycle_[:-2]
				#print(f"\t\t{cycle} -> {cycle_} !!")
				return cycle_

	return None

def gen_mol(adj_matrix):
    """
    Generate the Mol object and edges from the adjacency matrix.
    """
    coords, edges = gen_xyz(adj_matrix)
    mol = Mol(list(map(lambda c: ("C", c[0], c[1], c[2]), coords)))
    return mol, edges

def get_contact_and_rotation(knot, contact):
	"""
	return the contact type and its orientation 
	"""
	ring_idx = list(map(lambda a: a.index, knot.atoms))
	if contact[0] in ring_idx and contact[1] in ring_idx:
		first = ring_idx.index(contact[0])
		second = ring_idx.index(contact[1])
		first, second = min(first, second), max(first, second)
		dist = second - first
		if dist == 1:
			contact_type = CONTACT_TYPES["xxoooo"]
			rotation = first
		elif dist == 5:
			contact_type = CONTACT_TYPES["xxoooo"]
			rotation = second
		elif dist == 2:
			contact_type = CONTACT_TYPES["xoxooo"]
			rotation = first
		elif dist == 4:
			contact_type = CONTACT_TYPES["xoxooo"]
			rotation = second
		elif dist == 3:
			contact_type = CONTACT_TYPES["xooxoo"]
			rotation = first
	elif contact[0] in ring_idx:
		contact_type = 1
		rotation = ring_idx.index(contact[0])
	elif contact[1] in ring_idx:
		contact_type = 1
		rotation = ring_idx.index(contact[1])
	else:
		contact_type = 0
		rotation = None
	if rotation is not None:
		center = np.array(knot.get_coord())
		dest = np.array(knot.atoms[rotation].get_coord())
		rotation = dest-center
	else:
		rotation = np.array((0., 0., 0.))
	return contact_type, rotation

def get_contacts_and_rotations(knots, contact):
	"""
	Return the contact types and their orientations for each knot.
	"""
	contacts = []
	rotations = []
	for knot in knots:
		cont, rotation = get_contact_and_rotation(knot, contact)
		contacts.append(cont)
		rotations.append(rotation)
	return contacts, rotations

def prepare_transmission_curve(trans, trans_min_x, trans_max_x, tot_min_x, tot_max_x, tot_min_y, tot_max_y):
	"""
	Prepare the transmission curve by padding and normalizing it and return the padded curve and the mask
	"""
	#print(f"trans_min_x: {trans_min_x}, tot_min_x: {tot_min_x}")
	pad_before = int(np.ceil(abs(trans_min_x - tot_min_x)/TRANSMISSION_STEP))
	l = int(np.ceil((tot_max_x - tot_min_x)/TRANSMISSION_STEP)) + 1
	pad_after = l - len(trans) - pad_before
	#print(f"pad_before: {pad_before}, pad_after: {pad_after}, l: {l}")
	mask = np.pad(np.ones(len(trans)), (pad_before, pad_after))
	norm = np.max([np.abs(tot_min_y), np.abs(tot_max_y)])
	padded_trans = np.pad(trans/norm, (pad_before, pad_after))
	return padded_trans, mask