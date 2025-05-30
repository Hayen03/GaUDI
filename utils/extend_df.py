import numpy as np
import pandas as pd
import re

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
	df  = df.drop([DFKeys.MIN_TRANS_Y, DFKeys.MAX_TRANS_Y, DFKeys.NB_CONTACTS, DFKeys.EIGVALS, DFKeys.EIGVECS], axis=1)
	wdf = pd.DataFrame({colname: [] for colname in df.columns})
	for _, row in df.iterrows():
		for i, contact in enumerate(row.loc[DFKeys.CONTACTS]):
			new_row = row.copy()
			new_row[DFKeys.CONTACTS] = contact
			new_row[DFKeys.TRANSMISSIONS] = row.loc[DFKeys.TRANSMISSIONS][i]
			wdf = pd.concat([wdf, new_row.to_frame().T], ignore_index=True)
	return wdf