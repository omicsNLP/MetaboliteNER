from datetime import datetime

def _norm(a):
	s = sum(a)
	if s == 0: s = 1
	return [i/s for i in a]


class ExtractedRandomForest(object):
	"""
	A Random Forest, extracted from a scikit-learn object. This works for chemlistem - not tested
	for other things in general.
	
	Members:
	ets - extracted trees. A list of dictionaries.
	"""
	
	def __init__(self, rf):
		"""
			Args:
				either a scikit-learn random forest, or an ets object previously made
				by this class (probably serialized and deserialized).
		"""
		if(type(rf)) == list:
			self.ets = rf
		else:
			self.ets = _extract_random_forest(rf)
	
	def predict_proba(self, data):
		"""
		Gives 'probabilities' for each outcome given the data. Classifies a single item, not multiple (unlike scikit-learn).
		
		Args:
			data: a list (or numpy array) of features
			
		Returns:
			a list of probabilities (should sum to 1) for the various outcomes.
		"""
		vals = []
		for et in self.ets:
			node = 0
			while(et[0][node] != -2):
				v = data[et[0][node]]
				if v <= et[1][node]:
					node = et[2][node]
				else:
					node = et[3][node]	
			vals.append(et[4][node])
		s = [0 for i in vals[0]]
		for i in range(len(vals)):
			for j in range(len(s)):
				s[j] += vals[i][j]
		gtot = sum(s)
		if gtot == 0: gtot = 1
		return [i/gtot for i in s]
					
	
def _extract_tree(est):
	feats = [int(i) for i in est.feature]
	thresh = [float(i) for i in est.threshold]
	lc = [int(i) for i in est.children_left]
	rc = [int(i) for i in est.children_right]
	values = [_norm(i[0]) for i in est.value]
	return [feats, thresh, lc, rc, values]
	
def _extract_random_forest(rf):
	ets = []
	for en in range(len(rf.estimators_)):
		est = rf.estimators_[en].tree_
		et = _extract_tree(est)
		#print(et)
		ets.append(et)
	return ets
	
def _apply_et(et, data):
	node = 0
	data = data[0]
	while(et[0][node] != -2):
		v = data[et[0][node]]
		if v <= et[1][node]:
			node = et[2][node]
		else:
			node = et[3][node]
	return node
	

