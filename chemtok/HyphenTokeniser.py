"""
Looks for places where tokens can be split on hyphens. This is used by ChemTokeniser, and
needs to be a separate class because it needs some word lists.
 
Strategy: look for hyphens, from the rear backwards, leaving two characters padding
on the front and rear of the token.
Sequentially apply a series of rules:
 
1) If what's after the hyphen is on the splitSuffixes list, split!
2) If the letters before the hyphen are on the noSplitPrefixes list, don't split!
3) If the two characters in front of the hyphen are lowercase, and all characters after, split!
4) Don't split!

Used by ChemTokeniser - not intended for use elsewhere.

"""

import re
from . import StringTools

_splitSuffixes = set()
_noSplitPrefixes = set()

_suffixPrefixPattern = re.compile("mono|di|tri|tetra|penta|hexa|hepta|un|de|re|pre")
_suffixSuffixPattern = re.compile(".*(ing|[ua]ted|i[sz]ed|ase)$") # Use in clm version

_splitSuffixes = {'related', 'promoted', 'soluble', 'linked', 'isomer', 'saturated', 'bonded', 'blood', 'SAMs', 'transferrin', 'analogous', 'layer', 'exchange', 'independent', 'treated', 'cation', 'enzyme', 'shaped', 'guest', 'derivatives', 'BSA', 'labelled', 'group', 'tethered', 'coordinated', 'biomonitor', 'intermediate', 'type', 'mediated', 'treatment', 'catalysed', 'lactoferrin', 'substituent', 'the', 'protein', 'protected', 'assisted', 'disubstituted', 'selectivity', 'configuration', ',', 'form', 'enriched', 'containing', 'only', 'like', 'catalyzed', 'octahedra', 'chains', 'initiated', 'moiety', 'ligand', 'binding', 'units', 'strategy', 'compounds', 'bond', 'bound', 'bonding', 'ions', 'bridged', 'modified', 'filled', '15', 'stabilised', 'derived', 'position', 'terminated', 'protecting', 'PAGE', 'bonds', 'complexes', 'substituted', 'hyperaccumulator', 'donor', 'coating', 'terminal', 'isomers', 'coated', 'peptide', 'based', 'complex', 'unsaturated'}

_clmSplitSuffixes = {'related', 'promoted', 'soluble', 'linked', 'isomer', 'saturated', 'bonded', 'blood', 'SAMs', 'transferrin', 'analogous', 'layer', 'exchange', 'independent', 'treated', 'cation', 'enzyme', 'shaped', 'guest', 'derivatives', 'BSA', 'labelled',
'group', 'tethered', 'coordinated', 'biomonitor', 'intermediate', 'type', 'mediated', 'treatment', 'catalysed', 'lactoferrin', 'substituent', 'the', 'protein', 'protected', 'assisted', 'disubstituted', 'selectivity', 'configuration', ',', 'form', 'enriched',
'containing', 'only', 'like', 'catalyzed', 'octahedra', 'chains', 'initiated', 'moiety', 'ligand', 'binding', 'units', 'strategy', 'compounds', 'bond', 'bound', 'bonding', 'ions', 'bridged', 'modified', 'filled', '15', 'stabilised', 'derived',
'position', 'terminated', 'protecting', 'PAGE', 'bonds', 'complexes', 'substituted', 'hyperaccumulator', 'donor', 'coating', 'terminal', 'isomers', 'coated', 'peptide', 'based', 'complex', 'unsaturated', "atom", "atoms", "analog", "analogs", "analogue",
"analogues", "induced", "terminal", "agonist", "agonistic", "agonists", "antagonist", "antagonistic", "antagonists", "attached", "dimer", "dimers", "responsive", "responsiveness", "linked", "link", "linkage", "linkages", "links", "receptor", "receptors",
"type", "types", "typed", "substituted", "donor", "donors", "terminal", "terminus", "garlic", "bound", "dependent", "independent", "inhibitor", "inhibitors", "inducer", "inducers", "ingredient", "tree", "trees", "and", "bond", "bonds", "component",
"sensitive", "bit", "S-transferase", "substituent", "polymer", "polymers", "lower", "upper"}

_noSplitPrefixes = {'kappa', 'cyclo', 'mono', 'ent', 'dl', 'tele', 'sec', 'hydro', 'rho', 'yl', 'sym', 'epsilon', 'nu', 'glucono', 'nor', 'cis', 'epi', 'myo', 'n-boc', 'catena', 'phi', 'trans', 'iso', 'neo', 'anti', 'sigma', 'nido', 'closo', 'amino', 'ortho', 'beta', 'gamma', 'unsym', 'eta', 'keto', 'muco', 'syn', 'chi', 'tert', 'triangulo', 'zeta', 'as', 'iota', 'scyllo', 'gluco', 'tau', 'meta', 'mu', 'sn', 'allo', 'homo', 'lambda', 'omicron', 'threo', 'bis', 'exo', 'semi', 'xi', 'omega', 'pi', 'alpha', 'theta', 'oxy', 'psi', 'upsilon', 'para', 'endo', 'non', 'meso', 'de', 'delta', 'arachno'}

_splitPrefixes = {'NAD', 'NADH', 'NADP', 'NADPH', 'ATP', 'AMP', 'LDL', 'HDL', 'PEG'} # Use in clm version

_minPrefixLength = min([ len(p) for p in _noSplitPrefixes])
_maxPrefixLength = max([ len(p) for p in _noSplitPrefixes])

_splitOnEnDash = True


def _indexOfSplittableHyphen(s, clm=False):
	try:
		return _indexOfSplittableHyphenInternal(s)
	except Exception as e:
		raise(e)
		return -1
 
def _indexOfSplittableHyphenInternal(s, clm=False):
	"""
	Works out where, if anywhere, to split a string which may contain a hyphen.

	Looks for places where tokens can be split on hyphens. This is used by ChemTokeniser, and
	needs to be a separate class because it needs some word lists.
	
	Strategy: look for hyphens, from the rear backwards, leaving two characters padding
	on the front and rear of the token.
	Sequentially apply a series of rules:
	 
	1) If what's after the hyphen is on the splitSuffixes list, split!
	2) If the letters before the hyphen are on the noSplitPrefixes list, don't split!
	3) If the two characters in front of the hyphen are lowercase, and all characters after, split!
	4) Don't split!
	
	Args:
		s: The string to analyse.
		clm: Whether to apply special rules for chemlistem.

	Returns:
		The index of the hyphen to split at, or -1.
	"""
 
	balancedBrackets = StringTools.bracketsAreBalanced(s)
	rr = range(len(s) - 2, 1, -1) if clm else range(len(s) - 2, 0, -1)
	
	for i in rr:
		if s[i] in StringTools.hyphens:
			# Don't split on tokens contained within brackets
			if balancedBrackets and not StringTools.bracketsAreBalanced(s[i + 1:]):
				continue 
			#  Split on en-dashes?
			if _splitOnEnDash and s[i] == StringTools.enDash:
				return i
			#  Always split on em-dashes
			if s[i] == StringTools.emDash:
				return i
			#  Suffixes?
			suffix = s[i+1:].lower()
			if suffix in (_clmSplitSuffixes if clm else _splitSuffixes):
				# if(!wouldTok) System.out.printf("%s %d\n", s, i);
				return i
			if clm:
				if _suffixSuffixPattern.match(suffix):
					#print(suffix, file=sys.stderr)
					return i
			m = _suffixPrefixPattern.match(suffix); 
			if m:
				suffix = suffix[m.end():]
			while len(suffix) >= 3:
				if suffix in _splitSuffixes:
					# if(!wouldTok) System.out.printf("%s %d\n", s, i);
					return i
				suffix = suffix[:-1]
			if clm:
				prefix = s[:i]
				# Prefixes with caps to split on
				if prefix in _splitPrefixes: return i			
				
			#  No suffix? Then don't examine hyphens in position 1
			if i == 1:
				continue								
			  
			  # Prefixes
			noSplit = False
 #		   for(int j=minPrefixLength;j<=maxPrefixLength && j<=i;j++) {

			for j in range(_maxPrefixLength , min(_maxPrefixLength,i)):
				prefix = s[i-j:i].lower()
				if prefix in _noSplitPrefixes:
					noSplit = True
					break
			if noSplit:
				break
			#  Check for lowercase either side of the token 
			if re.match("[a-z][a-z][" + StringTools.hyphens + "][a-z]+$", s[i - 2:]):
				# if(!wouldTok) System.out.printf("%s %d\n", s, i);
				return i
			# if(wouldTok) System.out.printf("* %s %d\n", s, i);

	return -1

# clm	
def _indexOfSplittableComma(s):
	for i in range(len(s) - 1, 0, -1):
		if s[i] == ",":
			prefix = s[:i]
			suffix = s[i+1]
			if "-" not in suffix and "-" not in prefix and "[" not in prefix and "(" not in prefix:
				#print(prefix, suffix, file=sys.stderr)
				return i
	return -1
	