import re
from . import StringTools
from . import HyphenTokeniser

class Token:
	"""
	A word or suchlike in chemical text.
	
	"Public" fields: 
	
	value: the string value of the token
	start: the start offset of the token
	end: the end offset of the token
	tokeniser: the parent chemtokeniser
	"""
	_oxidationStatePattern = re.compile("\\((o|i{1,4}|i{0,3}[xv]|[xv]i{0,4})\\)" + "$" , re.IGNORECASE) ###
	_oxidationStateEndPattern = re.compile(".*\\((o|i{1,4}|i{0,3}[xv]|[xv]i{0,4})\\)" + "$" , re.IGNORECASE)###
	_trademarkPattern = re.compile(".+?(\\((TM|R)\\)|\\(\\((TM|R)\\)\\))"+ "$" )###
	_primesRe = "[" + StringTools.primes + "]*"
	_locantRe = "(\\d+" + _primesRe + "[RSEZDLH]?|" + "\\(([RSEZDLH\u00b1]|\\+|" + StringTools.hyphensRe + ")\\)|" + "[DLRSEZ]|" + "([CNOS]|Se)\\d*|" + "\\d*[" + StringTools.lowerGreek + "]|" + "cis|trans|o(rtho)?|m(eta)?|p(ara)?|asym|sym|sec|tert|catena|closo|enantio|ent|endo|exo|" + "fac|mer|gluco|nido|aci|erythro|threo|arachno|meso|syn|anti|tele|cine" + ")" + _primesRe
	_prefixPattern = re.compile("(" + _locantRe + "(," + _locantRe + ")*)" + "[" + StringTools.hyphens + "](\\S*)" + "$" ) ###
	_elementSymRe = "(Zr|Zn|Yb|Y|Xe|W|V|U|Tm|Tl|Ti|" + "Th|Te|Tc|Tb|Ta|Sr|Sn|Sm|Si|Sg|Se|Sc|Sb|S|Ru|Rn|Rh|Rf|Re|Rb|" + "Ra|Pu|Pt|Pr|Po|Pm|Pd|Pb|Pa|P|Os|O|Np|No|Ni|Ne|Nd|Nb|Na|N|Mt|" + "Mo|Mn|Mg|Md|Lu|Lr|Li|La|Kr|K|Ir|In|I|Hs|Ho|Hg|Hf|He|H|Ge|Gd|" + "Ga|Fr|Fm|Fe|F|Eu|Es|Er|Dy|Ds|Db|Cu|Cs|Cr|Co|Cm|Cl|Cf|Ce|Cd|" + "Ca|C|Br|Bk|Bi|Bh|Be|Ba|B|Au|At|As|Ar|Am|Al|Ag|Ac)"
	_splitEqualsRe = re.compile("[RAXYM]\\d*(=)")

	#  E001 is a double bond character in RSC papers
	_bondCharRe = "(" + StringTools.hyphensRe + "|" + StringTools.midElipsis + "|=|\ue001)"
	_atomInBondRe = "(" + _elementSymRe + "\\d*" + _primesRe + "|" + _elementSymRe + "\\(\\d+\\))"
	_bondPattern = re.compile(_atomInBondRe + "(" + _bondCharRe + _atomInBondRe + ")+" + "$" )
	
	def __init__(self,value, startval, endval, ct):
		"""
		Makes a Token, with specified start and end positions.
	
		Args:
			value: The string value of the token.
			start: The start offset of the token.
			end: The end offset of the token.
			tokr: The ChemTokeniser that holds the token.
		
		"""
		self.value = value
		self.start = int(startval)
		self.end = int(endval)
		self.tokeniser = ct
	  

	def _split(self, clm=False):
		tokenList = self._splitInternal(clm)
		if tokenList == None:
			return None
		goodTokens = 0
		for t in tokenList:
			if t.end - t.start > 0:
				goodTokens += 1
		if goodTokens > 1:
			return tokenList
		return None

	def splitAt(self, splitOffset0, splitOffset1 = None):
		"""
		Split a token at a given offset, or pair of offsets
		
		Args:
			splitOffset0: the position in the sentence
			splitOffset1: another position in the sentence, or null
		"""
		if splitOffset1 == None:	
			internalOffset = splitOffset0 - self.start
			tokens = list()
			tokens.append(Token(self.value[:internalOffset], self.start, splitOffset0, self.tokeniser))
			tokens.append(Token(self.value[internalOffset:], splitOffset0, self.end, self.tokeniser))
			return tokens
		else:
			internalOffset0 = splitOffset0 - self.start
			internalOffset1 = splitOffset1 - self.start
			tokens = list()
			tokens.append(Token(self.value[:internalOffset0], self.start, splitOffset0, self.tokeniser))
			tokens.append(Token(self.value[internalOffset0:internalOffset1], splitOffset0, splitOffset1, self.tokeniser))
			tokens.append(Token(self.value[internalOffset1:], splitOffset1, self.end, self.tokeniser))
			return tokens

	
	def _splitInternal(self, clm=False):
		middleValue = ""
		if len(self.value) > 2:
			middleValue = self.value[:-1]
		#  Don't split special lexicon entries 
		if self.value.startswith("$"):
			return None
		#  One-character tokens don't split! 
		if self.end - self.start < 2:
			return None
		  
		firstchar = self.value[0]
		lastchar = self.value[-1]

		  
		if self.value == "--":
			return None
		#  Preserve oxidation states whole - don't eat for brackets 
		# } 
		if self._oxidationStatePattern.match(self.value):
			return None
			#  Split unmatched brackets off the front 
		if firstchar in "([{" and (StringTools.isBracketed(self.value) or StringTools.isLackingCloseBracket(self.value)):
			return self.splitAt(self.start + 1)
			#  Split unmatched brackets off the end 
		if lastchar in ")]}" and (StringTools.isBracketed(self.value) or StringTools.isLackingOpenBracket(self.value)):
			return self.splitAt(self.end - 1)
			#  Split oxidation state off the end 
		if self._oxidationStateEndPattern.match(self.value):
			return self.splitAt(self.start + self.value.rfind('('))
			#  Split some characters off the front of tokens 
		if firstchar in (StringTools.relations + StringTools.quoteMarks) + (",-/><=." if clm else ""):
			return self.splitAt(self.start + 1)
			#  Split some characters off the back of tokens 
		if lastchar in (".,;:!?\u2122\u00ae" + StringTools.quoteMarks):
			#  Careful with Jones' reagent
			if not re.match(r"^([A-Z][a-z]+s')$",self.value):
				return self.splitAt(self.end - 1)
		m = self._trademarkPattern.match(self.value)
		if m and m.start(1) > 0:
			return self.splitAt(self.start + m.start(1))
		#  characters to split on 
		if "<" in middleValue:
			return self.splitAt(self.start + self.value.find("<"), self.start + self.value.find("<") + 1)
		if ">" in middleValue:
			return self.splitAt(self.start + self.value.find(">"), self.start + self.value.find(">") + 1)
		if clm and ";" in middleValue:
			return self.splitAt(self.start + self.value.find(";"), self.start + self.value.find(";") + 1)	
		if "/" in middleValue:
			return self.splitAt(self.start + self.value.find("/"), self.start + self.value.find("/") + 1)
		if ":" in middleValue:
			#  Check to see if : is nestled in brackets, such as in ring systems
			if StringTools.bracketsAreBalanced(self.value) and StringTools.bracketsAreBalanced(self.value[self.value.find(":") + 1:]):
				return self.splitAt(self.start + self.value.find(":"), self.start + self.value.find(":") + 1)
		if "+" in middleValue:
			index = self.value.find("+")
			if index > 0 and index < len(self.value)-1 :
				if lastchar == "-":
					pass
				elif StringTools.bracketsAreBalanced(self.value) and StringTools.bracketsAreBalanced(self.value[index+1:]):
					return self.splitAt(self.start+index, self.start+index + 1)
				else:
					return self.splitAt(self.start+index, self.start+index + 1);				
		if StringTools.midElipsis in middleValue:
			return self.splitAt(self.start + self.value.find(StringTools.midElipsis), self.start + self.value.find(StringTools.midElipsis) + 1)
		if clm and "=" in middleValue:
			m = self._splitEqualsRe.search(self.value)
			if m is not None:
				return self.splitAt(m.start(1)+self.start, m.start(1)+1+self.start)
		#  Hyphens
		if clm and ",-" in middleValue:
			return self.splitAt(self.start + self.value.find(",-") + 1)
		if "--" in middleValue:
			return self.splitAt(self.start + self.value.find("--"), self.start + self.value.find("--") + 2)
		if clm:
			splittableCommaIndex = HyphenTokeniser._indexOfSplittableComma(self.value)
			if splittableCommaIndex != -1:
				return self.splitAt(self.start + splittableCommaIndex, self.start + splittableCommaIndex + 1)					
		splittableHyphenIndex =  HyphenTokeniser._indexOfSplittableHyphen(self.value, clm)
		#  Split on appropriate hyphens 
		if splittableHyphenIndex != -1 and not re.search(r"[a-z][a-z]",self.value) and re.search(r"[A-Z]",self.value):
			if self._bondPattern.match(self.value):
				splittableHyphenIndex = -1
		if splittableHyphenIndex != -1:
			if self.value.endswith("NMR"):
				return self.splitAt(self.start + splittableHyphenIndex, self.start + splittableHyphenIndex + 1)
			elif self._prefixPattern.match(self.value):
				return self.splitAt(self.start + splittableHyphenIndex + 1)
			else:
				return self.splitAt(self.start + splittableHyphenIndex, self.start + splittableHyphenIndex + 1)
		else:
			return None

	def getNAfter(self, n):
		"""
		Gets the Token n tokens after the current Token.
	  	E.g. getNAfter(1) will get the next token, getNAfter(-1) will
	  	get the previous token, etc.
		
		Args:
			n: n
		
		Returns:
			The nth token after this one.
		"""
		
		pos = n + self.id
		if len(self.tokeniser.getTokenList()) <= pos or pos <= 0:
			return None
		return self.tokeniser.getToken(pos)

