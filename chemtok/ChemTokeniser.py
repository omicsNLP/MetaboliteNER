from .Token import Token
import re
  
class ChemTokeniser:
	"""Adaptation of ChemTok tokeniser from Oscar3 (also originally by Peter Corbett). Python translation originally by Adam Bernard.
	
	Original version at: https://sourceforge.net/projects/oscar3-chem/
	Based on a tokeniser described in:
	
	Annotation of Chemical Named Entities
	Peter Corbett, Colin Batchelor, Simone Teufel
	Proceedings of BioNLP 2007
	
	Tokenisation, especially for chemistry and related text.
	
	"""

	_tokenRe = re.compile("\\S+")

	def __init__(self,s, offset=0, splitOffsets=None, aggressive=False, charbychar=False, clm=False):
		"""
		Creates a Tokeniser, containing the tokenisation of a string, with a specified
	 	offset, and ensuring tokenisation at various positions. For example, you may
	 	wish to tokenise a string that has been taken from an XML document, and
		analysis of the XML shows that certain substrings are (for example) citation
	 	references, which should be treated as separate tokens. By including the start
	 	and end offsets of all of these in the splitOffsets parameter, you can ensure
	 	that these substrings are turned into separate tokens.

		The clm option is used by chemlistem - it seems to give better results on
		patent text, which is less well copy-edited than the papers the Oscar3 tokeniser
		was originally developed for.
		
		Args:
			s: The string to tokenise
			offset: The offset of the start of the string in the document it was taken from.
			splitOffsets: The offsets to split at.
			aggressive: If True, use a different tokenisation algorithm, finding much smaller tokens.
			charbychar: If True, find single character "tokens", including whitespace.
			clm: If True, use the modifications for chemlistem.
		"""
		self.tokens = list()
		self.sourceString = ""
		if charbychar:
			for i in range(len(s)):
				self.tokens.append(Token(s[i:i+1], i, i+1, self))
			self.numberTokens()
		elif aggressive:
			for m in re.finditer("([A-Z]*[a-z]+|[A-Z]+|[0-9]+|\\S)", s):
				self.tokens.append(Token(m.group(0), m.start(), m.end(), self))
			self.numberTokens()
		else:
			self._tokenise(s, offset, splitOffsets, clm)

 
	def _tokenise(self, s, offset, splitOffsets, clm):
		self.tokens = list()
		self.sourceString = s
		self.offset = offset
		matches = self._tokenRe.finditer(s)
		id = 0
		for m in matches:
			id += 1
			self.tokens.append(Token(m.group(0), m.start(), m.end(), self))
		if splitOffsets != None:
			self._splitAtOffsets(splitOffsets)
		self._splitTokens(clm)
		self._discardEmptyTokens()
		self.numberTokens()

	def _splitAtOffsets(self, splitOffsets):
		offsets = list(splitOffsets).sorted()
		tokenNo = 0
		offsetNo = 0
		splitOffset = -1
		while tokenNo < len(self.tokens) and offsetNo < len(offsets):
			t=self.tokens[tokenNo]
			splitOffset = offsets.get(offsetNo)
			if t.getEndOffset() <= splitOffset:
				tokenNo += 1
			elif t.getStartOffset() >= splitOffset:
				offsetNo += 1
			else:
				self.tokens.remove(tokenNo)
				self.tokens.addAll(tokenNo, t._splitAt(splitOffset))

	def _splitTokens(self, clm):		
		i = 0
		while i < len(self.tokens):
			results = self.tokens[i]._split(clm)
			#  Returns null if no splitting occurs 
			if results == None:
				i += 1
			else:
				self.tokens.pop(i)
				results.reverse()
				for onetok in results:
				  self.tokens.insert(i, onetok)
				#  Note: NO i++ here. This allows the resulting tokens
				#  to be recursively subtokenised without recursion. Neat huh? 

	def _discardEmptyTokens(self):		
		tmpTokens = list()
		for t in self.tokens:
			if t.value not in  (None, ""):
				tmpTokens.append(t)
		self.tokens = tmpTokens

	def numberTokens(self):
		"""
		Assigns an id attribute to each Token.
		"""
		id = 0
		for t in self.tokens:
			t.id = id
			id += 1

	def getToken(self, num):
		"""
		Gets a Token from the list of tokens produced.
		  
		Args:
			num: The position of Token to get (0 = first, 1 = second etc.).
		
		Returns:
			The Token at that position.
		"""
		return self.tokens[num]

	def getWhitespace(self, num):
		"""
		Gets the whitespace to the left of on the token at the specified position.
		This means that getWhitespace(0) will return the whitespace at the
		very start of the source string.

		Args:
			num: The position of Token to get the whitespace to the left of (0 = first, 1 = second etc.).

		Returns:
			The whitespace.		
		"""
		if len(self.tokens) == 0:
			return ""
		startpos = int()
		end = int()
		if num == 0:
			startpos = self.offset
			end = self.tokens.get(num).getStartOffset()
		elif num == len(self.tokens):
			startpos = self.tokens.get(num - 1).getEndOffset()
			end = self.offset + len(self.sourceString)
		else:
			startpos = self.tokens.get(num - 1).getEndOffset()
			end = self.tokens.get(num).getStartOffset()
		return self.getSubString(startpos, end)

	def getTokenList(self):
		"""
		Gets the list of Tokens found.
		
		Returns:
			The list of Tokens.
		"""
		return self.tokens

	def getTokenStringList(self):
		"""
		Gets the list of Tokens found, as a list of strings.

		Returns:
			The list of strings.
		"""
		return [t.value for t in self.tokens]




		 


