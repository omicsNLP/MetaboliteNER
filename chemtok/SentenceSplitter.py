"""
Very simple sentence splitting for use with ChemTokeniser.
"""

import re

_splitTokens = set((".", "?", "!", "\""))
_impossibleBefore = set()
_impossibleAfter = set(("Fig", "al", "i.e", "ie", "eg", "e.g", "ref", "Dr", "Prof", "Sir"))
_lowerRe = re.compile("^[a-z]+$")

def sentenceString(sentence):
	"""
	Get a string for a sentence given a list of tokens.
	
	Args:
		sentence: list of tokens
		
	Returns:
		string for sentence
	"""
	if len(sentence) == 0: return ""
	startTok = sentence[0]
	endTok = sentence[-1]
	tokr = startTok.tokeniser
	offset = tokr.offset
	return tokr.sourceString[startTok.start-offset:endTok.end-offset]
	
def makeSentences(tokens):
	"""
	Take a list of Tokens and convert to a list of lists of Tokens corresponding to sentences.
	
	Args:
		tokens: list of Tokens
		
	Returns:
		list of list of tokens, one list per sentence
	"""
	sentences = []
	sentence = []
	sentences.append(sentence)
	prevSentence = None
	for t in tokens:
		sentence.append(t)
		split = False
		value = t.value
		if value in _splitTokens:
			split = True
			next = t.getNAfter(1);
			prev = t.getNAfter(-1)
			if next != None and prev != None:
				nextStr = next.value
				prevStr = prev.value
				if value == "\"" and prevStr not in _splitTokens:
					split = False
				elif prevStr in _impossibleAfter:
					split = False
				elif nextStr in _impossibleBefore:
					split = False
				elif nextStr in _splitTokens:
					split = False
				elif _lowerRe.match(nextStr):
					split = False
		if split:
			prevSentence = sentence
			sentence = []
			sentences.append(sentence)
	return sentences