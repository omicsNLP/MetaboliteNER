"""
Functions for string manipulation. Used by ChemTokeniser and associated classes. May well be useful elsewhere.
"""  

#  Lowercase Greek Unicode characters   
lowerGreek = "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c1\u03c2\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9"

#  Quotation marks of various Unicode forms 
quoteMarks = "\"'\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F"

#  Hyphens, dashes and the like 
hyphens = "-\u2010\u2011\u2012\u2013\u2014\u2015"

#  A regex to pick up a hyphen/dash/etc. 
hyphensRe = "(?:-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015)"

#  Apostrophes, backticks, primess etc 
primes = "'`\u2032\u2033\u2034"

#print(primes)

#  An en dash 
enDash = "\u2013"

#  An em dash 
emDash = "\u2014"

#  Three dots, at middle-level (eg representing a hydrogen bond 
midElipsis = "\u22ef"

#  Less than, greater than, equals, and complicated things in Unicode 
relations = "=<>\u2260\u2261\u2262\u2263\u2264\u2265\u2266\u2267\u2268" + "\u2269\u226a\u226b"



def isLackingOpenBracket(s):
	""" 
	Would adding an open bracket to the start of the string make it balanced?
	E.g. "example)" would return true, whereas "example", "(example)" and "(example"
	would return false.

	Args:
		s: The string to test.

	Returns:
		Whether an open bracket would balance the string.
	"""
	bracketLevel = 0
	for c in s:
	  if c in '([{':
		  bracketLevel += 1
	  elif c in ')]}':
		  bracketLevel -= 1
	  if bracketLevel == -1:
		  return True
	return False


def isLackingCloseBracket(s):
	"""
	Would adding a close bracket to the end of the string make it balanced?
	E.g. "(example" would return true, whereas "example", "(example)" and "example)"
	would return false.
	 
	Args:
		s: The string to test.
	
	Returns:
		Whether a close bracket would balance the string.
	"""
	bracketLevel = 0
	for c in s[::-1]: #reversing
		if c in '([{':
			bracketLevel -= 1
		elif c in ')]}':
			bracketLevel += 1
		if bracketLevel == -1:
			return True
	return False


def bracketsAreBalanced(s):
	"""
	Whether the string has matching brackets.
	eg.	
	"foo" gives true
	"(foo" gives false
	"(foo)" gives true
	"foo()bar" gives true
	"(foo)bar" gives true
	"foo)(bar" gives false
	
	Args:
		s: The string to test.

	Returns:
		Whether it has matching brackets.
	"""
	bracketLevel = 0
	#print("----->",s)
	for c in s:
		if c in '([{':
			bracketLevel += 1
		elif c in ')]}':
			bracketLevel -= 1
			
		if bracketLevel == -1:
			return False
	
	if bracketLevel == 0:
		return True
	return False


def isBracketed(s):
	"""
	Whether a string has matching, balanced brackets around the outside.

	Args:
		s: The string to test.

	Returns:
		Whether it has matching, balanced brackets.
	"""
	if s == None or len(s) < 3:
		return False
	first = s[0]
	last  = s[-1]
	if not ((first == '(' and last == ')') or (first == '[' and last == ']') or (first == '{' and last == '}')):
		return False
	if not bracketsAreBalanced(s[1:-1]):
		return False
	return True
