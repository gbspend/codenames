import gensim
from collections import defaultdict
from itertools import chain, combinations
from random import randrange

#checks for valid hints: one word only, no acronyms, all alphabetical chars
def isValid(word):
	return '_' not in word and not word.isupper() and word.isalpha()

#associates words for a given set of positively and negatively associated words
#abstract superclass, implement for given LM
class Assoc:
	def __init__(self):
		pass
	
	#returns a list of potential associations with a confidence/prob for each
	#abstract function
	def getAssocs(self, pos, neg):
		raise NotImplementedError

class W2VAssoc(Assoc):
	def __init__(self):
		super().__init__()
		self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	
	def getAssocs(self, pos, neg):
		return self.model.most_similar(
			positive=pos,
			negative=neg,
			restrict_vocab=50000
		)

#make sure all the codenames words are in GPT2 (check with Tokenizer)
#class GPT2PromptAssoc(Assoc):

#class GPT2EmbedAssoc(Assoc):

#===========================================

#NOTE: guess may be None for "pass"
#Maybe rename/refactor to SimilarityGuesser?
class Guesser:
	def __init__(self):
		self.curr_hint = None
		self.num_guesses = 0 #increment with each guess, should never get higher than the num in hint[1]
	
	def isCheat(self):
		return False
	
	#hint is (word,num) tuple
	def newHint(self, hint):
		self.curr_hint = hint
		self.num_guesses = 0
	
	#returns one of the words from choices as the guess (not board, just list of possible words)
	#game class will only ask for guesses if the guesser has some left
	def nextGuess(self, choices):
		hint = fixCap(self.model, self.curr_hint[0].lower())
		max_v = -9999
		max_w = None
		for ch in choices:
			ch = fixCap(self.model, ch)
			s = self.getSimilarity(hint, ch)
			if s > max_v:
				max_v = s
				max_w = ch
		self.num_guesses += 1
		return max_w.lower()
	
	#return the similarity between 2 words
	#ABSTRACT
	def getSimilarity(self, a, b):
		raise NotImplementedError

#make sure to pair with Cheatmaster, otherwise the num in the hint might be less than self.n
class CheatGuesser(Guesser):
	def __init__(self, n):
		super().__init__()
		self.answers = None
		self.n = n
	
	def isCheat(self):
		return True
	
	#call this before every guess because board changes
	def cheat(self, board, isBlue):
		self.answers = board['U'] if isBlue else board['R']
	
	def nextGuess(self, choices):
		if self.answers is None:
			raise ValueError("CheatGuesser was never given answers via cheat()")
		if self.num_guesses < self.n:
			self.num_guesses += 1
			return self.answers.pop()
		else:
			return None

#return capitalized version of w if w not in model
#kinda hacky, but w2v has New_York but not new_york etc
def fixCap(model, w):
	try:
		model.key_to_index[w]
	except KeyError:
		w = '_'.join([part[0].upper()+part[1:] for part in w.split('_')])
	return w

class W2VGuesser(Guesser):
	def __init__(self):
		super().__init__()
		self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	
	'''
	def nextGuess(self, choices):
		hint = fixCap(self.model, self.curr_hint[0].lower())
		max_v = -9999
		max_w = None
		for ch in choices:
			ch = fixCap(self.model, ch)
			s = self.model.similarity(hint, ch)
			if s > max_v:
				max_v = s
				max_w = ch
		self.num_guesses += 1
		return max_w.lower()
	'''
	
	def getSimilarity(self, a, b): 
		return self.model.similarity(a, b)
	
		
#class GPT2PromptGuesser(Guesser):

#class GPT2EmbedGuesser(Guesser):

#===========================================

def powerset(iterable, rng=range(2,5)):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in rng)

#should be model-agnostic
class Spymaster:
	def __init__(self, assoc):
		self.assoc = assoc #subclass of Assoc class
	
	class Combo:
		def __init__(self):
			self.scores = [] #all similarity scores gen'd for this combo, regardless of hint
			#also track hint with max sim score
			self.max_hint = None
			self.max_sim = -9999
		
		def addOption(self, hint, sim):
			self.scores.append(sim)
			if self.max_sim < sim:
				self.max_sim = sim
				self.max_hint = hint
		
		def getAvgSim(self):
			return sum(self.scores)/len(self.scores)
	
	#returns (hint, number) tuple
	#IDEA: if there are only 3-4 words left, lean more toward hail marys
	def makeHint(self, board, blue):
		neg = board['N'] + board['A'] + (board['R'] if blue else board['U'])
		pos = board['U'] if blue else board['R']
		
		#hacky, but w2v is picky about capitals
		neg = [fixCap(self.assoc.model, w) for w in neg]
		pos = [fixCap(self.assoc.model, w) for w in pos]

		#Game AI approach:
		#1. find combo with highest avg hint similarity (hyp: most likely to be closest related combo)
		#2. pick the highest-scoring hint for that combo as our hint (# is just len of combo ofc)
		
		combos = defaultdict(Spymaster.Combo)
		
		if len(pos) == 1: #powerset 2-4 will return []!
			pow_set = pos
		else:
			pow_set = powerset(pos)
		for combo in pow_set:
			curr = self.assoc.getAssocs(list(combo),neg)
			for hint,sim in curr:
				if isValid(hint):
					combos[combo].addOption(hint, sim)
		
		max_avg_sim = -9999
		max_combo = None
		
		 # bc I got "TypeError: object of type 'NoneType' has no len()" for len(max_combo) below???
		if not combos.keys():
			print(board,blue,pos,neg)
			assert False
		
		for combo in combos.keys():
			avg_sim = combos[combo].getAvgSim()
			if max_avg_sim < avg_sim:
				max_avg_sim = avg_sim
				max_combo = combo
		
		#print(max_combo) #DEBUG
		return (combos[max_combo].max_hint, len(max_combo))
		

class Cheatmaster(Spymaster):
	def __init__(self):
		super().__init__(None) #doesn't need Assoc
	
	def makeHint(self, board, blue):
		return ("CHEAT", 9999) #this is so the cheat guesser can (perfectly) guess as many times as it needs

#===========================================

if __name__ == "__main__":
	board = {
		'U': [
			'ambulance', 'hospital', 'spell', 'lock', 
			'charge', 'tail', 'link', 'cook', 'web'
		],
		'R': [
			'cat', 'button', 'pipe', 'pants', 
			'mount', 'sleep', 'stick', 'file'
		],
		'N': ["giant", "nail", "dragon", "stadium", "flute", "carrot", "wake"],
		'A': ['doctor']
	}
	m = Spymaster(W2VAssoc())
	print(m.makeHint(board, True))
#

'''
OUTSTANDING BUGS:
File "cngame.py", line 107, in play
    guess = guesser.nextGuess(choices) #string from board
  File "/home/brad/codenames/cnai.py", line 99, in nextGuess
    s = self.model.similarity(hint, ch)
KeyError: "Key 'Csa' not present"

Traceback (most recent call last):
  File "cngame.py", line 166, in <module>
    winner, hist = testCheatVsW2V(1)
  File "cngame.py", line 162, in testCheatVsW2V
    return game.play()
  File "cngame.py", line 101, in play
    self.curr_hint = master.makeHint(board, self.bluesTurn) # returns (word,num) tuple
  File "/home/brad/codenames/cnai.py", line 162, in makeHint
    curr = self.assoc.getAssocs(list(combo),neg)
  File "/home/brad/codenames/cnai.py", line 27, in getAssocs
    return self.model.most_similar(
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 773, in most_similar
    mean.append(weight * self.get_vector(key, norm=True))
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 438, in get_vector
    index = self.get_index(key)
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 412, in get_index
    raise KeyError(f"Key '{key}' not present")
KeyError: "Key 'a' not present"
??? How "list(combo)" and "neg" are drawn from the board! how did "a" (or "Csa") get in there???


FIXED
"TypeError: object of type 'NoneType' has no len()" for len(max_combo) in the return statement of makeHint (for W2V Assoc)
Happened when RED was trying to make a hint for:
	{'U': ['death', 'pole'], 'R': ['saturn'], 'N': ['mole', 'root', 'casino', 'cycle', 'bear'], 'A': ['chest']}
It happens when len(pos) is 1!

'''









