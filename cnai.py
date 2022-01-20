import gensim
from random import randrange

#associates words for a given set of positively and negatively associated words
#abstract superclass, implement for given LM
class Assoc:
	def __init__(self):
		pass
	
	#returns a list of potential associations with a confidence/prob for each
	#abstract function
	def getAssoc(self, pos, neg):
		raise NotImplementedError

class W2VAssoc(Assoc):
	def __init__(self):
		super().__init__()
		self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	
	def getAssoc(self, pos, neg):
		return model.most_similar(
			positive=pos,
			negative=neg,
			restrict_vocab=50000
		)

#make sure all the codenames words are in GPT2 (check with Tokenizer)
#class GPT2PromptAssoc(Assoc):

#class GPT2EmbedAssoc(Assoc):

#===========================================

#NOTE: guess may be None for "pass"
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
	#abstract
	def nextGuess(self, choices):
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

class W2VGuesser(Guesser):
	def __init__(self):
		super().__init__()
		self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	
	#kinda hacky, but w2v has New_York but not new_york etc
	def fixCap(self, w):
		try:
			self.model.key_to_index[w]
		except KeyError:
			w = '_'.join([part[0].upper()+part[1:] for part in w.split('_')])
		return w
	
	def nextGuess(self, choices):
		hint = self.fixCap(self.curr_hint[0].lower())
		max_v = -9999
		max_w = None
		for ch in choices:
			ch = self.fixCap(ch)
			s = self.model.similarity(hint, ch)
			if s > max_v:
				max_v = s
				max_w = ch
		self.num_guesses += 1
		return max_w.lower()
		
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
	
	def makeHint(self, board):
		neg = board['N'] + board['A'] + board['R'] if blue else board['U']
		pos = board['U'] if blue else board['R']

		options = []
		#try all combos to find the one we're most sure of
		#IDEA: if there are only 3-4 words left, lean more toward hail marys
		for combo in powerset(pos):
			self.assoc.getAssoc(pos,neg)
			#append combo so we know # (and for debug)
			#also filter out _ (phrases) and hints containing one of the words in the combo (against the rules)
			curr = [(*hint,combo) for hint in curr if '_' not in hint[0] and not containsAny(hint[0], combo)]
			options += curr[:5] #try just top 5 (after filtering)

		options.sort(key=lambda x: x[1], reverse=True)

class Cheatmaster(Spymaster):
	def __init__(self):
		super().__init__(None) #doesn't need Assoc
	
	def makeHint(self, board):
		return ("CHEAT", 9999) #this is so the cheat guesser can (perfectly) guess as many times as it needs

#===========================================

if __name__ == "__main__":
	m = Spymaster(W2VAssoc())
#
















