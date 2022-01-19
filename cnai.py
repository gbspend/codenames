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
	
	#hint is (word,num) tuple
	def newHint(self, hint):
		self.curr_hint = hint
		self.num_guesses = 0
	
	#returns one of the words from board as the guess
	#game class will only ask for guesses if the guesser has some left
	#abstract
	def nextGuess(self, board):
		raise NotImplementedError

#answer_key is just list of that colors words
#make sure to pair with Cheatmaster, otherwise the num in the hint might be less than self.n
class CheatGuesser(Guesser):
	def __init__(self, answer_key, n):
		super().__init__()
		self.answers = answer_key
		self.n = n
	
	def nextGuess(self, board):
		if self.num_guesses < self.n:
			self.num_guesses += 1
			return self.answers.pop()
		else:
			return None

class W2VGuesser(Guesser):
	def __init__(self):
		super().__init__()
	
	def nextGuess(self, board):
		raise NotImplementedError

#class GPT2PromptGuesser(Guesser):

#class GPT2EmbedGuesser(Guesser):

#===========================================

class Spymaster:
	def __init__(self, assoc):
		self.assoc = assoc #subclass of Assoc class
	
	def makeHint(self, board):
		raise NotImplementedError
		#TODO copy spymast() from cn.py to start

class Cheatmaster:
	def __init__(self):
		super().__init__(None) #doesn't need Assoc
	
	def makeHint(self, board):
		return ("CHEAT", 9999) #this is so the cheat guesser can (perfectly) guess as many times as it needs

#===========================================

if __name__ == "__main__":
	m = Spymaster(W2VAssoc())