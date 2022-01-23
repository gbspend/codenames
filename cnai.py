import gensim
import numpy as np
import torch
from collections import defaultdict
from itertools import chain, combinations
from random import randrange
from sentence_transformers import util
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

#=HELPERS===================================

def powerset(iterable, rng=range(2,5)):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in rng)

#checks for valid hints: one word only, no acronyms, all alphabetical chars
def isValid(word):
	return '_' not in word and not word.isupper() and word.isalpha()

#list of lists into one list
def flatten(t):
    return [item for sublist in t for item in sublist]

def w2vPreprocess(model, w):
	try:
		model.key_to_index[w]
	except KeyError:
		w = '_'.join([part[0].upper()+part[1:] for part in w.split('_')])
	return w

#TODO: capitalization? :/
#	>>> print(tokenizer.encode("new york", return_tensors='pt'))
#	>>> print(tokenizer.encode("New York", return_tensors='pt'))
#	tensor([[3605,  331,  967]])
#	tensor([[3791, 1971]])
def GPT2Preprocess(w):
	return w.replace("_", " ")

# ｡･:*:･ﾟ★,｡･:*:･ﾟ☆
#lm is GPT2LMHeadModel, embed is embedding numpyarray
def embed2Tok(lm, embed):
	embed = torch.from_numpy(embed)
	head = lm.lm_head
	probs = head(embed)
	return torch.argmax(probs)
# ｡･:*:･ﾟ★,｡･:*:･ﾟ☆

#=ASSOC=====================================

#associates words for a given set of positively and negatively associated words
#abstract superclass, implement for given LM
class Assoc:
	def __init__(self):
		pass
	
	#returns a list of potential associations with a confidence/prob for each
	#abstract function
	def getAssocs(self, pos, neg):
		raise NotImplementedError
	
	#preprocess word before getting embedding (e.g. w2v checks capitalization, gpt converts _ to space)
	def preprocess(self, w):
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
	
	def preprocess(self, w):
		return w2vPreprocess(self.model, w)
	
class GPT2EmbedAssoc(Assoc):
	def __init__(self):
		super().__init__()
		self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
		self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		self.vectors = self.lm.get_input_embeddings().weight.data.numpy() #nparray of ebmedding space
		self.norms = np.linalg.norm(self.vectors, axis=1)
	
	def preprocess(self,w):
		return GPT2Preprocess(self.lm, w)
	
	#gets a normalized embedding vector for word w
	def getNormVec(self,w):
		ids = torch.squeeze(self.tokenizer.encode(w, return_tensors='pt'))
		embed = self.vectors[ids]
		norm = np.linalg.norm(embed)
		return embed/norm
	
	#takes list of pos embeddings and list of neg embeddings and returns topn most similar embeddings
	#	alg from most_similar in RaRe-Technologies/gensim/gensim/models/keyedvectors.py line 703
	def getAssocs(self, pos, neg, topn):
		clip_start = 0
		clip_end = len(self.vectors)

		#if restrict_vocab:
		#	clip_end = restrict_vocab

		pos = flatten([self.getNormVec(w) for w in pos]) # word -> [norm_embeds]; [words] -> [[norm_embeds]] --flatten--> [norm_embeds]
		neg = flatten([self.getNormVec(w) for w in neg])
		
		# add weights for each key; default to 1.0 for positive and -1.0 for negative keys
		positive = [(item, 1.0) for item in pos]
		negative = [(item, -1.0) for item in neg]

		# compute the weighted average of all keys
		all_keys, mean = set(), []
		for key, weight in positive + negative:
			mean.append(weight * key)
			#if self.has_index_for(key): WORKING
			#	all_keys.add(self.get_index(key)) WORKING
		if not mean:
			raise ValueError("cannot compute similarity with no input")
		mean = gensim.matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
		
		return #WORKING

		dists = dot(self.vectors[clip_start:clip_end], mean) / self.norms[clip_start:clip_end]
		if not topn:
			return dists
		best = gensim.matutils.argsort(dists, topn=topn + len(all_keys), reverse=True)
		# ignore (don't return) keys from the input
		result = [
			(self.index_to_key[sim + clip_start], float(dists[sim]))
			for sim in best if (sim + clip_start) not in all_keys
		]
		return result[:topn]


#make sure all the codenames words are in GPT2 (check with Tokenizer)
#class GPT2PromptAssoc(Assoc):

#=GUESSER===================================

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
		hint = self.preprocess(self.curr_hint[0].lower())
		max_v = -9999
		max_w = None
		for ch in choices:
			ch = self.preprocess(ch)
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
	
	#preprocess word before getting embedding (e.g. w2v checks capitalization, gpt converts _ to space)
	def preprocess(self, w):
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
	
	def getSimilarity(self, a, b): 
		return self.model.similarity(a, b)
	
	#return capitalized version of w if w not in model
	#kinda hacky, but w2v has New_York but not new_york etc
	def preprocess(self, w):
		return w2vPreprocess(self.model, w)
		
class GPT2EmbedGuesser(Guesser):
	def __init__(self):
		super().__init__()
		self.model = GPT2Model.from_pretrained("gpt2")
		self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	
	def getSimilarity(self, a, b):
		#tokenizer may result in len > 1, even for single word
		a_ids = self.tokenizer.encode(a, return_tensors='pt')
		b_ids = self.tokenizer.encode(b, return_tensors='pt')
		
		#if embeddings are different lengths, append padding to shorter
		diff = abs(len(a_ids) > len(b_ids))
		if diff:
			pad = torch.full((1,diff), self.tokenizer.eos_token_id)
			#the 1 in "(1,diff)" comes from input_ids
			#	tokenizing "robin, line, plate, band, " results in a torch.Size([1, 10]) tensor: tensor([[ 305, 8800,   11, 1627,   11, 7480,   11, 4097,   11,  220]])
			if len(a_ids) > len(b_ids):
				b_ids = torch.cat([b_ids,pad],1) #dim=1 because, again, shape from tokenizer
			else:
				a_ids = torch.cat([a_ids,pad],1)
			
		a_embeds =  self.model.get_input_embeddings()(a_ids)
		b_embeds =  self.model.get_input_embeddings()(b_ids)
		
		assert len(a_embeds) == len(b_embeds)
		
		a_embeds = np.squeeze(a_embeds.detach().numpy())
		b_embeds = np.squeeze(b_embeds.detach().numpy())
		
		cos_sim = util.cos_sim(a_embeds, b_embeds)
		squozen = np.squeeze(cos_sim.numpy())
		return float(np.mean(squozen))
		
	def preprocess(self, w):
		return GPT2Preprocess(w)
	
#class GPT2PromptGuesser(Guesser):

#=SPYMASTER=================================

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
		neg = [self.assoc.preprocess(w) for w in neg]
		pos = [self.assoc.preprocess(w) for w in pos]
		
		#DEBUG (see bug below):
		if 'a' in pos or 'a' in neg:
			print(blue,board,pos,neg)
			assert False

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

#=TEST======================================

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
	
	pos = board['U']
	neg = board['R'] + board['A']
	
	g = GPT2EmbedAssoc()
	g.getAssocs(pos, neg, 10)
	
	'''
	m = Spymaster(W2VAssoc())
	hint = m.makeHint(board, True)
	gg = GPT2EmbedGuesser()
	
	test = gg.getSimilarity("dog","cat")
	print(test, type(test))
	
	gg.newHint(hint)
	choices = sum(board.values(), [])
	print(gg.nextGuess(choices))
	'''

#

'''
OUTSTANDING BUGS:
File "cngame.py", line 107, in play
    guess = guesser.nextGuess(choices) #string from board
  File "/home/brad/codenames/cnai.py", line 99, in nextGuess
    s = self.model.similarity(hint, ch)
KeyError: "Key 'Csa' not present"

Traceback (most recent call last):
  File "cngame.py", line 171, in <module>
    winner, hist = testCheatVsW2VGPT(1)
  File "cngame.py", line 167, in testCheatVsW2VGPT
    return game.play()
  File "cngame.py", line 101, in play
    self.curr_hint = master.makeHint(board, self.bluesTurn) # returns (word,num) tuple
  File "/home/brad/codenames/cnai.py", line 228, in makeHint
    curr = self.assoc.getAssocs(list(combo),neg)
  File "/home/brad/codenames/cnai.py", line 44, in getAssocs
    return self.model.most_similar(
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 773, in most_similar
    mean.append(weight * self.get_vector(key, norm=True))
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 438, in get_vector
    index = self.get_index(key)
  File "/home/brad/.local/lib/python3.8/site-packages/gensim/models/keyedvectors.py", line 412, in get_index
    raise KeyError(f"Key '{key}' not present")
KeyError: "Key 'a' not present"

Has happened more than once but infrequently: I have a debug if to catch it above...
??? How "list(combo)" and "neg" are drawn from the board! how did "a" (or "Csa") get in there???



FIXED
"TypeError: object of type 'NoneType' has no len()" for len(max_combo) in the return statement of makeHint (for W2V Assoc)
Happened when RED was trying to make a hint for:
	{'U': ['death', 'pole'], 'R': ['saturn'], 'N': ['mole', 'root', 'casino', 'cycle', 'bear'], 'A': ['chest']}
It happens when len(pos) is 1!

'''









