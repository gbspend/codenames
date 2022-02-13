import gensim
import numpy as np
from re import sub
import torch
from collections import defaultdict
from itertools import chain, combinations
from nltk.corpus import words
from random import randrange
from sentence_transformers import util
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, pipeline

#=HELPERS===================================

all_words = set(words.words())

#-Model Singletons-------------------
w2v_model = None
def getW2vModel():
	global w2v_model
	if not w2v_model:
		w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	return w2v_model

gpt_model = None
def getGptModel():
	global gpt_model
	if not gpt_model:
		gpt_model = GPT2Model.from_pretrained("gpt2")
	return gpt_model

gpt_lm = None
def getGptLM():
	global gpt_lm
	if not gpt_lm:
		gpt_lm = GPT2LMHeadModel.from_pretrained("gpt2")
	return gpt_lm

gpt_tok = None
def getGptTok():
	global gpt_tok
	if not gpt_tok:
		gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
	return gpt_tok

gen_pipe = None
def getGenPipe():
	global gen_pipe
	if not gen_pipe:
		gen_pipe = pipeline("text-generation")
		gen_pipe.model.config.max_length=100
	return gen_pipe
#------------------------------------

#converts (parts,prob) into (longest_str,mean_prob)
#	parts are the vocab equiv of a token (likely not a whole word), e.g. 'amb' for 4131
#	assumes words appear sequentially (i.e. not trying all combos @_@)
def dists2words(dists):
	parts,probs = zip(*dists)
	parts = [p.strip() for p in parts]
	#print(parts)
	ret = []
	i = 0
	while i < len(parts):
		j = i+1
		longest = None
		mean_prob = -1 #avg prob of all parts in the longest word
		while j < len(parts):
			curr = ''.join(parts[i:j])
			#print(curr,curr in all_words)
			if curr in  all_words:
				longest = curr #just keep longest
				mean_prob = sum(probs[i:j])/(j-i)
			j+=1
		if longest is not None:
			ret.append((longest, mean_prob))
		i+=1
	return ret

def powerset(iterable, rng=range(2,5)):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in rng)

#checks for valid hints:
#	not on board, one word only, no acronyms, all alphabetical chars
def isValid(word, board_words):
	if word in board_words:
		return False
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
def GPT2Preprocess(model, w):
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
	def getAssocs(self, pos, neg, topn):
		raise NotImplementedError
	
	#preprocess word before getting embedding (e.g. w2v checks capitalization, gpt converts _ to space)
	def preprocess(self, w):
		raise NotImplementedError

class W2VAssoc(Assoc):
	def __init__(self):
		super().__init__()
		self.model = getW2vModel()
	
	def getAssocs(self, pos, neg, topn):
		return self.model.most_similar(
			positive=pos,
			negative=neg,
			topn=topn,
			restrict_vocab=50000
		)
	
	def preprocess(self, w):
		return w2vPreprocess(self.model, w)
	
class GPT2EmbedAssoc(Assoc):
	def __init__(self):
		super().__init__()
		self.lm = getGptLM() #GPT2LMHeadModel.from_pretrained("gpt2")
		self.tokenizer = getGptTok() # GPT2Tokenizer.from_pretrained("gpt2")
		self.vectors = self.lm.get_input_embeddings().weight.data.numpy() #nparray of ebmedding space
		self.norms = np.linalg.norm(self.vectors, axis=1)
	
	def preprocess(self,w):
		return GPT2Preprocess(self.lm, w)
	
	#gets a normalized embedding vector for word w
	def getNormVec(self,w):
		ids = self.tokenizer.encode(w, return_tensors='pt')[0].tolist()
		ret = []
		for i in ids: #do each individually because one word may result in more than one id
			embed = self.vectors[i]
			norm = np.linalg.norm(embed)
			ret.append((embed/norm,i))
		return ret
	
	#takes list of pos words and list of neg words and returns topn most similar words
	#	alg from most_similar in RaRe-Technologies/gensim/gensim/models/keyedvectors.py line 703
	def getAssocs(self, pos, neg, topn):
		clip_end = len(self.vectors)

		#if restrict_vocab:
		#	clip_end = restrict_vocab

		# word -> [norm_embeds]; [words] -> [[norm_embeds]] --flatten--> [norm_embeds]
		pos = flatten([self.getNormVec(w) for w in pos])
		neg = flatten([self.getNormVec(w) for w in neg])
		
		# add weights for each key; default to 1.0 for positive and -1.0 for negative keys
		positive = [(*item, 1.0) for item in pos]
		negative = [(*item, -1.0) for item in neg]

		# compute the weighted average of all keys
		all_keys, mean = set(), []
		for key, tok, weight in positive + negative:
			mean.append(weight * key)
			#if self.has_index_for(key):
			all_keys.add(tok)
		if not mean:
			raise ValueError("cannot compute similarity with no input")
		mean = gensim.matutils.unitvec(np.array(mean, dtype=object).mean(axis=0)).astype(np.float32)
		
		# ｡･:*:･ﾟ★,｡･:*:･ﾟ☆
		dists = np.dot(self.vectors[:clip_end], mean) / self.norms[:clip_end]
		# ｡･:*:･ﾟ★,｡･:*:･ﾟ☆
		
		#if not topn:
		#	return dists
		
		# times 5 because: a) may include board words, and b) tok < word
		best = gensim.matutils.argsort(dists, reverse=True)[:topn*5]
	
		'''
		# I don't like this because it could eliminate a *tok* that's
		#		in the input but that could be used to make a different word
		#		hinter can just filter out board words post hoc
		# ignore (don't return) keys from the input
		parts_with_probs = [
			(self.tokenizer.decode(sim.item()), float(dists[sim.item()]))
			for sim in best if sim not in all_keys
		]
		'''
		
		parts_with_probs = [
			(self.tokenizer.decode(sim.item()), float(dists[sim.item()]))
			for sim in best
		]
		
		final_words = dists2words(parts_with_probs)
		return final_words

default_prompt = '''These words are related to ambulance: paramedic, emergency, doctor.
These words are related to boat: water, fish, captain.
These words are related to PROMPT: '''

class GPT2PromptAssoc(Assoc):
	def __init__(self, prompt=None):
		super().__init__()
		self.pipe = getGenPipe()
		if not prompt:
			self.base_prompt = default_prompt
		else:
			self.base_prompt = prompt
	
	def preprocess(self,w):
		return GPT2Preprocess(self.pipe.model, w)
		
	#TODO:
	def testAssoc(self,prompt):
		raw = self.pipe(self.prompt)[0]['generated_text']
		output = raw[len(self.prompt):]
		newi = output.find('\n')
		if newi > 0:
			output = output[:newi]
		parts = [s.strip() for s in output.split(",")]
		parts = [sub(r'[^\w\s]', '', p) for p in parts if p]
		return parts
	
	#takes list of pos words and list of neg words and returns topn most similar words
	def getAssocs(self, pos, neg, topn):
		#TODO:
		ret = set()
		for w in pos:
			prompt = self.base_prompt.replace("PROMPT",w)
			ret.update(self.testAssoc(prompt))
		#Verified that w2v similarity probs don't sum to 1
		return [(w,0.5) for w in ret] #Be careful with these probs! They will likely overshadow real words!

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
		self.model = getW2vModel()
	
	def getSimilarity(self, a, b): 
		return self.model.similarity(a, b)
	
	#return capitalized version of w if w not in model
	#kinda hacky, but w2v has New_York but not new_york etc
	def preprocess(self, w):
		return w2vPreprocess(self.model, w)
		
class GPT2EmbedGuesser(Guesser):
	def __init__(self):
		super().__init__()
		self.model = getGptModel() #GPT2Model.from_pretrained("gpt2")
		self.tokenizer = getGptTok() #GPT2Tokenizer.from_pretrained("gpt2")
	
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
		return GPT2Preprocess(self.model, w)
	
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
		board_words = set([item for sublist in list(board.values()) for item in sublist])
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
			curr = self.assoc.getAssocs(list(combo),neg, 10)
			
			any_added = False #DEBUG
			for hint,sim in curr:
				if isValid(hint, board_words):
					combos[combo].addOption(hint, sim)
					any_added = True
			if not any_added:
				print("NONE ADDED:", [hint for hint,sim in curr])
		
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
	
	'''
	g = GPT2EmbedAssoc()
	assocs = g.getAssocs(pos, neg, 10)
	for word,prob in assocs:
		print(word, "%.4f" % prob)
	'''
	
	m = Spymaster(GPT2EmbedAssoc())
	hint = m.makeHint(board, True)
	
	print(hint)
	
	gg = GPT2EmbedGuesser()
	gg.newHint(hint)
	choices = sum(board.values(), [])
	print(gg.nextGuess(choices))

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









