import gensim
import pprint
from collections import defaultdict
from itertools import chain, combinations
from random import sample, shuffle

pp = pprint.PrettyPrinter(indent=4)

words = [
	"hollywood", "well", "foot", "new_york", "spring", "court", "tube", "point", "tablet", "slip", "date", "drill", "lemon", "bell", "screen",
	"fair", "torch", "state", "match", "iron", "block", "france", "australia", "limousine", "stream", "glove", "nurse", "leprechaun", "play",
	"tooth", "arm", "bermuda", "diamond", "whale", "comic", "mammoth", "green", "pass", "missile", "paste", "drop", "pheonix", "marble", "staff",
	"figure", "park", "centaur", "shadow", "fish", "cotton", "egypt", "theater", "scale", "fall", "track", "force", "dinosaur", "bill", "mine",
	"turkey", "march", "contract", "bridge", "robin", "line", "plate", "band", "fire", "bank", "boom", "cat", "shot", "suit", "chocolate",
	"roulette", "mercury", "moon", "net", "lawyer", "satellite", "angel", "spider", "germany", "fork", "pitch", "king", "crane", "trip", "dog",
	"conductor", "part", "bugle", "witch", "ketchup", "press", "spine", "worm", "alps", "bond", "pan", "beijing", "racket", "cross", "seal",
	"aztec", "maple", "parachute", "hotel", "berry", "soldier", "ray", "post", "greece", "square", "mass", "bat", "wave", "car", "smuggler",
	"england", "crash", "tail", "card", "horn", "capital", "fence", "deck", "buffalo", "microscope", "jet", "duck", "ring", "train", "field",
	"gold", "tick", "check", "queen", "strike", "kangaroo", "spike", "scientist", "engine", "shakespeare", "wind", "kid", "embassy", "robot",
	"note", "ground", "draft", "ham", "war", "mouse", "center", "chick", "china", "bolt", "spot", "piano", "pupil", "plot", "lion", "police",
	"head", "litter", "concert", "mug", "vacuum", "atlantis", "straw", "switch", "skyscraper", "laser", "scuba_diver", "africa", "plastic",
	"dwarf", "lap", "life", "honey", "horseshoe", "unicorn", "spy", "pants", "wall", "paper", "sound", "ice", "tag", "web", "fan", "orange",
	"temple", "canada", "scorpion", "undertaker", "mail", "europe", "soul", "apple", "pole", "tap", "mouth", "ambulance", "dress", "ice_cream",
	"rabbit", "buck", "agent", "sock", "nut", "boot", "ghost", "oil", "superhero", "code", "kiwi", "hospital", "saturn", "film", "button",
	"snowman", "helicopter", "loch_ness", "log", "princess", "time", "cook", "revolution", "shoe", "mole", "spell", "grass", "washer", "game",
	"beat", "hole", "horse", "pirate", "link", "dance", "fly", "pit", "server", "school", "lock", "brush", "pool", "star", "jam", "organ",
	"berlin", "face", "luck", "amazon", "cast", "gas", "club", "sink", "water", "chair", "shark", "jupiter", "copper", "jack", "platypus",
	"stick", "olive", "grace", "bear", "glass", "row", "pistol", "london", "rock", "van", "vet", "beach", "charge", "port", "disease", "palm",
	"moscow", "pin", "washington", "pyramid", "opera", "casino", "pilot", "string", "night", "chest", "yard", "teacher", "pumpkin", "thief",
	"bark", "bug", "mint", "cycle", "telescope", "calf", "air", "box", "mount", "thumb", "antarctica", "trunk", "snow", "penguin", "root", "bar",
	"file", "hawk", "battery", "compound", "slug", "octopus", "whip", "america", "ivory", "pound", "sub", "cliff", "lab", "eagle", "genius",
	"ship", "dice", "hood", "heart", "novel", "pipe", "himalayas", "crown", "round", "india", "needle", "shop", "watch", "lead", "tie", "table",
	"cell", "cover", "czech", "back", "bomb", "ruler", "forest", "bottle", "space", "hook", "doctor", "ball", "bow", "degree", "rome", "plane",
	"giant", "nail", "dragon", "stadium", "flute", "carrot", "wake", "fighter", "model", "tokyo", "eye", "mexico", "hand", "swing", "key",
	"alien", "tower", "poison", "cricket", "cold", "knife", "church", "board", "cloak", "ninja", "olympus", "belt", "light", "death", "stock",
	"millionaire", "day", "knight", "pie", "bed", "circle", "rose", "change", "cap", "triangle"
]

def powerset(iterable):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(2,5))

#TODO? the preset board layout cards? ...no?
def makeBoard(blueFirst=True):	
	#I don't know how many red/blue for anything other than 5x5
	
	spaces = ['U']*8 + ['R']*8 + ['N']*7 + ['A']
	spaces += ['U'] if blueFirst else ['R']
	shuffle(spaces)
	
	selected = sample(words, 25)
	
	assert len(spaces) == len(selected) == 25
	
	board = {'U':[], 'R':[], 'N':[], 'A':[]}
	
	for i in range(25):
		color = spaces[i]
		word = selected[i]
		board[color].append(word)
	
	return board,spaces,selected

#=============================================================

def containsAny(source, targets):
	for t in targets:
		if t in source:
			return True
	return False

def spymast(model,board,blue=True):
	pp.pprint(board)
	
	neg = board['N'] + board['A'] + board['R'] if blue else board['U']
	pos = board['U'] if blue else board['R']
	
	options = []
	#try all combos to find the one we're most sure of
	#IDEA: if there are only 3-4 words left, lean more toward hail marys
	for combo in powerset(pos):
		curr = model.most_similar(
			positive=list(combo),
			negative=neg,
			restrict_vocab=50000
		)
		#append combo so we know # (and for debug)
		#also filter out _ (phrases) and hints containing one of the words in the combo (against the rules)
		curr = [(*hint,combo) for hint in curr if '_' not in hint[0] and not containsAny(hint[0], combo)]
		options += curr[:5] #try just top 5
	
	options.sort(key=lambda x: x[1], reverse=True)
	
	#collapse duplicate hints into one (preserve prob + target words)
	coll = defaultdict(list)
	order = defaultdict(float)
	for t in options:
		hint,prob,target = t
		coll[hint].append((prob,target))
		order[hint] = max(prob, order[hint])
	
	top_hints = sorted(list(order.keys()), key=lambda k: order[k], reverse=True)
	for hint in top_hints[:10]:
		print(hint)
		for combo in coll[hint]:
			print("\t",combo[1])
	return #
	
	temp = [(hint[0], "{:.4f}".format(hint[1]), hint[2]) for hint in options] #truncate floats for readability
	pp.pprint(temp)
	#TODO: choose one, return

def findBads(model):
	bads = []
	for w in words:
		try:
			model.key_to_index[w]
		except KeyError:
			bads.append(w)
	return bads
			
if __name__ == '__main__':
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
	
	#correct bad captializations (e.g., w2v has Saturn but not saturn)
	for i in range(len(words)):
		w = words[i]
		try:
			model.key_to_index[w]
		except KeyError:
			cap = '_'.join([part[0].upper()+part[1:] for part in w.split('_')])
			print(w,cap)
			try:
				model.key_to_index[cap]
			except KeyError:
				print("still not good!",cap)
			else:
				words[i] = cap
			
	
	board,spaces,selected = makeBoard()	
	spymast(model,board)























