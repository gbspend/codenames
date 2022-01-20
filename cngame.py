import cnai
from random import randint, sample, shuffle

BOARD_SIZE = 25

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

#(not going to bother with the preset board layout cards)
def newGame(blueFirst=True):	
	#I don't know how many red/blue for anything other than 5x5
	
	spaces = ['U']*8 + ['R']*8 + ['N']*7 + ['A']
	spaces += ['U'] if blueFirst else ['R']
	shuffle(spaces)
	
	selected = sample(words, BOARD_SIZE)
	
	#assert len(spaces) == len(selected) == BOARD_SIZE
	
	return spaces,selected

#TODO?: make it smart so it can handle 1p or 2p?
#probably need to reimplement to add human players (async?)
class Codenames:
	def __init__(self, us, ug, rs, rg):
		self.blue_spymaster = us
		self.blue_guesser = ug
		self.red_spymaster = rs
		self.red_guesser = rg
		self.initGame()
	
	def initGame(self):
		#make "blue" always start if it's 1 player
		self.bluesTurn = bool(randint(0,1))
		self.spaces, self.selected = newGame(self.bluesTurn)
		self.covered = [False] * BOARD_SIZE
		
		self.hist = [self.selected, self.spaces]
		self.curr_hint = None
		self.guesses_made = 0
	
	#turn the game state (spaces, selected, covered) into board dict
	#doesn't need to be saved, just re-gen when needed
	def makeBoard(self):
		board = {'U':[], 'R':[], 'N':[], 'A':[]}
		for i in range(BOARD_SIZE):
			if self.covered[i]:
				continue
			color = self.spaces[i]
			word = self.selected[i]
			board[color].append(word)
		return board
	
	#return winner bool (T: blue, F: red) and history list
	def play(self):
		if any(self.covered):
			raise ValueError("Game already finished. Call initGame() to start over.")
		while True:
			#debug:
			#print(self.hist[2:])
			#print()
			
			master = self.blue_spymaster if self.bluesTurn else self.red_spymaster
			guesser = self.blue_guesser if self.bluesTurn else self.red_guesser
			
			board = self.makeBoard()
			
			#for testing
			if guesser.isCheat():
				guesser.cheat(board, self.bluesTurn)
			
			if not self.curr_hint:
				self.curr_hint = master.makeHint(board, self.bluesTurn) # returns (word,num) tuple
				guesser.newHint(self.curr_hint)
				self.guesses_made = 0
				self.hist.append(("HINT", 'U' if self.bluesTurn else 'R', *self.curr_hint))
			else:
				choices = [word for i,word in enumerate(self.selected) if not self.covered[i]]
				guess = guesser.nextGuess(choices) #string from board
				match = None
				if guess is not None:
					guess = guess
					i = self.selected.index(guess)
					assert not self.covered[i]
					self.guesses_made += 1

					self.covered[i] = True
					color = self.spaces[i]
					match = (color == 'U' and self.bluesTurn) or (color == 'R' and not self.bluesTurn)
				self.hist.append(("GUESS", 'U' if self.bluesTurn else 'R', guess, match))
				if guess is None or not match or self.guesses_made > self.curr_hint[1]:
					self.bluesTurn = not self.bluesTurn
					self.curr_hint = None
					self.guesses_made = 0
			
			#check if game is over using most recently updated covered board
			check = self.makeBoard()
			winner = None #True for blue won, False red won
			assert check['U'] or check['R'] #only one team can win at a time
			if not check['A']:
				winner = self.bluesTurn #kinda counterintuitive, but if the team picked A it would have already switchted turns because it wasn't a correct guess
			elif not check['U']:
				winner = True
			elif not check['R']:
				winner = False
			if winner is not None:
				return winner, self.hist
#
def pprintHist(hist):
	print([(hist[0][i], hist[1][i]) for i in range(BOARD_SIZE)])
	for h in hist[2:]:
		if type(h) is not tuple:
			print(h)
		else:
			tabs = 1 #HINT
			if h[0] == "GUESS":
				tabs = 2
			print("\t"*tabs,h)
	
def sanity_test():
	blueShouldWin = bool(randint(0,1))
	game = Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(2 if blueShouldWin else 1), cnai.Cheatmaster(), cnai.CheatGuesser(1 if blueShouldWin else 2))
	blueWon, dummy = game.play()
	assert blueShouldWin == blueWon

#tests whether w2v can win with a terrible hint (CHEAT) each time against cheater-n
def testcheatw2v(n = 1):
	game = Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Cheatmaster(), cnai.W2VGuesser())
	return game.play()

#play 1 game of cheater-n (U) vs W2V hinter/guesser (R)
def testCheatVsW2V(n):
	game = Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Spymaster(cnai.W2VAssoc()), cnai.W2VGuesser())
	return game.play()
			
if __name__ == "__main__":	
	for i in range(10):
		winner, hist = testCheatVsW2V(1)
		print(i, "Blue won..." if winner else "RED WON!")
		pprintHist(hist)
		print()
	
#















