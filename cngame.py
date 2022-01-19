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
def makeBoard(blueFirst=True):	
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
		self.spaces, self.selected = makeBoard(self.bluesTurn)
		self.covered = [False] * BOARD_SIZE
		
		self.hist = []
		self.curr_hint = None
		self.guesses_made = 0
	
	#turn the game state (spaces, selected, covered) into board dict
	#doesn't need to be saved, just re-gen when needed
	def makeBoard(self):
		board = {'U':[], 'R':[], 'N':[], 'A':[]}
		for i in range(BOARD_SIZE):
			if self.covered[i]:
				continue
			color = spaces[i]
			word = selected[i]
			board[color].append(word)
		return board
	
	#return history
	def play(self):
		pass

if __name__ == "__main__":
	game = Codenames1P()
	game.play()