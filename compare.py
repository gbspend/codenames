# inspired by "30 Weird Chess Algorithms: Elo World" by suckerpinch
#	https://www.youtube.com/watch?v=DpXy041BIlA

import cnai
import cngame

#all teams that will be played against each other (excluding cheats bc they don't mix)
#	(name,hinter_obj,guesser_obj,include) tuple
#		include is bool: True for everything except cheat teams; means they won't appear on left axis
def popTeams():
	teams = []
	
	#(name,obj) tuples
	hinters = [("w2v",cnai.Spymaster(cnai.W2VAssoc())), ("gpte", cnai.Spymaster(cnai.GPT2EmbedAssoc()))]
	guessers = [("w2v",cnai.W2VGuesser()), ("gpte", cnai.GPT2EmbedGuesser())]
	
	for h in hinters:
		hname, h = h
		for g in guessers:
			gname, g = g
			teams.append((hname+"4"+gname,h,g))
			#e.g. w2v4gpte "word2vec makes hints FOR gpte"

	return teams

if __name__ == '__main__':
	teams = popTeams()
	