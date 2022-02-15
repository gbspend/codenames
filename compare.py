# inspired by "30 Weird Chess Algorithms: Elo World" by suckerpinch
#	https://www.youtube.com/watch?v=DpXy041BIlA

import cnai
import cngame
import json
import os
from itertools import combinations

#if fname exists, rename to [fname].bak
def backup(fname):
	bak = fname+".bak"
	try:
		os.remove(bak)
	except FileNotFoundError:
		pass
	try:
		os.rename(fname,bak)
	except FileNotFoundError:
		pass

#==================================

#all teams that will be played against each other (excluding cheats bc they don't mix)
#	(name,hinter_obj,guesser_obj,include) tuple
#		include is bool: True for everything except cheat teams; means they won't appear on left axis
def makeTeams():
	teams = {}
	
	#(name,obj) tuples
	hinters = [("w2v",cnai.Spymaster(cnai.W2VAssoc())), ("gpte", cnai.Spymaster(cnai.GPT2EmbedAssoc()))]
	guessers = [("w2v",cnai.W2VGuesser()), ("gpte", cnai.GPT2EmbedGuesser())]
	
	for h in hinters:
		hname, h = h
		for g in guessers:
			gname, g = g
			tname = hname+"4"+gname
			teams[tname] = (h,g)
			#e.g. w2v4gpte "word2vec makes hints FOR gpte"

	return teams

# NOTE: This is not for debugging purposes; only records win/loss
if __name__ == '__main__':
	teams = makeTeams()
	team_names = teams.keys()
	
	#for each team pairing, play 10(?) games, write total cumulative results to disk, repeat
	combos = combinations(team_names,2)
	
	#dict: for each key, for each key store [win,loss]
	results = {key : {sub : [0,0] for sub in team_names if sub != key} for key in team_names}
	n_games = 10
	fname = "results.json"
	
	#while True:
	for combo in combos:
		blue_n,red_n = combo
		params = [*teams[blue_n],*teams[red_n]]
		
		print(blue_n,'vs',red_n)

		for i in range(n_games):
			game = cngame.Codenames(*params)
			blue_won,hist = game.play()
			if blue_won:
				winner = blue_n
				loser = red_n
			else:
				winner = red_n
				loser = blue_n
			results[winner][loser][0]+=1
			results[loser][winner][1]+=1
			
		backup(fname)
		with open(fname,'w') as f:
			json.dump(results, f)
#














