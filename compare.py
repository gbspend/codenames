# inspired by "30 Weird Chess Algorithms: Elo World" by suckerpinch
#	https://www.youtube.com/watch?v=DpXy041BIlA

import cnai
import cngame
import json
import os
from itertools import combinations
from sys import argv

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

#all teams that will be played against each other
#	dict of name -> (hinter_obj,guesser_obj) tuple
#	exclude cheats manually later
def makeTeams():
	teams = {}
	
	#(name,obj) tuples
	hinters = [
		("w2v",cnai.Spymaster(cnai.W2VAssoc())),
		("gpte", cnai.Spymaster(cnai.GPT2EmbedAssoc())),
		("gptp", cnai.Spymaster(cnai.GPT2PromptAssoc()))
	]
	guessers = [("w2v",cnai.W2VGuesser()), ("gpte", cnai.GPT2EmbedGuesser())]
	
	for h in hinters:
		hname, h = h
		for g in guessers:
			gname, g = g
			tname = hname+"4"+gname
			teams[tname] = (h,g)
			#e.g. w2v4gpte "word2vec makes hints FOR gpte"
	
	for i in [1,2]:
		teams["cheat"+str(i)] = (cnai.Cheatmaster(), cnai.CheatGuesser(i))
	teams["rand"] = (cnai.Cheatmaster(), cnai.RandomGuesser())

	return teams

if __name__ == '__main__':
	repeat = False
	if len(argv) > 1 and argv[1] == '-r':
		repeat = True
		print("repeating...")
	
	teams = makeTeams()
	team_names = teams.keys()
	
	#for each team pairing, play 10(?) games, write total cumulative results to disk, repeat
	combos = list(combinations(team_names,2)) #list so it's not a 1-time iterator
	
	fname = "results.json"
	results = None
	try:
		if os.path.exists(fname):
			with open(fname) as fin:
				results = json.load(fin)
				print("json loaded")
	except: 
		pass
	if results is None:
		#dict: for each key, for each key store [win,loss]
		results = {key : {sub : [0,0] for sub in team_names if sub != key} for key in team_names}
	n_games = 10
	
	print("NO ASSASSIN") #test

	while True:
		for combo in combos:
			blue_n,red_n = combo
			params = [*teams[blue_n],*teams[red_n]]
		
			print(blue_n,'vs',red_n)

			for i in range(n_games):
				print(' init ', end='')
				game = cngame.Codenames(*params)
				print('start ', end='')
				game.count_assassin = False #test
				#print('.', end='')
				blue_won,hist = game.play()
				if blue_won:
					winner = blue_n
					loser = red_n
				else:
					winner = red_n
					loser = blue_n
				results[winner][loser][0]+=1
				results[loser][winner][1]+=1
			print()
		
		print()
		print(results)
		print()
		backup(fname)
		with open(fname,'w') as f:
			json.dump(results, f)
		if not repeat:
			break
#














