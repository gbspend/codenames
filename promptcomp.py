# inspired by "30 Weird Chess Algorithms: Elo World" by suckerpinch
#	https://www.youtube.com/watch?v=DpXy041BIlA

import cnai
import cngame
import json
import os
#from itertools import combinations
from sys import argv

from cnai import jurassic

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


prompt_2pos = '''This is a list of words related to flag and state: country, government, county.
This is a list of words related to mammoth and pyramid: ancient, large, heavy.
This is a list of words related to bridge and skyscraper: concrete, blueprint, tall.
This is a list of words related to POS and POS: '''

prompt_posneg = '''This is a list of words that are related to ambulance but not doctor: siren, engine, fast.
This is a list of words that are related to bat but not duck: cave, night, fur.
This is a list of words that are related to queen but not king: regina, woman, wife.
This is a list of words that are related to POS but not NEG: '''

if __name__ == '__main__':
	repeat = False
	if len(argv) > 1 and argv[1] == '-r':
		repeat = True
		print("repeating...")

	teams = {}
	
	#(name,obj) tuples
	hinters = [
		#("w2v",cnai.Spymaster(cnai.W2VAssoc())),
		#("gpte", cnai.Spymaster(cnai.GPT2EmbedAssoc())),
		("1pos", cnai.Spymaster(cnai.GPT2PromptAssoc(1))), #default is 1 pos
		("2pos", cnai.Spymaster(cnai.GPT2PromptAssoc(2))),
		("posneg", cnai.Spymaster(cnai.GPT2PromptAssoc(3)))
	]
	guessers = [("w2v",cnai.W2VGuesser()), ("gpte", cnai.GPT2EmbedGuesser())]
	
	for h in hinters:
		hname, h = h
		for g in guessers:
			gname, g = g
			tname = hname+"4"+gname
			teams[tname] = (h,g)
			#e.g. w2v4gpte "word2vec makes hints FOR gpte"
	
	opps = {}
	for i in [1,2]:
		opps["cheat"+str(i)] = (cnai.Cheatmaster(), cnai.CheatGuesser(i))
	opps["rand"] = (cnai.Cheatmaster(), cnai.RandomGuesser())

	team_names = teams.keys()
	
	#for each team pairing, play 10(?) games, write total cumulative results to disk, repeat
	#combos = list(combinations(team_names,2)) #list so it's not a 1-time iterator
	combos = []
	for t in teams:
		for o in opps:
			combos.append((t,o))
	
	fname = "prompt_results.json"
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
		results = {key : {op : [0,0] for op in opps.keys()} for key in team_names}
	n_games = 10
	
	while True:
		for combo in combos:
			blue_n,red_n = combo
			params = [*teams[blue_n],*opps[red_n]]
		
			print(blue_n,'vs',red_n)

			for i in range(n_games):
				game = cngame.Codenames(*params)
				blue_won,hist = game.play()
				windex = 0 if blue_won else 1
				results[blue_n][red_n][windex]+=1
		
		print()
		print(results)
		print()
		backup(fname)
		with open(fname,'w') as f:
			json.dump(results, f)
		if not repeat:
			break
#














