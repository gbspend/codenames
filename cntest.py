import cnai, cngame
from random import randint

def sanity_test():
	blueShouldWin = bool(randint(0,1))
	game = cngame.Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(2 if blueShouldWin else 1), cnai.Cheatmaster(), cnai.CheatGuesser(1 if blueShouldWin else 2))
	blueWon, dummy = game.play()
	assert blueShouldWin == blueWon

#tests whether w2v can win with a terrible hint (CHEAT) each time against cheater-n
def testcheatw2v(n = 1):
	game = cngame.Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Cheatmaster(), cnai.W2VGuesser())
	return game.play()

#play 1 game of cheater-n (U) vs W2V hinter/guesser (R)
def testCheatVsW2V(n):
	game = cngame.Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Spymaster(cnai.W2VAssoc()), cnai.W2VGuesser())
	return game.play()

#play 1 game of cheater-n (U) vs W2V hinter + GPT2Embed guesser (R)
def testCheatVsW2VGPTE(n, assas=True):
	game = cngame.Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Spymaster(cnai.W2VAssoc()), cnai.GPT2EmbedGuesser())
	game.count_assassin = assas
	return game.play()

#play 1 game of cheater-n (U) vs GPT2Embed hinter + GPT2Embed guesser (R)
def testCheatVsGPTEGPTE(n, assas=True):
	game = cngame.Codenames(cnai.Cheatmaster(), cnai.CheatGuesser(n), cnai.Spymaster(cnai.GPT2EmbedAssoc()), cnai.GPT2EmbedGuesser())
	game.count_assassin = assas
	return game.play()

#test GPT2PromptAssoc with the provided prompt with/against other provided AI
#	prompt=None for default
#	testPrompt(cnai.Cheatmaster(), cnai.CheatGuesser(1),PROMPT,cnai.GPT2EmbedGuesser(),False)
def testPrompt(blue_hint, blue_guess, red_prompt, red_guess, assas=True):
	game = cngame.Codenames(blue_hint,blue_guess, cnai.Spymaster(cnai.GPT2PromptAssoc(red_prompt, False)), red_guess)
	game.count_assassin = assas
	return game.play()

p1 = '''These words are related to ambulance: paramedic, emergency, doctor.
These words are related to foot: shoe, sock, race.
These words are related to oil: barrel, gas, crude.
These words are related to PROMPT: '''

if __name__ == "__main__":
	for i in range(3):
		blueWon,hist = testPrompt(cnai.Cheatmaster(), cnai.CheatGuesser(1), p1, cnai.GPT2EmbedGuesser(), False)
		print("blue" if blueWon else "RED","won")
		cngame.pprintHist(hist)
	exit(0)
	
	sanity_test()
	testcheatw2v()
	testCheatVsW2V(2)
	testCheatVsW2VGPTE(1,False)
	testCheatVsGPTEGPTE(1)
	exit(0)

	for i in range(10):
		winner, hist = testCheatVsGPTEGPTE(1,False)
		print(i, "Blue won..." if winner else "RED WON!")
		if not winner:
			pprintHist(hist)
			print()
	
#

#TODO: test harness for a GPT2Prompt spymaster given an arbitrary prompt :)
#	for completeness, test that 'master with all (both?) guessers

