from cnai import W2VAssoc, GPT2EmbedAssoc, dists2words
from cngame import Codenames

def assoc2Text(assoc):
    return ", ".join([w for w,p in assoc])

w = W2VAssoc()
g = GPT2EmbedAssoc()

def compare(pos,neg):
    print("pos:",", ".join(pos))
    print("neg:",", ".join(neg))

    w_assocs = w.getAssocs([w.preprocess(s) for s in pos], [w.preprocess(s) for s in neg], 10)
    print("W2V:",assoc2Text(w_assocs))
    
    parts = g.getAssocs(pos, neg, 10)
    g_words = dists2words([(s,0.1) for s in parts])
    print("GPT:",assoc2Text())
    print()

if __name__ == '__main__':
    
    for i in range(10):
        board = Codenames(None,None,None,None).makeBoard()
        pos = board['U']
        neg = board['R'] + board['A']
        compare(pos,neg)
        
