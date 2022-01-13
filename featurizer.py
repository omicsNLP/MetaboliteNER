import time
import sys
import os
import math
import random
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import json
from extract_rf import ExtractedRandomForest

from transformers import AutoTokenizer


def _getpath(fn):
    if __name__ == "__main__":
        return fn
    d = os.path.split(__file__)[0]
    return os.path.join(d, fn)


def mi(joint, a, b, total):
    """
    Signed Mutual information between two binary things.

    Example: papers in a corpus that may or may not contain Word A or Word B

    Args:
            joint: number of items containing both A and B
            a: number of items containing A (regardless of B)
            b: number of items containing B (regardless of A)
            total: number of items (regardless of A or B)

    Returns:
            Mutual Information between A and B, multiplied by -1 if A and B are anti-correlated.
    """
    aa = joint
    ab = a - joint
    ba = b - joint
    bb = total - ab - ba - joint
    eaa = a * b / total
    eab = a * (total-b) / total
    eba = b * (total-a) / total
    ebb = (total-a) * (total-b) / total
    o = [aa, ab, ba, bb]
    e = [eaa, eab, eba, ebb]
    score = 0
    for i in range(4):
        try:
            if o[i] != 0:
                score += (o[i]/total) * math.log(o[i]/e[i])
        except:
            print(joint, a, b, total, file=sys.stderr)
            print(o, file=sys.stderr)
            print(e, file=sys.stderr)
            raise Exception()
    if o[0] < e[0]:
        score = -score
    return score


lcre = re.compile("[a-z]")
acre = re.compile("[A-Za-z]")
digre = re.compile("[0-9]")

featres = {"char": re.compile("^.*[A-Za-z].*$"),
           "hasdigit": re.compile("^.*[0-9].*"),
           "isdigit": re.compile("^[0-9]"),
           "2lc": re.compile("^.*[a-z][a-z].*"),
           "singlelower": re.compile("^[a-z]$"),
           "singleupper": re.compile("^[A-Z]$"),
           "2uc": re.compile("^.*[A-Z][A-Z].*$"),
           "d_and_c": re.compile("^.*([A-Za-z].*[0-9]|[0-9].*[A-Za-z]).*$"),
           "num": re.compile("^-?[0-9]+(\\.[0-9]+)?$"),
           "pc": re.compile("^-?[0-9]+(\\.[0-9]+)?%$")
           }
frk = sorted(featres.keys())

ws_d = re.compile("[0-9]+")
ws_l = re.compile("[a-z][a-z]+")
ws_sl = re.compile("[a-z]")
ws_c = re.compile("[A-Z][A-Z]+")
ws_sc = re.compile("[A-Z]")


def wordshape(word):
    """
    Transform a word to a "word shape" - a string of digits indicating what sorts of characters are involved. 

    Args:
            word: the word to transform

    Returns:
            a string.	
    """
    ws = word
    ws = ws_d.sub("0", ws)
    ws = ws_l.sub("1", ws)
    ws = ws_sl.sub("2", ws)
    ws = ws_c.sub("3", ws)
    ws = ws_sc.sub("4", ws)
    return ws


class Featurizer(object):
    """
    Generates token-internal features for individual tokens, mainly to tell whether they are part of a chemical
    named entity or not. Most notable - output from a Random Forest.
    """

    def __init__(self, train=None, model=None, extrachem=None, bert_pretrain_path="bert-base-cased"):#"path/to/biobert"):
        """
        Train the Featurizer, or 

        Args:
                train: if not None, the training sequences from corpusreader
                model: if not None, a text file or JSON object produced by the Featurizer
        """
        self.extrachem = extrachem
        self.bert_pretrain_path=bert_pretrain_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_pretrain_path, do_lower_case=False)
        self._load_dicts()

        if train is not None:
            self._init_train(train)
        else:
            # model can be a text file or a JSON object
            self._deserialize(model)

        

    def _init_train(self, train):
        self.train = train

        self.cache = {}

        # Tokens that are in entities...
        enttok = set()
        # Or not in entities
        otok = set()

        # Count occurrences
        self.tokcounts = {}

        # Training data comes in pairs: title and abstract
        # For each title-and-abstract pair, get set of tokens, then add the tokens in the set to the count
        for i in range(0, len(train), 2):
            btok = set()
            for seq in train[i:i+2]:
                for i in range(len(seq["tokens"])):
                    btok.add(seq["tokens"][i])
            for tok in btok:
                if tok not in self.tokcounts:
                    self.tokcounts[tok] = 0
                self.tokcounts[tok] += 1

        # See if tokens appear inside entities or not
        for seq in train:
            for i in range(len(seq["tokens"])):
                if seq["bio"][i] == "O":
                    otok.add(seq["tokens"][i])
                else:
                    enttok.add(seq["tokens"][i])

        # Remove those which can appear in both contexts
        netok = [i for i in enttok if i not in otok]
        notok = [i for i in otok if i not in enttok]
        self.etokset = set(netok)
        self.otokset = set(notok)
        # All the non-ambiguous tokens worth considering
        self.toks = netok + notok
        # ...and an answer key
        self.isne = [1 for i in netok] + [0 for i in notok]
        # Shuffle the lists so we can do 5-fold stuff etc. later
        random.seed(0)
        shuf_ids = list(range(len(self.toks)))
        random.shuffle(shuf_ids)
        tt = [self.toks[i] for i in shuf_ids]
        self.toks = tt
        self.tokd = {self.toks[i]: i for i in range(len(self.toks))}
        isn = [self.isne[i] for i in shuf_ids]
        self.isne = isn

        self.necount = len(netok)
        self.totcount = len(self.isne)

        # For each token, generate features
        self.feats = [self._feats_for_tok(i) for i in self.toks]
        numfeats = len(self.feats[0][0])

        strfeats = set()
        for f in self.feats:
            strfeats.update(f[1])
        lstrfeats = list(strfeats)
        sfd = {lstrfeats[i]: i for i in range(len(lstrfeats))}
        fv = []

        t = time.time()

        # Count how many times the features appear in e and o tokens
        ecounts = [0 for i in lstrfeats]
        ocounts = [0 for i in lstrfeats]
        tecount = 0
        tocount = 0
        for i in range(len(self.feats)):
            counts = ecounts if self.isne[i] else ocounts
            if self.isne[i]:
                tecount += 1
            else:
                tocount += 1
            for f in self.feats[i][1]:
                counts[sfd[f]] += 1
        # Calculate mutual information scores for features
        for i in range(len(lstrfeats)):
            e = ecounts[i]
            o = ocounts[i]
            sf = lstrfeats[i]
            m = mi(e, e+o, tecount, tecount+tocount)
            fv.append(
                (sf.encode("charmap", errors="replace").decode("charmap"), sf, e, o, m))

        print("Gather time:", time.time()-t, file=sys.stderr)
        print("Select from", len(lstrfeats), "features", file=sys.stderr)
        fvs = sorted(fv, key=lambda x: -abs(x[4]))

        # Select top 1000 features
        self.selfeat = [i[0] for i in fvs[:1000]]
        self.selfeatd = {self.selfeat[i]: i for i in range(len(self.selfeat))}

        # Different features - string features to go straight to the neural network ("nnfeats")
        nnstrfeats = set()
        # Count how many times each nnfeat occurs
        fctr = Counter()
        for i in range(len(self.feats)):
            n = self.tokcounts[self.toks[i]
                               ] if self.toks[i] in self.tokcounts else 0
            # self.feats[i][2] = the nnfeats
            for ff in self.feats[i][2]:
                fctr[ff] += n
        # 100 most common
        self.nnfeats = [i[0] for i in fctr.most_common(100)]
        self.nnfeatd = {self.nnfeats[i]: i for i in range(len(self.nnfeats))}

        # Initialise a cache
        self.cachescores = [0 for i in range(len(self.toks))]

        # Build the random forest, and populate the cache
        self._xval_and_train_rf()

    def _load_dicts(self):
        # Load in a few lexicons
        self.usdw = set()
        self.usdwl = set()
        # /usr/share/dict/words
        f = open(_getpath("words.txt"), "r",
                 encoding="utf-8", errors="replace")
        for l in f:
            l = l.strip()
            if len(l) > 0:
                self.usdw.add(l)
                self.usdwl.add(l.lower())
        self.elements = set()

        self.chebinames = set()
        self.chebinamesl = set()
        self.chebinameswl = set()
                
        f = open("MetaboliteNames.txt", "r", encoding="utf-8", errors="replace")
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_pretrain_path)
        for l in f:
            l = l.strip()
            if len(l) > 0:
                ct = list(self.tokenizer.tokenize(l))
                for tok in ct:
                    self.chebinames.add(tok)
                    self.chebinamesl.add(tok.lower())
                if len(ct) == 1:
                    self.chebinameswl.add(l.lower())
        if self.extrachem is not None:
            self.chemspider = set()
            self.chemspiderl = set()
            f = open(self.extrachem, "r", encoding="utf-8", errors="replace")
            for l in f:
                l = l.strip()
                if len(l) > 0:
                    ct = self.tokenizer(l)
                    for tok in ct.getTokenStringList():
                        self.chemspider.add(tok)
                        self.chemspiderl.add(tok.lower())

    def to_json_obj(self):
        """
        Makes an object suitable for JSON serialization that can be loaded back in later.

        Returns:
                dictionary
        """
        jo = {}
        jo["selfeat"] = self.selfeat
        jo["nnfeats"] = self.nnfeats
        jo["rf"] = self.erf.ets
        jo["cache"] = self.cachecopy
        jo["tokd"] = self.tokd
        return jo

    def _deserialize(self, arg):
        if type(arg) is str:
            fn = arg
            f = open(fn, "r", encoding="utf-8")
            jo = json.load(f)
            f.close()
        else:
            jo = arg
        self.selfeat = jo["selfeat"]
        self.selfeatd = {self.selfeat[i]: i for i in range(len(self.selfeat))}
        self.nnfeats = jo["nnfeats"]
        self.nnfeatd = {self.nnfeats[i]: i for i in range(len(self.nnfeats))}
        self.erf = ExtractedRandomForest(jo["rf"])
        self.cachescores = jo["cache"]
        self.cache = {}
        self.tokd = jo["tokd"]

    def _make_rf(self, x, y):
        t = time.time()
        rf = RandomForestClassifier(n_estimators=100, oob_score=False)
        rff = rf.fit(x, y)
        print("Forest in", time.time()-t, file=sys.stderr)
        return rf

    def _xval_and_train_rf(self):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        # 5-fold cross-validation - mainly to see what's going on,
        # but also to get scores for tokens in our training set
        for fold in range(5):
            # select
            testset = []
            trainset = []
            for i in range(len(self.toks)):
                if i % 5 == fold:
                    testset.append(i)
                else:
                    trainset.append(i)
            # gather
            x = np.array([self._sfeats_to_bits(self.feats[i][1])
                         for i in trainset])
            y = np.array([self.isne[i] for i in trainset])
            test_x = np.array([self._sfeats_to_bits(
                self.feats[i][1]) for i in testset])
            test_y = np.array([self.isne[i] for i in testset])
            # build the RF
            rf = self._make_rf(x, y)
            # Memorize the scores
            ppreds = rf.predict_proba(test_x)
            for i in range(len(testset)):
                self.cachescores[testset[i]] = ppreds[i][1]
            # Gather stats on how we're doing
            for i in range(len(ppreds)):
                if ppreds[i][1] > 0.5:
                    if test_y[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if test_y[i] == 1:
                        fn += 1
                    else:
                        tn += 1

        # Take a copy of cachescores so we can serialize it
        self.cachecopy = self.cachescores.copy()
        # How's the score?
        if(tp+fp+fn+tn > 0):
            print(tp, fp, fn, tp*2/(tp+tp+fp+fn), (tp+tn) /
                  (tp+fp+fn+tn), sep="\t", file=sys.stderr)
        # OK, enough with the cross-validation, let's generate the classifier for unknown tokens.
        x = np.array([self._sfeats_to_bits(self.feats[i][1])
                     for i in range(len(self.toks))])
        y = np.array([self.isne[i] for i in range(len(self.toks))])
        self.rforest = self._make_rf(x, y)
        # scikit-learn random forests need pickling, which creates compatibility issues,
        # so let's extract the data and put it in our own object
        self.erf = ExtractedRandomForest(self.rforest)

    def _sfeats_to_bits(self, feats):
        b = [0 for i in self.selfeat]
        for f in feats:
            if f in self.selfeatd:
                b[self.selfeatd[f]] = 1
        return b

    def _nnfeats_to_bits(self, feats):
        b = [0 for i in self.nnfeats]
        for f in feats:
            if f in self.nnfeatd:
                b[self.nnfeatd[f]] = 1
        return b

    def _feats_for_tok(self, tok):
        """
        Generate features for a token, including those to be fed only to the Random Forest
        """
        # Numerical features - i.e. those that aren't necessarily 1 or 0.
        num_feats = [len(tok),
                     math.log(len(tok)),
                     math.sqrt(len(tok)),
                     len(lcre.sub("", tok)),
                     len(lcre.sub("", tok))/len(tok),
                     len(acre.sub("", tok)),
                     len(acre.sub("", tok))/len(tok),
                     len("".join(digre.findall(tok)))
                     ]
        # features for random forest
        rf_feats = []
        ws = wordshape(tok)
        rf_feats.append("ws=%s" % ws)
        lf = "^%s$" % tok
        rf_feats += ["4g="+lf[i:i+4] for i in range(len(lf)-3)]
        rf_feats += ["3g="+lf[i:i+3] for i in range(len(lf)-2)]
        rf_feats += ["2g="+lf[i:i+2] for i in range(len(lf)-1)]
        rf_feats += ["1g="+lf[i:i+1] for i in range(1, len(lf)-1)]
        for featre in featres:
            if featres[featre].match(tok):
                rf_feats.append("re="+featre)
        if tok in self.usdw:
            rf_feats.append("dict=usdw")
        if tok in self.chebinames:
            rf_feats.append("dict=chebi")
        if self.extrachem is not None and tok in self.chemspider:
            rf_feats.append("dict=cs")
        if tok.lower() in self.usdwl:
            rf_feats.append("dict=usdwl")
        if tok.lower() in self.chebinamesl:
            rf_feats.append("dict=chebil")
        if tok.lower() in self.chebinameswl:
            rf_feats.append("dict=chebiwl")
        if self.extrachem is not None and tok.lower() in self.chemspiderl:
            rf_feats.append("dict=csl")

        # Not strictly speaking "true" numerical features, but include them here because
        # we want to pass them direct to the neural net
        num_feats.extend([
            1 if tok in self.usdw else 0,
            1 if tok in self.chebinames else 0,
            #1 if tok in self.chemspider else 0,
            1 if tok.lower() in self.usdwl else 0,
            1 if tok.lower() in self.chebinamesl else 0,
            1 if tok.lower() in self.chebinameswl else 0  # ,
        ])
        num_feats.extend(
            [1 if featres[featre].match(tok) else 0 for featre in frk])

        # We can only pass the top hundred or so of these to the neural net
        nn_feats = []
        nn_feats.append("ws=%s" % ws)
        nn_feats.append("suf3="+tok[-3:])
        nn_feats.append("suf2="+tok[-2:])

        return (num_feats, set(rf_feats), set(nn_feats))

    def num_feats_for_tok(self, tok):
        """
        Generate numerical features for a token.

        Args:
                tok: the token string

        Returns:
                Numpy array of floats
        """

        if tok in self.cache:
            return self.cache[tok]
        feats = self._feats_for_tok(tok)

        numfeats = []
        if tok in self.tokd:
            pp = self.cachescores[self.tokd[tok]]
        else:
            sfb = self._sfeats_to_bits(feats[1])
            pp = self.erf.predict_proba(sfb)[1]

        numfeats.append(pp)
        nnb = self._nnfeats_to_bits(feats[2])

        numfeats.extend(nnb + feats[0])

        numfeats = np.array(numfeats)
        self.cache[tok] = numfeats
        return numfeats
