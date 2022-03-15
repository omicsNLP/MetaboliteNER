import time
import sys
import random
import json
import numpy as np
import shutil
from collections import defaultdict

from datetime import datetime

import tensorflow as tf
import transformers

from featurizer import Featurizer
from utils import *
from corpusreader import CorpusReader

class MetaboListem(object):
    """
    A metabolite named entity recognition model, adapted from the chemical named entity recognition model ChemListem.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        # bert_pretrain_path = "PATH/TO/biobert-base-cased-v1.2"
        bert_pretrain_path = "bert-base-cased"
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            bert_pretrain_path, do_lower_case=False)

    def _str_to_seq(self, s):
        # Generate sequence of str. Used for input of BIOES
        seq = {"tokens": [], "bio": [],
               "tokstart": [], "tokend": [], "str": s}
        tokenized = self.tokenizer.tokenize('[CLS]'+s)[1:]
        seq["tokens"] = list(tokenized)

        while tokenized:
            tok = tokenized.pop()
            if len(tok) > 2:
                if tok[0:2] == '##':
                    tok = tok[2:]

            tokstart = s.rfind(tok)
            tokend = tokstart+len(tok)
            s = s[:tokstart]
            seq["tokstart"].insert(0, tokstart)
            seq["tokend"].insert(0, tokend)

        return seq

    def _prepare_seqs(self, seqs, verbose=True):
        """
        Add features to sequences.
        """
        if verbose:
            print("Number words at", datetime.now(), file=sys.stderr)
        # Tokens to numbers, for use with embeddings
        for seq in seqs:
            seq["wordn"] = [self.tokdict[i] if i in self.tokdict else self.tokdict["[UNK]"]
                            for i in seq["tokens"]]
        if verbose:
            print("Generate features at", datetime.now(), file=sys.stderr)
        # Words to name-internal features
        for seq in seqs:
            seq["ni"] = np.array([self.fzr.num_feats_for_tok(i)
                                 for i in seq["tokens"]])
        if verbose:
            print("Generated features at", datetime.now(), file=sys.stderr)

        # Take a note of how many name-internal features there are
        self.nilen = len(seqs[0]["ni"][0])
        
    def build_model(self,em):
        ohn = len(self.toklist)
        il = tf.keras.Input(shape=(None, self.nilen))
        cl = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu',
                                    input_shape=(None, self.nilen), name="conv")(il)
        cdl = tf.keras.layers.Dropout(0.5)(cl)
        ei = tf.keras.Input(shape=(None,), dtype='int32')
        if em is not None:
            el = tf.keras.layers.Embedding(ohn, 300, weights=[em])(ei)
        else:
            el = tf.keras.layers.Embedding(ohn, 300)(ei)
        ml = tf.keras.layers.concatenate([cdl, el])

        blld = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.5, kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=0.000001, l2=0.000001),
            recurrent_dropout=0, recurrent_activation="sigmoid", activation='tanh'), merge_mode="concat", name="lstm")(ml)
        dl = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            len(self.lablist), activation="softmax"), name="output")(blld)
        
        model = tf.keras.models.Model(inputs=[ei, il], outputs=dl)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

    def train(self, textfile, annotfile, glovefile, runname):
        """
        Train a new MetaboListem.

        Using pre-trained embedding vectors from GloVe is recommended. This is not contained in the package, but can be downloaded at: 
        
        https://nlp.stanford.edu/projects/glove/

        The training process presumes the 6B 300d GloVe embedding ("glove.6B.300d.txt").

        This function produces a trained model, which is constituted of two main files:

        metabolistem_$RUNNAME.h5 - the keras model
        metabolistem_$RUNNAME.json - auxilliary information

        Each epoch in the training process also produces a model file on its own; the files are named: 

        epoch_$EPOCHNUM_$RUNAME.h5

        All such models share the same json file as the auxilliary information remains unchanged.

        Args:
            textfile: the filename of the file containing the training text - e.g. "TrainingSet.txt"
            annotfile: the filename of the training annotation file - e.g. "TrainingSetAnnot.tsv"
            glovefile: None, or the filename of the glove file - e.g. "glove.6B.300d.txt"
            runname: Custom name of output file
        """

        # Get training and test sequences
        cr = CorpusReader(textfile, annotfile)
        train = cr.trainseqs
        # test = cr.testseqs

        seqs = train# +test

        # Initialise some stuff
        toklist = []
        tokdict = {}
        tokcounts = {}
        labels = set()
        self.toklist = toklist
        self.tokdict = tokdict
        self.tokcounts = tokcounts
        self.fzr = None  # Make later
        self.lablist = None  # Do later
        self.labdict = None  # Do later
        self.model = None  # Do later

        #

        tokdict["[PAD]"] = len(toklist)
        toklist.append("[PAD]")

        # Count tokens in training data....
        for seq in train:
            for tok in seq["tokens"]:
                if tok not in tokcounts:
                    tokcounts[tok] = 0
                tokcounts[tok] += 1

        # and keep those that occur more than twice
        for tok in list(tokcounts.keys()):
            if tokcounts[tok] > 2:
                tokdict[tok] = len(toklist)
                toklist.append(tok)
        for u in ["[UNK]"]:
            tokdict[u] = len(toklist)
            toklist.append(u)

        ohn = len(toklist)

        # Initialise embeddings using GloVe if present
        em = None
        ei = {}
        self.ei = ei
        if glovefile is not None:
            t = time.time()

            f = open(glovefile, "r", encoding="utf-8")
            for l in f:
                ll = l.strip().split()
                w = ll[0]
                c = np.asarray(ll[1:], dtype='float32')
                ei[w] = c
            em = np.zeros((ohn, 300))
            for i in range(ohn):
                if toklist[i] in ei:
                    em[i] = ei[toklist[i]]
            print("Embeddings read in:", time.time() - t, file=sys.stderr)

        # Collect labels for tokens
        for seq in seqs:
            for i in seq["bio"]:
                labels.add(i)
        lablist = sorted(labels)
        lablist.reverse()
        labdict = {lablist[i]: i for i in range(len(lablist))}
        self.lablist = lablist
        self.labdict = labdict

        # Convert SOBIE tags to numbers
        for seq in seqs:
            seq["bion"] = [labdict[i] for i in seq["bio"]]

        # Build the "featurizer" which generates token-internal features
        print("Make featurizer at", datetime.now(), file=sys.stderr)
        fzr = Featurizer(train)
        self.fzr = fzr

        # Marshal features for each token
        self._prepare_seqs(seqs)

        # Gather together sequences by length
        print("Make train dict at", datetime.now(), file=sys.stderr)

        train_l_d = {}
        for seq in train:
            l = len(seq["tokens"])
            if l not in train_l_d:
                train_l_d[l] = []
            train_l_d[l].append(seq)

        self.build_model(em)
        model=self.model

        # Serialize the auxilliary intformation
        outjo = {
            "tokdict": self.tokdict,
            "fzr": self.fzr.to_json_obj(),
            "lablist": self.lablist
        }

        print("Serialize at", datetime.now(), file=sys.stderr)
        jf = open("metabolistem_%s.json" % runname, "w", encoding="utf-8")
        json.dump(outjo, jf)
        jf.close()

        sizes = list(train_l_d)

        best_epoch = -1
        best_f = 0.0

        # Start training
        for epoch in range(30):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)

            # Train in batches of different sizes - randomize the order of sizes
            random.shuffle(sizes)
            tnt = 0
            for size in sizes:
                tnt += size * len(train_l_d[size])
            totloss = 0
            totacc = 0
            div = 0
            for size in sizes:
                if size == 1:
                    continue  # For unknown reasons we can't train on a single token
                batch = train_l_d[size]

                tx2 = np.array([seq["ni"] for seq in batch])
                ty = np.array([[tobits(i, len(lablist))
                              for i in seq["bion"]] for seq in batch])
                tx = np.array([seq["wordn"] for seq in batch])
                history = model.fit([tx, tx2], ty, verbose=0, epochs=1)
                div += size * len(batch)
                totloss += history.history["loss"][0] * size * len(batch)
                totacc += history.history["accuracy"][0] * size * len(batch)
                # This trains in mini-batches

            print("Trained at", datetime.now(), "Loss", totloss /
                  div, "Accuracy", totacc / div, file=sys.stderr)
            self.model.save("epoch_%s_%s.h5" % (epoch, runname))
            
    def load(self, jfile, mfile):
        """
        Load in model data.

        Args:
            jfile: the filename of the .json file
            mfile: the filename of the .h5 file
        """
        self.unkcache = {}

        jf = open(jfile, "r", encoding="utf-8")
        jo = json.load(jf)
        jf.close()
        self.tokdict = jo["tokdict"]
        self.lablist = jo["lablist"]
        self.fzr = Featurizer(None, jo["fzr"])
        print("Auxillary information read at", datetime.now(), file=sys.stderr)
        self.model = tf.keras.models.load_model(mfile)
        print("Model read at", datetime.now(), file=sys.stderr)

    def process(self, instr, threshold=0.5, domonly=True):
        """
        Find named entities in a string.

        Entities are returned as list of tuples:
        (start_charater_position, end_character_position, string, score, is_dominant)

        Entities are dominant if they are not partially or wholly overlapping with a higher-scoring entity.

        Args:
            instr: the string to find entities in.
            threshold: the minimum score for entities.
            domonly: if True, discard non-dominant entities.
        """

        results = []
        post_processed_res = []
        if len(instr) == 0:
            return results, post_processed_res
        seq = self._str_to_seq(instr)
        if len(seq["tokens"]) == 0:
            return results, post_processed_res
        self._prepare_seqs([seq], False)  # Not verbose
        mm = self.model.predict(
            [np.array([seq["wordn"]]), np.array([seq["ni"]])])[0]
        seq["tagfeat"] = mm
        pents, pxe = sobie_scores_to_char_ents(seq, threshold, instr)
        if domonly:
            pents = [i for i in pents if pxe[i]["dom"]]

        loc_word_dict = {}
        for ent in pents:
            loc_word_dict[(ent[1], ent[2])] = instr[ent[1]:ent[2]]
            

        results=post_process(loc_word_dict,instr)
        return results

    def batchprocess(self, instrs, threshold=0.5, domonly=True):
        """
        Find named entities in a set of strings. This is potentially faster as neural network calculations
        run faster in batches.

        Entities are returned as tuples:
        (start_charater_position, end_character_position, string, score, is_dominant)

        Entities are dominant if they are not partially or wholly overlapping with a higher-scoring entity.

        Args:
            instrs: the string to find entities in.
            threshold: the minimum score for entities.
            domonly: if True, discard non-dominant entities.
        """
        pairs = [(n, self._str_to_seq(i)) for n, i in enumerate(instrs)]
        seqs = [i[1] for i in pairs]
        self._prepare_seqs(seqs, False)
        seq_l_d = defaultdict(lambda: [])
        res = [list() for i in instrs]
        for pair in pairs:
            seq = pair[1]
            l = len(seq["tokens"])
            seq_l_d[len(seq["tokens"])].append(pair)
        for l in seq_l_d:
            if l == 0:
                continue
            else:
                ppairs = seq_l_d[l]
                mm = self.model.predict([np.array([p[1]["wordn"] for p in ppairs]), np.array(
                    [p[1]["ni"] for p in ppairs])])
                for n, p in enumerate(ppairs):
                    p[1]["tagfeat"] = mm[n]
                    pents, pxe = sobie_scores_to_char_ents(
                        p[1], threshold, p[1]["str"])
                    if domonly:
                        pents = [i for i in pents if pxe[i]["dom"]]
                        
                    loc_word_dict = {}
                    for ent in pents:
                        loc_word_dict[(ent[1], ent[2])] = p[1]["str"][ent[1]:ent[2]]

                    rr=post_process(loc_word_dict,p[1]["str"])
                    res[p[0]] = rr
        return res
