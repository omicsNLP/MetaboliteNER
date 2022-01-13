import time
import sys
import os
import random
import json
import numpy as np
import re

from datetime import datetime

import tensorflow as tf
import transformers

from featurizer import Featurizer
from utils import *
from corpusreader import CorpusReader

import re
import chemtok

class TaboListem(object):
    """
    A model for metabolite named entity recognition.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        # bert_pretrain_path = "PATH/TO/biobert-base-cased-v1.2"
        bert_pretrain_path='bert-base-cased'
        self.bert_pretrain_path=bert_pretrain_path
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
            bert_pretrain_path, do_lower_case=False)
        self.max_seq_len=256

    def _str_to_seq(self, s):
        '''
        Computes (truncated) sequence data of input s
        
        Args:
                s: Sentence string to be processed

        Returns:
                The sequence data seq of the sentence s. Seq is a dictionary, containing: 
                    tokens: s tokenized by the predefined Transformer tokenizer, as a list of str
                    bio: BIOES tags of the tokens, as a list of str. This is an empty list before filling by consequent procedure
                    tokstart: the start position of each token of s, as a list of int
                    tokend: the end position of each token of s, as a list of int
                    chemtok_tokens: s tokenized by chemtok, as a list of str
                    chemtok_rep: number of tokens that a chemtok token splits into by Transformer, as a list of int
                    str: the input text
        '''
        seq = {"tokens": [], "bio": [],
               "tokstart": [], "tokend": [],"chemtok_tokens":[],"chemtok_rep":[], "str": s}
        ct = chemtok.ChemTokeniser(s, clm=True)
        chemtok_tokens=[t.value for t in ct.tokens]
        
        token_list=[self.tokenizer.tokenize(t) for t in chemtok_tokens]
        
        tokenized = sum(token_list,[])
        
        chemtok_rep=[len(t) for t in token_list]
        # Truncation
        len_token=len(tokenized)
        if len_token>self.max_seq_len-2:
            tokenized = tokenized[:self.max_seq_len-2]
            while len_token>self.max_seq_len-2:
                truncated_token=chemtok_tokens.pop()
                truncated_token_rep=chemtok_rep.pop()
                len_token-=truncated_token_rep
        
        seq["chemtok_tokens"]=chemtok_tokens
        seq["tokens"] = tokenized.copy()
        seq["chemtok_rep"] = chemtok_rep
        

        while tokenized:
            tok = tokenized.pop()
            if len(tok) > 2:
                if tok[0:2] == '##':
                    tok = tok[2:]

            tokstart = s.rfind(tok)
            tokend = tokstart+len(tok)
            s = s[:tokstart]
            seq["tokstart"].append(tokstart)
            seq["tokend"].append(tokend)
        seq["tokstart"].reverse()
        seq["tokend"].reverse()

        return seq

    def _prepare_seqs(self, seqs, verbose=True,save_path='Seqs.npy'):
        """
        Add numerical features to sequences. 
        
        Args:
            seqs: list of dictionaries 'seq'
            save_path: path to save prepared seqs for reusability
        """
        if verbose:
            print("Number words at", datetime.now(), file=sys.stderr)
        # Tokens to numbers, for use with embeddings
        for seq in seqs:
            seq["wordn"]=[self.tokenizer.vocab['[CLS]']]+[self.tokenizer.vocab[i] for i in seq["tokens"]][:sum(seq['chemtok_rep'])]+[self.tokenizer.vocab['[SEP]']]
        if verbose:
            print("Generate features at", datetime.now(), file=sys.stderr)
        # Words to name-internal features        
        nilen=len(self.fzr.num_feats_for_tok(seqs[0]['chemtok_tokens'][0]))
        ni_cls=[0]*nilen
        ni_sep=[0]*nilen
        
        for seq in seqs:
            tok_rep=sum([[t]*j for t,j in zip(seq['chemtok_tokens'], seq['chemtok_rep'])],[])
            seq["ni"] = np.array([ni_cls]+[self.fzr.num_feats_for_tok(i)
                                 for i in tok_rep]+[ni_sep])
        if verbose:
            print("Generated features at", datetime.now(), file=sys.stderr)

        if save_path:
            np.save(save_path,seqs)
            print("Seqs saved as "+save_path)

        # Take a note of number of name-internal features
        self.nilen=nilen
        
    def load_seqs(self,load_file='Seqs2.npy'):
        """Load the saved prepared seqs

        Args:
            load_file (str, optional): Path to data file. Defaults to 'Seqs2.npy'.

        Returns:
            list<dict>: prepared seqs
        """
        seqs=np.load(load_file,allow_pickle=True)
        self.nilen=len(seqs[0]['ni'][0])
        return seqs
    
    def build_model(self):
        """
        Build TaboListem model structure
        """
        il = tf.keras.Input(shape=(None, self.nilen))
        cl = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu',
                                    input_shape=(None, self.nilen), name="conv")(il)
        cdl = tf.keras.layers.Dropout(0.5)(cl)

        input_id_tensor = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name='input_id')
        attention_mask_tensor = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name='input_masks')

        transformer_model = transformers.TFBertForTokenClassification.from_pretrained(
            self.bert_pretrain_path, from_pt=True)
        el = transformer_model([input_id_tensor, attention_mask_tensor])[0]

        el_drop = tf.keras.layers.SpatialDropout1D(0.1)(el)
        ml = tf.keras.layers.concatenate([el_drop, cdl])

        blld = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.5, kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=0.000001, l2=0.000001),
            recurrent_dropout=0, recurrent_activation="sigmoid", activation='tanh'), merge_mode="concat", name="lstm")(ml)
        dl = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            len(self.lablist), activation="softmax"), name="output")(blld)

        model = tf.keras.models.Model(inputs=[input_id_tensor, attention_mask_tensor, il], outputs=dl)
        model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

    def train(self, textfile, annotfile, runname):
        """
        Train a new TaboListem.
        This produces several important files:

        epoch_$EPOCHNUM_$RUNNAME_weights.h5 - the weights of the keras model itself (note that only weights is saved, not the model, due to known Transformer bugs)
        tabolistem_$RUNNAME.json - various bits of auxilliary information

        These consititute the trained model.


        Args:
            textfile: the filename of the training text file
            annotfile: the filename of the training annotation file
            runname: a string, part of the output filenames.
        """

        # Get training and test sequences
        cr = CorpusReader(textfile, annotfile)
        train = cr.trainseqs
        test = cr.testseqs

        seqs = train+test

        # Initialise
        toklist = []
        tokcounts = {}
        labels = set()
        self.toklist = toklist
        self.tokdict = self.tokenizer.vocab
        self.tokcounts = tokcounts
        self.fzr = None
        self.lablist = None
        self.labdict = None
        self.model = None  

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
            seq["bion"] = [labdict["O"]]+[labdict[i] for i in seq["bio"]]+[labdict["O"]]

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

        self.build_model()
        model=self.model

        # Serialize the auxilliary information
        outjo = {
            "fzr": self.fzr.to_json_obj(),
            "lablist": self.lablist,
            "nilen":self.nilen
        }

        print("Serialize at", datetime.now(), file=sys.stderr)
        jf = open("./TrainedModels/tabolistem_%s.json" % runname, "w", encoding="utf-8")
        json.dump(outjo, jf)
        jf.close()

        sizes = list(train_l_d)

        # OK, start actually training
        for epoch in range(10):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)

            # Train in batches of different sizes - randomize the order of sizes
            random.shuffle(sizes)
            tnt = 0
            for size in sizes:
                tnt += size * len(train_l_d[size])
            totloss = 0
            totacc = 0
            div = 0
            
            time_initial = time.time()
            est_hr, est_min, est_sec = 0, 0, 0
            batch_size=4
            
            total_batches=sum([len(train_l_d[size])//batch_size+1 for size in sizes])
            total_batch_counter=0
            
            for size in sizes:
                if size == 1:
                    continue  # For unknown reasons we can't train on a single token
                
                batch_counter=0
                formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)

                
                
                while batch_counter<len(train_l_d[size]):
                    if div == 0:
                        _loss = 0
                        _acc = 0
                    else:
                        _loss = totloss/div
                        _acc = totacc/div
                    print('\rTraining on size {}, batch {}/{}... Loss: {}; Accuracy: {}; Estimated elapsed time: N/A'.format(str(size),
                    str(total_batch_counter), str(total_batches), str(_loss), str(_acc)), end='', flush=True)
                    
                    
                    batch = train_l_d[size][batch_counter:batch_counter+batch_size]
                    batch_counter+=batch_size
                    total_batch_counter+=1
                
                    wordn_batch=[seq["wordn"] for seq in batch]

                    tx2 = np.array([seq["ni"] for seq in batch])
                    ty = np.array([[tobits(i, len(lablist))
                                for i in seq["bion"]] for seq in batch])
                    tx_id = tf.convert_to_tensor(wordn_batch)
                    attention_mask=tf.convert_to_tensor(np.ones(tx_id.shape,dtype='int32'))
                    history = model.fit([tx_id, attention_mask,tx2], ty, verbose=0, epochs=1)
                    div += size * batch_size
                    totloss += history.history["loss"][0] * size * batch_size
                    totacc += history.history["accuracy"][0] * size * batch_size
                    
                    estimate_time = int((time.time()-time_initial) *
                                    (total_batches/total_batch_counter-1))
                    est_hr = int(estimate_time/3600)
                    est_min = int((estimate_time % 3600)/60)
                    est_sec = int(estimate_time % 60)

            print("Trained at", datetime.now(), "Loss", totloss /
                  div, "Accuracy", totacc / div, file=sys.stderr)
            model.save_weights("./TrainedModels/epoch_%s_%s_weights" %
                                    (epoch, runname))


    def load(self, json_file, model_path):
        """
        Load in model data.

        Args:
            json_file: the filename of the .json file
            model_path: the path to the model weight files (e.g. path/to/epoch_$EPOCHNUM_$RUNNAME_weights) (without suffixes)
        """
        with open(json_file,"r",encoding="utf-8") as jf:
            jo=json.load(jf)
            
        print("Loading model...")
        self.lablist=jo["lablist"]
        self.fzr=Featurizer(None,jo["fzr"])
        self.nilen=jo["nilen"]
        self.labdict = {self.lablist[i]: i for i in range(len(self.lablist))}
        print("Auxillary information read at", datetime.now(), file=sys.stderr)
        
        self.build_model()
        print("Model initialised at ",datetime.now(),file=sys.stderr)
        self.model.load_weights(model_path)
        print("Model weight loaded at ",datetime.now(),file=sys.stderr)
        print("Model established.")

    def score_to_ent(self,seq,threshold=0.5):
        l=seq['tagfeat'].shape[0]
        ents={}
        for i in range(l):
            # End token
            for j in range(i, l):
                # S is special
                if i == j:
                    if seq['tagfeat'][i,self.labdict['S']] > threshold:
                        # sobie.append(self.labdict['S'])
                        ents[(seq["tokstart"][i], seq["tokend"][j])]=seq['str'][seq["tokstart"][i]:seq["tokend"][j]]
                else:
                    try:
                        # Score for B, then for some number of I, then for E
                        pseq = [seq['tagfeat'][i,self.labdict['B']]] + [k[self.labdict['I']] for k in seq['tagfeat'][i+1:j,:]] + [seq['tagfeat'][j,self.labdict['E']]]
                    except:
                        print(seq['tagfeat'].shape[0], i, j)
                        raise Exception
                    
                    if min(pseq) > threshold:
                        ents[(seq["tokstart"][i], seq["tokend"][j])]=seq['str'][seq["tokstart"][i]:seq["tokend"][j]]
                    
                    if seq["tagfeat"][j,self.labdict['I']] <= threshold: break
        return ents

    def process(self, text,threshold=0.5):
        """
        Find named entities in a string.

        Entities are returned as list of tuples:
        (start_character_position, end_character_position, string)

        Args:
            text: the string to find entities in.
        """

        results=[]
        if len(text) == 0:
            return results
        seq=self._str_to_seq(text)
        if len(seq["tokens"]) == 0:
            return results
        self._prepare_seqs([seq],False,save_path=None)
        tx_id=np.array([seq["wordn"]])
        attention_mask=tf.convert_to_tensor(np.ones(tx_id.shape,dtype='int32'))
        tx2=np.array([seq['ni']])
        
        outputs=self.model.predict([tx_id,attention_mask,tx2])[0]
        seq["tagfeat"]=outputs[1:-1]
        
        
        loc_word_dict = self.score_to_ent(seq,threshold=threshold)
                
        results=post_process(loc_word_dict,text)
        return results

    def batchprocess(self, instrs, threshold=0.5):
        """
        Find named entities in a set of strings. This is potentially faster as neural network calculations
        run faster in batches.

        Entities are returned as tuples:
        (start_character_position, end_character_position, string)

        Args:
            instrs: list of strings to find entities in.
            threshold: the minimum score for entities.
        """
        
        batch_results=[None]*len(instrs)
        train_l_d={}
        seq_index=-1
        
        print("Start processing {} input strings...".format(str(len(instrs))))
        time_initial_all=time.time()
        time_initial = time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        for text in instrs:
            formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
            
            seq_index+=1
            print('\rPreparing sequences {}/{} (ETA {})...'.format(str(seq_index+1),str(len(instrs)),formatted_time), end='', flush=True)
            seq=self._str_to_seq(text)
            seq['idx']=seq_index
            self._prepare_seqs([seq],False,save_path=None)
            token_len=len(seq["ni"])
            if len(text)==0 or token_len==0:
                batch_results[seq_index]=[]
                continue
            
            try:
                train_l_d[token_len].append(seq)
            except KeyError:
                train_l_d[token_len]=[seq]
                
            estimate_time = int((time.time()-time_initial) *
                            (len(instrs)/(seq_index+1)-1))
            est_hr = int(estimate_time/3600)
            est_min = int((estimate_time % 3600)/60)
            est_sec = int(estimate_time % 60)

        batch_counter=0
        time_initial=time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        for batch in train_l_d:
            formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
            print('\rGenerating predictions {}/{} (ETA {})...'.format(str(batch_counter),str(len(instrs)),formatted_time), end='', flush=True)
            seqs=train_l_d[batch]
            tx_id=np.array([seq["wordn"] for seq in seqs])
            attention_mask=tf.convert_to_tensor(np.ones(tx_id.shape,dtype='int32'))
            tx2=np.array([seq['ni'] for seq in seqs])
        
            outputs=self.model.predict([tx_id,attention_mask,tx2])
            for i,seq in enumerate(seqs):
                seq["tagfeat"]=outputs[i][1:-1]
        
                loc_word_dict = self.score_to_ent(seq,threshold=threshold)
        
                batch_results[seq['idx']]=post_process(loc_word_dict,seq['str'])
                
            batch_counter+=len(seq)
            estimate_time = int((time.time()-time_initial) *
                            (len(instrs)/(batch_counter)-1))
            est_hr = int(estimate_time/3600)
            est_min = int((estimate_time % 3600)/60)
            est_sec = int(estimate_time % 60)
            
        estimate_time = int((time.time()-time_initial_all))
        est_hr = int(estimate_time/3600)
        est_min = int((estimate_time % 3600)/60)
        est_sec = int(estimate_time % 60)
        formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
        print()
        print('Done. Finished in {}.'.format(formatted_time))
        return batch_results

        
        
    def process_abbrev(self, text,pmc_id,abbrev_folder,threshold=0.5):
        """
        Find metabolite abbreviations.

        Entities are returned as list of tuples:
        (start_character_position, end_character_position, string)

        Args:
            text: the string to find entities in.
            pmc_id: the source of text (since abbrev files are article dependent)
            abbrev_folder: the folder containing abbrev files produced by AutoCORPus
        """

        def get_abbrev_set(pmc_id,abbrev_folder,threshold):
            abbrev_set=set()
            abbrev_path = os.path.join(abbrev_folder,pmc_id+'_abbreviations.json')
            try:
                with open(abbrev_path,'r',encoding='utf-8-sig') as abbrev_file:
                    abbrev_dict=json.load(abbrev_file)
            except FileNotFoundError:
                print("{} doesn't exist, returning empty set...".format(abbrev_path))
                return abbrev_set

            for abbrev_type in abbrev_dict:
                for abbrev in abbrev_dict[abbrev_type]:
                    entity=abbrev_dict[abbrev_type][abbrev]
                    output=self.process(entity,threshold=threshold)
                    if not output:
                        continue
                    if output[0][2]==entity:
                        abbrev_set.add(abbrev)
            return abbrev_set
                    
                    
        abbrev_set = get_abbrev_set(pmc_id,abbrev_folder,threshold)
        abbrev_res=[]
        for abbrev in abbrev_set:
            abbrev_iter=re.finditer(r'\b{}\b'.format(abbrev),text)
            for abb in abbrev_iter:
                pos=abb.span()
                abbrev_res+=[(pos[0],pos[1],abbrev)]
                
        abbrev_res.sort(key=lambda x:x[0])
        return abbrev_res