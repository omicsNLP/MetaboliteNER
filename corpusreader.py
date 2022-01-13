import time
from datetime import datetime

from transformers import BertTokenizer
import numpy as np
import chemtok


import pandas as pd


class CorpusReader(object):
    """
    Reads, tokenises and locates entities in training data. Splits the
    data into two parts with ratio training:test = 0.85:0.15.

    Members:
    trainseqs - train sequences
    testseqs - test sequences

    Each seq object is a dictionary, containing:
    "tokens": the tokens, as a list of strings
    "tokstart": the start offsets for each token, as a list
    "tokend": the end offsets for each token, as a list
    "bio": BIOES tags
    "ss": the string for the sequence 
    "chemtok_tokens": s tokenized by chemtok, as a list of str
    "chemtok_rep": number of tokens that a chemtok token splits into by Transformer, as a list of int
    "ents": a list of tuples, one per entity, corresponding to the five fields in the annotations file eg:
            PMC2267737	R06002	137	147	creatinine
            PMCID, SentenceCode, StartOffset, EndOffset, String
    """

    def __init__(self, text_file, annot_file, max_seq_len=256, seed=42, 
                 bert_pretrain_path=#"path/to/biobert")
                "bert-base-cased"):
        print("Reading corpus at", datetime.now())
        self.aggressive = False
        self.charbychar = False
        self.alle = True  # convert all entity types to "E"
        self.tosobie = True

        self.trainseqs = {}
        self.testseqs = {}
        
        self.max_seq_len=max_seq_len

        np.random.seed(seed)

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_pretrain_path, do_lower_case=False)

        corpus_df = pd.read_csv(text_file, sep='\t', names=[
            'corpus', 'section', 'text'], encoding="utf-8-sig")
        annot_df = pd.read_csv(annot_file, encoding="utf-8-sig", sep='\t',
                               names=['corpus', 'section',  'start', 'end', 'metabolite'])

        print('Splitting dataset into train:test=0.85:0.15 with random seed {}...'.format(seed))
        corpus_list = list(
            pd.Series(list(set(corpus_df.corpus))).sample(frac=0.85))
        train_df = corpus_df[corpus_df.corpus.apply(lambda s:s in corpus_list)]

        test_df = corpus_df.drop(train_df.index)

        print('Split into {}:{} sentences'.format(
            train_df.shape[0], test_df.shape[0]))

        print('Processing train data...')
        timer = time.time()
        self.trainseqs = self.to_bioes(train_df, annot_df)
        pd.DataFrame(self.trainseqs).to_csv('TrainSeqs.txt', header=False,
                                            encoding='utf_8_sig', index=False, sep='\t')
        print('Elapsed in {}'.format(str(time.time()-timer)))

        print('Processing test data...')
        timer = time.time()
        self.testseqs = self.to_bioes(test_df, annot_df)
        pd.DataFrame(self.testseqs).to_csv('TestSeqs.txt', header=False,
                                           encoding='utf_8_sig', index=False, sep='\t')
        print('Elapsed in {}'.format(str(time.time()-timer)))

    def str_to_seq(self, s):
        '''
        Computes truncated sequence data of input s
        
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
        # Generate sequence of str. Used for input of BIOES
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

    def _pos_to_bioes(self, pos_list, b):
        '''
        Determines the BIOES tag of a token base
        
        Args:
                pos_list: a list of positions of (untokenised) annotated entities
                b: position of the token that awaits BIOES assignment
        
        Returns:
                The BIOES tag of the token at position b. This is a string with following possible values: 
                    # B:=Beginning of an entity
                    # I:=Inside entity
                    # O:=Not part of entity
                    # E:=End of entity
                    # S:=Singleton
        '''
        # Assumed that domains defined by pairs in 'a' are mutually exclusive
        for a in pos_list:
            if b[0] >= a[0] and b[1] <= a[1]:
                # is nested
                if b[0] == a[0] and b[1] == a[1]:
                    return 'S'
                if b[0] == a[0] and b[1] != a[1]:
                    return 'B'
                if b[0] != a[0] and b[1] == a[1]:
                    return 'E'
                return 'I'

            if b[0] <= a[0] and b[1] >= a[1]:  # b contains a
                return 'S'
            if b[0] >= a[0] and b[0] <= a[1]:  # b start at middle of a
                return 'E'
            if b[1] >= a[0] and b[1] <= a[1]:  # b end at middle of a
                return 'B'

        return 'O'

    def revise(self, bioes_seq):
        '''
            A quick sanity check for the BIOES assignment
            
            Args:
                bioes_seq: a list of BIOES tags
                
            Returns:
                Updated bioes_seq
        '''
        # S cannot be followed by E (which happens e.g. "glucose, mannose")
        for i in range(1, len(bioes_seq)):
            if bioes_seq[i] == 'E':
                if bioes_seq[i-1] not in ['B', 'I']:
                    bioes_seq[i] = 'O'#? too strict?
        return bioes_seq

    def to_bioes(self, text_df, _annot_df):
        '''
            Computes the bioes and other auxillary information of the given texts and annotations
            
            Args:
                text_df: a dataframe that contains textual information of the corpus, consists of fields 'corpus', 'section' and 'text'
                _annot_df: a dataframe that contains all annotations of texts in text_df
                
            Returns:
                a list of seq (the dictionary as specified in str_to_seq), each seq corresponds to the processed result of a sentence in text_df
        '''
        seqs = []
        for i in text_df.index:  # loop over sentences
            seq_dict = {}
            corpus_id = text_df.loc[i, 'corpus']
            section_id = text_df.loc[i, 'section']
            text = text_df.loc[i, 'text']
            seq_dict['ss'] = text

            # Zoom in df
            annot_df_sub = _annot_df[_annot_df.corpus == corpus_id]
            annot_df_sub = annot_df_sub[annot_df_sub.section == section_id]

            start_list = list(annot_df_sub.start)
            end_list = list(annot_df_sub.end)

            seq_dict['ents'] = [(corpus_id, section_id, start_list[j], end_list[j],
                                 text[start_list[j]:end_list[j]]) for j in range(len(start_list))]


            pos_list = list(zip(start_list, end_list))

            # Tokenize
            seq = self.str_to_seq(text)
            tok_pos = list(zip(seq['tokstart'], seq['tokend']))
            seq_dict['tokens'] = seq['tokens']
            seq_dict['tokstart'] = seq['tokstart']
            seq_dict['tokend'] = seq['tokend']
            seq_dict["chemtok_tokens"] = seq["chemtok_tokens"]
            seq_dict["chemtok_rep"] = seq["chemtok_rep"]

            bioes_seq = self.revise(
                [self._pos_to_bioes(pos_list, b) for b in tok_pos])
            seq_dict['bio'] = bioes_seq

            seqs.append(seq_dict)

        return seqs

