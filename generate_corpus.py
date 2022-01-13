import json
import pandas as pd
import re
import spacy
import os
import time
import argparse
from utils import *

# Automatic rule-based training set generation

def segment_word(s):
    '''
    Segment string s into nested lists of words. Words in string s are broken into sublists by spaCy-defined stop words.
    '''

    if s[-1] in [',', '.', '!', '?', ';', ':']:
        s = s[:-1]

    s_list = re.split(r'[,\.\!\?;:]? ', s)
    s_stopword = [i.lower() in nlp.Defaults.stop_words for i in s_list]

    sep_word_list = []
    _nested_sep_word_list = []
    for i in range(len(s_list)):
        if not s_stopword[i]:
            _nested_sep_word_list.append(s_list[i])
        else:
            if _nested_sep_word_list:
                sep_word_list.append(_nested_sep_word_list)
            _nested_sep_word_list = []
    if _nested_sep_word_list:
        sep_word_list.append(_nested_sep_word_list)

    return sep_word_list


def generate_n_gram(word_list, n=4):
    '''
    Generate n-grams (n<=4)
    
    Note: Since only 219 entities (219/144865=0.15% of all HMDB) are space-separated into more than 5 words (and are discardable by manual inspection), such HMDB items are disregarded. Thence only <=4-grams should be considered in dictionary checkings here.
    '''
    n_gram_list = []
    if not word_list:
        return n_gram_list
    n = min(len(word_list), n)
    # 1-gram
    i_gram = word_list  # i-gram where i=1
    n_gram_list += i_gram
    for i in range(1, n):
        i_gram = [' '.join(gram) for gram in zip(i_gram[:-1], word_list[i:])]
        n_gram_list += i_gram

    return n_gram_list



def metabolite_hmdb(doc_sent, metabolite_names):
    '''
    Identify the HMDB metabolite present in doc_sent
    '''
    seg_sents = segment_word(doc_sent)
    n_gram = []
    for seg_sent in seg_sents:
        n_gram += generate_n_gram(seg_sent)

    n_gram = set(n_gram)  # deduplicate
    n_gram = [fix_parenthesis(i) for i in n_gram]
    n_gram = [i for i in n_gram if i]
    n_gram_lower = [i.lower() for i in n_gram]  # To lowercase
    metabolite_set = set(
        metabolite_names[metabolite_names.name.isin(n_gram_lower)].name)
    if metabolite_set:
        metabolite_set = set(
            [g for g in n_gram if g.lower() in metabolite_set])
    return metabolite_set


def metabolite_re(doc_sent, regex_list):
    seg_sents = segment_word(doc_sent)
    metabolite_set = set()
    for seg_sent in seg_sents:
        for w in seg_sent:
            for regex in regex_list:
                if re.search(regex, w, flags=re.I):
                    metabolite_set.add(w)
    return metabolite_set


def locate_word(doc_sent, word_list):
    '''
    Return the start/end index of the words in word_list in doc_sent
    
    Args:
        doc_sent: sentence
        word_list: a list of entities to be found in doc_sent
        
    Returns:
        A dictionary of the form {(start_idx,end_idx):entity}
    '''
    loc_word_dict = {}
    doc_sent_vanilla = doc_sent
    # doc_sent = doc_sent.lower()
    for w in word_list:
        if w in doc_sent:
            start_idx = 0
            while True:
                start_idx = doc_sent.find(w, start_idx)
                if start_idx == -1:
                    break
                end_idx = start_idx+len(w)
                doc_sent_vanilla[start_idx:end_idx]

                bool_seg = (start_idx == 0 or doc_sent_vanilla[start_idx-1] == ' ') and (end_idx+1 >= len(
                    doc_sent) or doc_sent_vanilla[end_idx] in [',', '.', '!', '?', ';', ':', ' '])

                if bool_seg:
                    loc_word_dict[(start_idx, end_idx)
                                  ] = doc_sent_vanilla[start_idx:end_idx]
                start_idx += len(w)
    return loc_word_dict


def cure_loc_dict(doc_sent, loc_word_dict, exclude_list_re, metabolite_set_hmdb):
    '''Remove items that satisfy re in exclude_list_re and not in hmdb'''
    pos_set = set(loc_word_dict.keys())
    for pos in pos_set:
        del_flag = False
        word = loc_word_dict[pos]
        if word not in metabolite_set_hmdb and not del_flag:
            if len(word) <= 3:
                del loc_word_dict[pos]
                del_flag = True
                continue
            for regex in exclude_list_re:
                if re.search(regex, word, re.I):
                    del loc_word_dict[pos]
                    del_flag = True
                    break

            # Exclude e.g.SRM1950 or S2 or mTORC1
            if re.search('^[a-z]*([A-Z]|[0-9])+( ?.*\))$', word) and not re.search('\-', word) and not del_flag and not re.search('\(|\)|:|,', word):
                if pos in loc_word_dict:
                    del loc_word_dict[pos]
                    del_flag = True
                continue
        # A few more filtering gates (to be optimised)
        if not del_flag:
            exclude_entity = ['^{}(s?)$'.format(entity)
                               for entity in ['alcohol', 'transit', 'preparing', 'overload', 'sheet', 'result', 'PC', 'PE', 'medicine', 'vortex(e?)', 'sherlock', 'isomer', 'acid', 'ester', 'ether', 'oate', 'xylate(d?)', 'adrenal', 'gas(e?)', 'aldehyde']]
            exclude_end = ['{}(s?)$'.format(unwant_end)
                           for unwant_end in ['ics', 'ia', 'yl']]

            exclude_entity += ['^{} (acid|ester|ether)(s?)$'.format(head) for head in ['urofuran', 'κeto', 'mineral', 'halogenated', 'gastric-tract', 'keto', 'included', 'palmi', 'representative', 'fat', 'to', 'terminal', 'total', 'aqueous', '\(caprylic\)', 'nuclei', '-hphaa\)\)', 'acetoace', 'without', 'these', 'aspartate', 'for', 'small', 'various', 'model(l?)ed', '\(dpa\)','from', 'amine', 'maximum', 'stomach', 'different', 'nuclear', 'tract', 'after', '\(dha\)', 'polyunsaturated', 'lewis', 'standard', 'sugar', 'conjugate', 'enhanced', 'tagged', 'other', 'osbond', 'others', 'biliary', '\(gca\)', 'cycle', 'aa', 'a-keto', 'volatile', 'prevent', '-bile', 'mild', 'lignocericc', 'bile', '\(c12\)', 'kyn', 'fractional']]
            exclude_entity += ['{} (acid|ester|ether)(s?)$'.format(head)
                                for head in ['ed', '[^i]al']]
            for regex in exclude_entity+exclude_end:
                if re.search(regex, word, re.I):
                    del loc_word_dict[pos]
                    break

            # Check if ic$ followed by acid
            if pos not in loc_word_dict:
                continue
            if re.search('ic$', word):
                pos_acid = doc_sent.find('acid', pos[1])
                pos_sent_end = -1
                for p in ['.', ';', '!', '?']:
                    p_pos = doc_sent.find(p, pos[1])
                    if p_pos != -1:
                        if pos_sent_end == -1:
                            pos_sent_end = p_pos
                        else:
                            pos_sent_end = min(pos_sent_end, p_pos)

                if pos_sent_end == -1:
                    del loc_word_dict[pos]
                elif pos_sent_end < pos_acid:
                    del loc_word_dict[pos]

    pos_set = set(loc_word_dict.keys())
    for pos in pos_set:
        if loc_word_dict[pos][-1] in [',', '.', '?', ':', ';', '!']:
            new_word = loc_word_dict[pos][:-1]
            new_pos = (pos[0], pos[1]-1)
            del loc_word_dict[pos]
            loc_word_dict[new_pos] = new_word

    pos_set = set(loc_word_dict.keys())
    for pos in pos_set:
        if loc_word_dict[pos][-1] in [',', '.', '?', ':', ';', '!', ' ']:
            new_word = loc_word_dict[pos][:-1]
            new_pos = (pos[0], pos[1]-1)
            del loc_word_dict[pos]
            loc_word_dict[new_pos] = new_word

    pos_set = set(loc_word_dict.keys())
    for pos in pos_set:
        cur_metabolite = loc_word_dict[pos]
        if re.search(r'v v\)$', cur_metabolite):
            new_pos_end = cur_metabolite.rfind('(')
            if new_pos_end == -1:
                del loc_word_dict[pos]
            else:
                if cur_metabolite[new_pos_end-1] == ' ':
                    new_pos_end -= 1
                new_word = cur_metabolite[:new_pos_end]
                new_pos = (pos[0], new_pos_end)
                del loc_word_dict[pos]
                loc_word_dict[new_pos] = new_word

    return loc_word_dict


def cure_sent(doc_sent):
    # Aim to pre-process the input sentence to achieve curation.
    res = doc_sent
    target_list = re.findall('[a-z]/[a-z]', doc_sent, re.I)
    target_list += re.findall('[a-z]:[a-z]', doc_sent, re.I)
    for i in target_list:
        res = res.replace(i, i[0]+' '+i[2])
    return res

def format_annot_output(loc_word_dict, corpus_id, sec_id, subsec_id, sent_id):
    output_dict = {'corpus': [], 'section': [], 'subsection': [
    ], 'start': [], 'end': [], 'metabolite': []}

    for pos in loc_word_dict:
        subsec_str = str(subsec_id).zfill(2)+str(sent_id).zfill(3)
        output_dict['corpus'].append(corpus_id)
        output_dict['section'].append(sec_id)
        output_dict['subsection'].append(subsec_str)
        output_dict['start'].append(pos[0])
        output_dict['end'].append(pos[1])
        output_dict['metabolite'].append(loc_word_dict[pos])

    df_output = pd.DataFrame(output_dict)
    df_output.section += df_output.subsection.apply(lambda s: str(s).zfill(5))
    df_output.drop('subsection', axis=1, inplace=True)
    return df_output


def format_corpus_output(corpus_id, sec_id, subsec_id, sent_id, textbody):
    '''
    Format the recognised sentences into desired dataframe output
    
    Args:
        corpus_id: PMC id
        sec_id: Section id (e.g. 'M' for Method section)
        subsec_id: Subsection id (e.g. 0 for the 1st subsection of sec_id). Presuming <100 subsections
        sent_id: Sentence id (e.g. 0 for the 1st sentence that contains an recognised metabolite)
        textbody: Textual content
    '''
    output_dict = {'corpus': [], 'section': [], 'body': []}

    subsec_str = str(subsec_id).zfill(2)+str(sent_id).zfill(3)
    output_dict['corpus'].append(corpus_id)
    output_dict['section'].append(sec_id+subsec_str)
    output_dict['body'].append(textbody)
    return pd.DataFrame(output_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-b', '--base_dir', type=str,
                        help='directory of pmc json files')
    parser.add_argument('-t', '--target_dir', type=str)
    parser.add_argument('-m', '--metabolite_file', type=str,
                        help='filepath to (HMDB) metabolite names',default='MetaboliteNames.txt')
    parser.add_argument('-n', '--output_file_name', type=str,
                        help='Name of output file (TrainingSet by default). Text (Name.txt) and annotation (NameAnnot.tsv) will be created. ')
    parser.add_argument('-r','--regex_list',type=str,help='filepath to a list of regular expressions for entity recognition',default='RegexList_RuleBasedAnnotation.txt')
    parser.add_argument('-e','--exclusion_list',type=str,help='filepath to a list of entities and regular expressions to be excluded',default='ExclusionList_RuleBasedAnnotation.txt')
    

    args = parser.parse_args()
    json_dir = args.base_dir

    save_dir = args.target_dir
    metabolite_names_dir = args.metabolite_file

    output_tsv = save_dir+'/{}Annot.tsv'.format(args.output_file_name)
    output_txt = save_dir+'/{}.txt'.format(args.output_file_name)
    
    regex_list_dir=args.regex_list
    exclude_list_dir=args.exclusion_list

    df_output = pd.DataFrame()
    df_text_output = pd.DataFrame()
    json_file_list = os.listdir(json_dir)

    # Read metabolite names
    metabolite_names = pd.read_csv(
        metabolite_names_dir, delimiter='\t', keep_default_na=False)

    metabolite_names_lower = metabolite_names.assign(
        name=metabolite_names.name.apply(lambda s: s.lower()))

    # Read regex for inclusion/exclusion. Assuming all entities are \t-separated
    with open(regex_list_dir,'r') as f:
        regex_list = f.read().split('\t')
    
    if exclude_list_dir:
        with open(exclude_list_dir,'r') as f:
            exclude_list_re = f.read().split('\t')
    else:
        exclude_list_re=[]

    try:
        element_list = list(pd.read_csv('elements.txt', header=None).loc[:, 0])
        for i in element_list:
            exclude_list_re.append('^{}$'.format(i))
    except FileNotFoundError:
        element_list = []
        print('Element list not found. Proceed without excluding elements.')

    # load spacy
    nlp = spacy.load("en_core_web_sm")

    text_type = {'M': 'methods section',
                 'R': 'results section', 'D': 'discussion section', 'A': "textual abstract section"}

    df_annot = pd.DataFrame()
    df_corpus = pd.DataFrame()

    time_initial = time.time()
    file_counter = 0

    est_hr, est_min, est_sec = 0, 0, 0

    for json_file in json_file_list:
        file_counter += 1
        json_file_dir = '/'.join([json_dir, json_file])
        corpus_id = json_file.split('_')[0]
        time_corpus = time.time()
        formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)

        print('\r'+'Processing {} ({}/{})...Estimated remaining time {}'.format(corpus_id,
              str(file_counter), str(len(json_file_list)), formatted_time), end='', flush=True)

        with open(json_file_dir, 'r', encoding='utf-8') as f:
            maintext = json.load(f)
        maintext = maintext['paragraphs']

        for tt_abbrev in text_type:
            tt = text_type[tt_abbrev]
            text_body = []
            _tb_idx = 0
            text_body_idx = []

            for mt in maintext:
                if tt in mt['IAO_term'] and 'checked' not in mt['IAO_term']:
                    mt['IAO_term'].append('checked')
                    text_body.append(mt['body'])
                    text_body_idx.append(_tb_idx)
                    _tb_idx += 1

            text_body = list(zip(text_body_idx, text_body))

            for (tb_idx, tb) in text_body:
                # Fix text body (subject to AutoCORPus updates)
                tb = tb.replace('\n', ' ')
                tb = tb.replace(' []', '')
                tb = tb.replace(' ()', '')
                tb = tb.replace('  ', ' ')

                doc = nlp(tb)

                doc_sents = [str(i) for i in doc.sents if str(i) != ' ']
                sent_id = -1
                for doc_sent in doc_sents:
                    sent_id += 1

                    metabolite_set_hmdb = metabolite_hmdb(
                        doc_sent, metabolite_names=metabolite_names_lower)
                    metabolite_set = metabolite_set_hmdb | metabolite_re(
                        doc_sent, regex_list)

                    if metabolite_set:
                        doc_sent_cured = cure_sent(doc_sent)
                        loc_word_dict = locate_word(
                            doc_sent_cured, metabolite_set)
                        loc_word_dict = post_process(loc_word_dict,doc_sent_cured,return_dict=True)
                        loc_word_dict = cure_loc_dict(doc_sent,
                                                      loc_word_dict, exclude_list_re, metabolite_set_hmdb, element_list)
                        if loc_word_dict:
                            df_annot = df_annot.append(format_annot_output(
                                loc_word_dict, corpus_id, tt_abbrev, tb_idx, sent_id), ignore_index=True)
                            df_corpus = df_corpus.append(format_corpus_output(
                                corpus_id, tt_abbrev, tb_idx, sent_id, doc_sent), ignore_index=True)
        estimate_time = int((time.time()-time_initial) *
                            (len(json_file_list)/file_counter-1))
        est_hr = int(estimate_time/3600)
        est_min = int((estimate_time % 3600)/60)
        est_sec = int(estimate_time % 60)
    df_annot.sort_values(['corpus', 'section', 'start'],
                         ignore_index=True, inplace=True)
    df_annot = df_annot.astype(
        {'corpus': str, 'section': str, 'start': int, 'end': int, 'metabolite': str})

    df_corpus.sort_values(['corpus', 'section'],
                          ignore_index=True, inplace=True)
    df_corpus = df_corpus.astype(
        {'corpus': str, 'section': str, 'body': str})

    df_annot.to_csv(output_tsv, encoding='utf_8_sig',
                    sep='\t', header=False, index=False)

    df_corpus.to_csv(output_txt, encoding='utf_8_sig',
                     sep='\t', header=False, index=False)
    print('Done! {} files processed in {} seconds.'.format(
        file_counter, time.time()-time_initial))
