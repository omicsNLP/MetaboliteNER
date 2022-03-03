import re

stop_words=[]
stop_word_path='StopWords.txt'
try:
    with open('StopWords.txt','r') as f:
        stop_words=f.read().split('\n')
except FileNotFoundError:
    'Stop words specification missing. Proceding without...'

def tobits(bit, bits):
	"""Make a one-high encoding.
	
	Args:
		bit: which bit should be high
		bits: how many bits there are
	
	Returns:
		List of 0 and 1 values
	"""
	
	return([1 if i == bit else 0 for i in range(bits)])

def sobie_scores_to_char_ents(seq, threshold, ss):
	"""Find all possible entities in a sequence of SOBIE tag score distributions, where the minimum score in the
	entity is greater or equal than the threshold.
	
	Args:
		seq: Dictionary. seq["tokens"] = sequence of token strings. seq["tagfeat"] = 2D array, dim 1=position, dim 2=
		S, O, I, E, B (reverse alphabetical)
		threshold: the lowest permissible value in seq["tagfeat"] for the relevant tag/token combos.
		ss: "sentence string" - the string form of the sentence.
		
	Returns:
		tuple: (
			List of entities - each entity is a tuple, (entity type, start character position, end character position),
			Dictionary of xents - from entity tuple to dictionary of additional values:
				ent - the entity
				pseq - the sequence of scores for the relevant tags in the entity
				oseq - as pseq, but for "O" in each position
				score - the minimum of pseq
				str - the string for the entity
				dom - whether the entity is "dominant" - i.e. not overlapping with a higher-scoring entity
		)
	"""
	
	l = len(seq["tokens"])
	ents = []
	xents = {}
	# Start token
	for i in range(l):
		# End token
		for j in range(i, l):
			# S is special
			if i == j:
				if seq["tagfeat"][i][0] > threshold:
					ent = ("E", seq["tokstart"][i], seq["tokend"][j])
					ents.append(ent)
					xe = {}
					xe["ent"] = ent
					xe["pseq"] = [seq["tagfeat"][i][0]] # Score for S
					xe["oseq"] = [seq["tagfeat"][i][1]] # Score for O
					xe["score"] = xe["pseq"][0] # Score for S
					xe["str"] = ss[ent[1]:ent[2]]
					xents[ent] = xe
			else:
				try:
					# Score for B, then for some number of I, then for E
					pseq = [seq["tagfeat"][i][4]] + [k[2] for k in seq["tagfeat"][i+1:j]] + [seq["tagfeat"][j][3]]
					# Score for some number of O
					oseq = [k[1] for k in seq["tagfeat"][i:j+1]]
				except:
					print(len(seq["tagfeat"]), i, j)
					raise Exception
				if min(pseq) > threshold:
					ent = ("E", seq["tokstart"][i], seq["tokend"][j])
					ents.append(ent)
					xe = {}
					xe["pseq"] = pseq
					xe["oseq"] = oseq
					xe["score"] = min(pseq)
					xe["str"] = ss[ent[1]:ent[2]]
					xents[ent] = xe
				# Check: if the score for I is below threshold, then all longer entities starting at this
				# position will be below threshold, so stop looking.
				if seq["tagfeat"][j][2] <= threshold: break
	# OK, now we have the entities, mark them dominance. Start with the best scoring entities...
	se = sorted(ents, key=lambda x:-xents[x]["score"])
	# Make a list of which character positions contain dominant entities - none yet
	uu = [False for i in range(len(ss))]
	# For each entity
	for e in se:
		# Dominant unless proved otherwise
		dom = True
		# Are the characters taken?
		for i in range(e[1],e[2]):
			if uu[i]:
				dom = False
				break
		xents[e]["dom"] = dom
		# If dominant, mark those characters as taken
		if dom:
			for i in range(e[1], e[2]): uu[i] = True
	return ents, xents

def get_prev_word(doc_sent, start_idx):
    sent_head = doc_sent[:start_idx]
    prev_end_idx = sent_head.rfind(' ')
    if prev_end_idx <= 0:
        return None
    prev_start_idx = doc_sent[:prev_end_idx].rfind(' ')+1

    if sent_head[prev_end_idx-1] in [',', '.', '!', '?', ';', ':', ' ']:
        prev_end_idx -= 1
    return prev_start_idx, prev_end_idx, doc_sent[prev_start_idx:prev_end_idx]


def get_next_word(doc_sent, end_idx):
    tail_cursor_idx = end_idx
    sent_tail = doc_sent[end_idx:]
    next_start_idx = sent_tail.find(' ')+1
    if next_start_idx == 0 or next_start_idx == len(sent_tail)-1:
        return None

    next_end_idx = sent_tail[next_start_idx:].find(' ')
    if next_end_idx == -1:
        next_end_idx = len(sent_tail[next_start_idx:])
    if sent_tail[next_end_idx] in [',', '.', '!', '?', ';', ':', ' ']:
        next_end_idx -= 1

    next_end_idx += next_start_idx

    next_start_idx += tail_cursor_idx
    next_end_idx += tail_cursor_idx

    if next_start_idx == next_end_idx:
        next_word = doc_sent[next_start_idx]
    else:
        next_word = doc_sent[next_start_idx:next_end_idx]
    return next_start_idx, next_end_idx, next_word




def check_neighbor(doc_sent, loc_word_dict):
    '''
    Checks neighbouring words of each metabolite name
    '''
    # if distance between word >1 and is not '- '
    target_set = loc_word_dict.values()
    pos_set = set(loc_word_dict.keys())

    # Special cases
    prev_special = ['glacial', 'cyclic', 'hydrate', 'amino']
    next_special = ['acid', 'isomer', 'ether', 'ester']

    def special_check(s, regex): return any(
        bool(re.search(reg, s, re.I)) for reg in regex)

    for pos in pos_set:
        if pos not in loc_word_dict.keys():
            continue
        prev_flag = True
        next_flag = True
        pos_changed_flag = False

        start_idx = pos[0]
        end_idx = pos[1]

        while prev_flag:
            prev_word = get_prev_word(doc_sent, start_idx)
            if not prev_word:
                prev_flag = False
                continue
            if prev_word[2] == '':
                prev_word = get_prev_word(doc_sent, prev_word[0])
                if not prev_word:
                    prev_flag = False
                    continue
            if prev_word[2]:
                is_colon = bool(
                    re.search('^\(?\d+:(0\D|1|2|3|4|5|6|7|8|9)', doc_sent[start_idx:end_idx], re.I)) and (prev_word[2].lower() not in stop_words)
                is_acid = 'acid' == doc_sent[start_idx:start_idx+len(
                    'acid')] or 'ester' == doc_sent[start_idx:start_idx+len('ester')]
                is_start_hyphen = doc_sent[start_idx] == '-' or prev_word[2][-1] == '-' or doc_sent[start_idx] == ','
                is_start_number = bool(re.search(
                    '^[1-9]', doc_sent[start_idx:end_idx]) and re.search(
                    '[0-9]$', prev_word[2]))
                if prev_word[2] in target_set or special_check(prev_word[2], prev_special) or is_colon or is_acid or is_start_hyphen or is_start_number:
                    # and prev_word[2][-1]!='-' and pos[2][0]!='-'):
                    if 'acid' in prev_word[2]:
                        prev_flag = False
                    elif (start_idx-prev_word[1] <= 1) or is_start_number:
                        if (prev_word[0], prev_word[1]) in loc_word_dict:
                            del loc_word_dict[(prev_word[0], prev_word[1])]
                        start_idx = prev_word[0]
                        pos_changed_flag = True
                    else:
                        prev_flag = False
                else:
                    prev_flag = False
            else:
                prev_flag = False

        while next_flag:
            next_word = get_next_word(doc_sent, end_idx)
            current_word = doc_sent[start_idx:end_idx]
            if not current_word:
                pass
            if next_word and current_word:
                if not next_word[2]:
                    next_flag = False
                    break
                is_start_end_num = re.search(
                    '^[^(a-z)]+$', current_word, re.I) or re.search('^\d.*\d$', doc_sent)
                is_end_hyphen = current_word[-1] == '-' or next_word[2][0] == '-'
                is_end_greek = re.search('[α-ω]$', current_word)
                if 'acid' in current_word:
                    next_flag = False
                elif next_word[2] in target_set or special_check(next_word[2], next_special) or is_start_end_num or is_end_hyphen or is_end_greek:
                    if (next_word[0]-end_idx <= 1):
                        if (next_word[0], next_word[1]) in loc_word_dict:
                            del loc_word_dict[(next_word[0], next_word[1])]
                        end_idx = next_word[1]
                        pos_changed_flag = True
                    else:
                        next_flag = False
                else:
                    next_flag = False
            else:
                next_flag = False

        if pos_changed_flag:
            del loc_word_dict[pos]
            loc_word_dict[(start_idx, end_idx)] = doc_sent[start_idx:end_idx]

    return loc_word_dict


def check_parenthesis_balance(s):
    '''
    Check if the parentheses in str s are balanced
    '''
    open_list = ["[", "{", "("]
    close_list = ["]", "}", ")"]
    stack = []
    for i in s:
        if i in open_list:
            stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and
                    (open_list[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                return False
    if len(stack) == 0:
        return True
    else:
        return False


def fix_parenthesis(s):
    '''
    Fixes unbalanced parenthesis in str s
    '''
    open_list = ["[", "{", "("]
    close_list = ["]", "}", ")"]
    balance_bool = check_parenthesis_balance(s)
    if any([i in s for i in open_list+close_list]):
        if balance_bool:
            if (s[0], s[-1]) in zip(open_list, close_list):
                if check_parenthesis_balance(s[1:-1]):
                    return s[1:-1]
                else:
                    return s
            else:
                return s
        else:
            s_cache = s
            if (s_cache[0], s_cache[-1]) in zip(open_list, close_list):
                s_cache_r = s_cache[1:]
                s_cache_l = s_cache[:-1]
                if check_parenthesis_balance(s_cache_r):
                    return s_cache_r
                if check_parenthesis_balance(s_cache_l):
                    return s_cache_l
            if s_cache[0] in open_list:
                s_cache = s_cache[1:]
            if check_parenthesis_balance(s_cache):
                return s_cache
            if s_cache[-1] in close_list:
                s_cache = s_cache[:-1]
            if check_parenthesis_balance(s_cache):
                return s_cache
            else:
                return s
    else:
        return s


def fix_parenthesis_dict(doc_sent, loc_word_dict):
    '''
    Fixes unbalanced parenthesis in the whole dict
    '''

    open_list = ["[", "{", "("]
    close_list = ["]", "}", ")"]

    pos_set = set(loc_word_dict.keys())

    for pos in pos_set:
        if pos not in loc_word_dict.keys():
            continue
        word = loc_word_dict[pos]
        cured_word = fix_parenthesis(loc_word_dict[pos])
        if word != cured_word:
            start_idx = pos[0]+word.find(cured_word)
            end_idx = start_idx + len(cured_word)
            del loc_word_dict[pos]
            loc_word_dict[(start_idx, end_idx)] = cured_word

    # Cure unbalanced bracket in the middle
    def check_parenthesis_stack(s):
        open_list = ["[", "{", "("]
        close_list = ["]", "}", ")"]
        stack = []
        for i in s:
            if i in open_list:
                stack.append(i)
            elif i in close_list:
                if len(stack) == 0:
                    stack.append(i)
                elif stack[-1] == open_list[close_list.index(i)]:
                    stack.pop()
                else:
                    stack.append(i)
        return stack
    pos_set = set(loc_word_dict.keys())

    for pos in pos_set:
        if pos not in loc_word_dict.keys():
            continue
        word = loc_word_dict[pos]
        stack = check_parenthesis_stack(word)
        start_idx = pos[0]
        end_idx = pos[1]
        while stack:
            i = stack.pop()
            if i in open_list:
                i_close = close_list[open_list.index(i)]
                end_idx += doc_sent[end_idx:].find(i_close)+1
            else:
                i_open = open_list[close_list.index(i)]
                start_idx = doc_sent[:start_idx].rfind(i_open)
        del loc_word_dict[pos]
        if pos[0] == 0 and start_idx == -1:
            start_idx = pos[0]
        if pos[1] == len(doc_sent) and end_idx == -1:
            end_idx = pos[1]
        loc_word_dict[(start_idx, end_idx)] = doc_sent[start_idx:end_idx]

    # Cure by step 1 once more
    pos_set = set(loc_word_dict.keys())

    for pos in pos_set:
        if pos not in loc_word_dict.keys():
            continue
        word = loc_word_dict[pos]
        cured_word = fix_parenthesis(loc_word_dict[pos])
        if word != cured_word:
            start_idx = word.find(cured_word)
            end_idx = len(cured_word)
            del loc_word_dict[pos]
            loc_word_dict[(start_idx, end_idx)] = cured_word
    return loc_word_dict


def merge_overlapped_pos(loc_word_dict, doc_sent):
    '''
    Merge overlapping entities
    '''
    def is_overlapping(a, b): return b[0] >= a[0] and b[0] <= a[1]

    def merge(arr):
        # sort the intervals by its first value
        arr.sort(key=lambda x: x[0])

        merged_list = []
        merged_list.append(arr[0])
        for i in range(1, len(arr)):
            pop_element = merged_list.pop()
            if is_overlapping(pop_element, arr[i]):
                new_element = pop_element[0], max(pop_element[1], arr[i][1])
                merged_list.append(new_element)
            else:
                merged_list.append(pop_element)
                merged_list.append(arr[i])
        return merged_list

    if loc_word_dict:
        pos_list = list(loc_word_dict.keys())
        merged_pos_list = merge(pos_list)
        for p in pos_list:
            if p not in merged_pos_list:
                del loc_word_dict[p]
        for p in merged_pos_list:
            if p not in pos_list:
                loc_word_dict[p] = doc_sent[p[0]:p[1]]
    return loc_word_dict


def merge_close_pos(loc_word_dict, doc_sent):
    '''
    Merge words that are 1-character away
    '''
    c_l = [[i] for i in loc_word_dict]

    def recur_merge(c_l):
        for i in c_l:
            for j in c_l:
                if j[0][0]-i[-1][1] == 1:
                    c_l.remove(i)
                    c_l.remove(j)
                    c_l.append(i+j)
                    return recur_merge(c_l)
        return c_l

    merged_pos_list = recur_merge(c_l)
    for i in merged_pos_list:
        if len(i) > 1:
            loc_word_dict[(i[0][0], i[-1][1])] = doc_sent[i[0][0]:i[-1][1]]
            for p in i:
                del loc_word_dict[p]

    return loc_word_dict


def complete_word(instr, loc_word_dict):
    '''
    Completes (grammarly incomplete) entities by farther fetch
    '''
    open_list = ["[", "{", "("]
    close_list = ["]", "}", ")"]

    for _ in range(3):
        remove_loc = []
        new_loc = []
        for loc in loc_word_dict:
            if loc[0] == -1 or loc[1] == -1:
                continue
            left_space = instr.rfind(' ', 0, loc[0])
            right_space = instr.find(' ', loc[1], len(instr))
            loc_left, loc_right = loc[0], loc[1]
            if left_space != -1:
                loc_left = left_space+1
            else:
                loc_left=0
            if right_space != -1:
                loc_right = right_space
            
            else:
                loc_right = len(instr)
            right_stop = ['.', ',', ':', '!', '?', '&', ';', '=']  # )
            # if loc_right < len(instr):
            if instr[loc_right-1] in right_stop:
                loc_right -= 1
            if instr[loc_left] in right_stop:
                loc_left+=1
            metabolite = instr[loc_left:loc_right]

            if not metabolite:
                remove_loc.append(loc)
            
            elif (metabolite[0], metabolite[-1]) in zip(open_list, close_list):
                if check_parenthesis_balance(metabolite[1:-1]):
                    loc_left += 1
                    loc_right -= 1
                    metabolite = instr[loc_left:loc_right]
            elif ((metabolite[0] in open_list) or (metabolite[-1] in close_list)):
                if metabolite[0] in open_list:
                    if check_parenthesis_balance(metabolite[1:]):
                        loc_left += 1
                elif metabolite[-1] in close_list:
                    if check_parenthesis_balance(metabolite[:-1]):
                        loc_right -= 1

            if loc != (loc_left, loc_right) and metabolite:
                remove_loc.append(loc)
                new_loc.append((loc_left, loc_right))
        for loc in remove_loc:
            del loc_word_dict[loc]
        for loc in new_loc:
            loc_word_dict[loc] = instr[loc[0]:loc[1]]

        # Split by colon (e.g. citrate:borate)
        for splitter in [':', '/']:
            remove_loc = []
            new_loc = []
            for loc in loc_word_dict:
                if re.search('[a-z]+:[a-z]+', loc_word_dict[loc], re.I):
                    remove_loc.append(loc)
                    split_entities = loc_word_dict[loc].split(splitter)
                    split_entities_len = [len(i) for i in split_entities]

                    start = loc[0]
                    end = start+split_entities_len[0]
                    new_loc.append((start, end))
                    for i in range(1, len(split_entities)):
                        start = end+1
                        end = start+split_entities_len[i]
                        new_loc.append((start, end))

            for loc in remove_loc:
                # if loc in loc_word_dict:
                del loc_word_dict[loc]
            for loc in new_loc:
                loc_word_dict[loc] = instr[loc[0]:loc[1]]

    return loc_word_dict


def exclude(loc_word_dict, exclude_list):
    '''
    Remove entities that follows certain regex rules
    '''
    remove_loc = []
    for i in loc_word_dict:
        if any([re.search(j, loc_word_dict[i], re.I) for j in exclude_list]) or len(loc_word_dict[i]) <= 1:
            remove_loc.append(i)
    for i in remove_loc:
        del loc_word_dict[i]
    return loc_word_dict

def post_process(loc_word_dict,text,return_dict=False):
    '''
    Main method that executes methods above for refining search results after the initial search
    '''
    exclude_list = ['trust', 'ham$', 'ow$', '[a-wz]y$', 'hz', 'echo$', 'ork(s?)$', 'burg', 'guideline', 'suite', 'process', '^tea$', 'ic$', 'coronal', 'gue$', 'ion$', 'tead$', 'able$', 'proxy$', 'gether', 'hy$',
                        'meter', 'success', 'watch', 'threshold', 'epoxy', 'determin', 'risk', 'intermediate', 'intern', 'incubate', 'debate', 'subset', '\wtype', 'scop(e|ic|ie)', 'cop(y|ies)$', 'perform', 'ion$', 'ics$', '^mode', '^state', '-of-', 'xy$','water','-based$']
    
    lwd_copy = {}
    while lwd_copy!=loc_word_dict:
        lwd_copy = loc_word_dict.copy()
        merge_overlapped_pos(loc_word_dict,text)
        
        loc_word_dict = check_neighbor(
            text, loc_word_dict)
        loc_word_dict = fix_parenthesis_dict(
            text, loc_word_dict)
        loc_word_dict = merge_overlapped_pos(
            loc_word_dict, text)
        loc_word_dict = merge_close_pos(loc_word_dict, text)
        loc_word_dict = complete_word(text, loc_word_dict)
    loc_word_dict = exclude(loc_word_dict, exclude_list)
    if return_dict:
        return loc_word_dict
    results=[(loc[0], loc[1], loc_word_dict[loc])
                            for loc in loc_word_dict]
    results.sort(key=lambda x:x[0])
    return results
