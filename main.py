import warnings
warnings.filterwarnings("ignore")

import numpy as np
import io
import pickle as pkl
import time
import itertools
import string

import multiprocessing as mp
from multiprocessing import Pool

from collections import defaultdict
from nltk import Tree
from tqdm import tqdm

import argparse


def parse_arguments():
    argparser = argparse.ArgumentParser(description='PCFG Parser arguments')
    
    # Input corpus parameters : 
    argparser.add_argument("--corpus_path", type=str, default="sequoia-corpus+fct.mrg_strict", help="Path to the corpus treebank to take as input")
    argparser.add_argument("--train_size", type=float, default=0.9, help="Size of the training corpus to extract")
    argparser.add_argument("--valid_size", type=float, default=0, help="Size of the validation corpus to extract")
    argparser.add_argument("--test_size", type=float, default=0.1, help="Size of the testing corpus to extract")
    argparser.add_argument("--shuffle_split", action="store_true", help="Shuffle the corpus before splitting the three corpus.")
    
    # OOV parameters :
    argparser.add_argument("--embed_path", type=str, default='polyglot-fr.pkl', help="Path to word embeddings pickle file")
    argparser.add_argument("--lev_candidates", type=int, default=2, help="Number of formal similar words candidates for the OOV module proposed.")
    argparser.add_argument("--emb_candidates", type=int, default=20, help="Number of embedding similar words candidates for the OOV module proposed.")
    argparser.add_argument("--use_damereau", action="store_true", help="Use Damereau-Levenstein distance for the formal similarities") 
    argparser.add_argument("--linear_interpolation", type=float, default=0.2, help="Linear interpolation coefficient for the bigram language model")
    
    # CYK parameters :
    argparser.add_argument("--written_sentence", action="store_true",
                           help="Parse a sentence provided through the argument `--sent_to_parse`. \
                           If not, parse a file of sentences provided through the argument `--file_to_parse`")
    argparser.add_argument("--sent_to_parse", type=str, default="Je suis un élève du master MVA .", help="Written sentence to parse")
    argparser.add_argument("--file_to_parse", type=str, default="test_sentences.txt", 
                           help="Path to the file containing sentences to parse (one sentence per line, exactly one whitespace between each token)")
    argparser.add_argument("--output_path", type=str, default="parse_results.txt", 
                           help="Output path for the sentences parses outputs.")
    
    # Multiprocessing parameters:
    argparser.add_argument("--multiprocess", action="store_true", help="Use multiprocessing. (This may not work on windows)")
    argparser.add_argument("--n_cpus", type=int, default=-1, help="Number of CPUs to use. If -1, use all the CPUs available.")
    
    argparser, _ = argparser.parse_known_args()
    
    return argparser

argparser = parse_arguments()

if argparser.train_size + argparser.valid_size + argparser.test_size != 1:
    raise ValueError('train_size, valid_size and test_size must sum to 1.')


#### MULTIPROCESSING FUNCTION
def multiprocess_func(func, args, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    with Pool(n_jobs) as p:
        res = p.map(func, args)
    return res

#### READ CORPUS
def read_and_ignore_labels(corpus_path):
    corpus = []
    with io.open(corpus_path, encoding='utf-8') as f:
        for line in f:
            sentence = line.rstrip().split(" ")
            sentence = [word.split("-")[0] if word[0]=='(' else word for word in sentence]
            corpus.append(" ".join(sentence))
    return corpus

def extract_sentences(parse_corpus):
    return [" ".join(Tree.fromstring(sentence_parse).leaves()) for sentence_parse in parse_corpus]

corpus = read_and_ignore_labels(argparser.corpus_path)
sentences = extract_sentences(corpus)

#### SPLIT CORPUS
def split_train_val_test(corpus, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=1, shuffle=False):
    corpus_size = len(corpus) 
    
    if not shuffle:
        corpus_test = corpus[int(corpus_size*(train_size+valid_size)):]
        corpus_valid = corpus[int(corpus_size*train_size):int(corpus_size*(train_size+valid_size))]
        corpus_train = corpus[:int(np.ceil(corpus_size*train_size))]

    else:
        from sklearn.model_selection import train_test_split
        slice_idx = int(np.ceil(corpus_size/(test_size*100)))
        corpus_test = corpus[-slice_idx:]
        corpus_train, corpus_valid = train_test_split(corpus[:-slice_idx],
                                                      test_size=test_size/(train_size+valid_size),
                                                      random_state=random_state)
    return corpus_train, corpus_valid, corpus_test

corpus_train, corpus_valid, corpus_test = split_train_val_test(corpus, 
                                                               train_size=argparser.train_size, 
                                                               valid_size=argparser.valid_size, 
                                                               test_size=argparser.test_size, 
                                                               shuffle=argparser.shuffle_split)

sentences_train = extract_sentences(corpus_train)
sentences_valid = extract_sentences(corpus_valid)
sentences_test = extract_sentences(corpus_test)

print("Train corpus length : {} ({:.0f}%)".format(len(corpus_train),len(corpus_train)/len(corpus)*100))
print("Valid corpus length : {} ({:.0f}%)".format(len(corpus_valid),len(corpus_valid)/len(corpus)*100))
print("Test corpus length : {} (Last {:.0f}%)".format(len(corpus_test),len(corpus_test)/len(corpus)*100),"\n")

vocabulary = []
for sentence in sentences_train:
    vocabulary.extend(sentence.split())

vocabulary = np.unique(vocabulary)
vocabulary_indices = {word:idx for idx,word in enumerate(vocabulary)}
print("Vocabulary size :",len(vocabulary))


#### FORMAL AND EMBEDDING SIMILARITY

special_chars = ['½', 'ô', 'ù', 'ä', 'ß', 'î', 'ê', 'ï', 'è', 'á', '©', ' ', 'ó', 'ë', 'ö', 'à', 'â', 'û', '±', 'ç', 'é', '°', 'µ']
ascii_lowercase = [letter for letter in string.ascii_lowercase]
punctuations = [punctuation for punctuation in string.punctuation]
digits = [digit for digit in string.digits]

alphabet = ascii_lowercase+ascii_lowercase+punctuations+digits+special_chars

def damerau_levenshtein(w1, w2):
    
    w1, w2 = w1.lower(), w2.lower()
    n1, n2 = len(w1), len(w2)

    dict_alphabet_indices = {char:idx for idx,char in enumerate(alphabet)}
    
    da = np.zeros(len(alphabet), dtype=np.int8)
    d = np.zeros((n1+2, n2+2), dtype=np.int8)
    maxdist = n1 + n2
    d[0,0] = maxdist
    for i in range(n1+1):
        d[i+1,0] = maxdist
        d[i+1,1] = i
    for j in range(n2+1):
        d[0,j+1] = maxdist
        d[1,j+1] = j
        
    for i in range(1,n1+1):
        db = 0
        for j in range(1,n2+1):
            k = da[dict_alphabet_indices[w2[j-1]]]
            l = db
            if w1[i-1] == w2[j-1]:
                cost = 0
                db = j              
            else:
                cost = 1     
            d[i+1, j+1] = min(d[i,j]+cost, d[i+1,j]+1, d[i,j+1]+1, d[k,l]+(i-k-1)+1+(j-l-1))
            da[dict_alphabet_indices[w1[i-1]]] = i
            
    return d[-1,-1]

def levenshtein(w1, w2): 
    w1, w2 = w1.lower(), w2.lower()
    n1, n2 = len(w1), len(w2)
    lev_ = np.zeros((n1+1, n2+1))
    for i in range(n1+1):
        lev_[i,0] = i
    for j in range(n2+1):
        lev_[0,j] = j
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if w1[i-1] == w2[j-1]:
                lev_[i,j] = min(lev_[i-1,j]+1, lev_[i,j-1]+1, lev_[i-1,j-1])
                
            else:
                lev_[i,j] = min(lev_[i-1,j]+1, lev_[i,j-1]+1, lev_[i-1,j-1]+1)
    return lev_[n1, n2]


words, embeddings = pkl.load(open(argparser.embed_path, 'rb'), encoding = 'latin')

dict_words_embeddings = {word:embedding for word, embedding in zip(words, embeddings)}

get_embedding = np.vectorize(lambda word : dict_words_embeddings.get(word,np.full(64, None)),signature="()->(n)")

vocabulary_ = np.array(list(set(vocabulary)&set(words)))
vocab_embeddings = np.vstack(get_embedding(vocabulary_))

del words, embeddings

print("Vocabulary size with available embeddings :",len(vocabulary_),"\n")


def find_closest_words_emb(word, k=5):
    
    if word in dict_words_embeddings:
        
        cosine_similarity = lambda x1,x2 : np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
        get_embedding = np.vectorize(lambda word : dict_words_embeddings.get(word,np.full(64, None)),signature="()->(n)")
        word_embedding = get_embedding(word)
        word_vocab_cosine = np.vectorize(lambda vocab_word_emb : cosine_similarity(vocab_word_emb,word_embedding), signature="(n)->()")
        vocab_similiraties = word_vocab_cosine(vocab_embeddings)
        return [word_emb_[1] for word_emb_ in sorted(zip(-vocab_similiraties,vocabulary_))[:k]]
    
    else:
        return []


def find_closest_words_lev(word, k=10, damereau=False):
    if damereau:
        levenstein = np.vectorize(lambda w : damerau_levenshtein(w,word))
    else:
        levenstein = np.vectorize(lambda w : levenshtein(w,word))
    vocab_distances = levenstein(vocabulary)
    return [word_lev_[1] for word_lev_ in sorted(zip(vocab_distances,vocabulary))[:k]]

#### LANGUAGE MODEL

class Bigrams_Unigrams():
    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)
        self.vocab_dict = {word:idx for idx,word in enumerate(vocabulary)}
        self.n_grams = 2
        self.transition_matrix = np.ones((len(vocabulary),len(vocabulary)))
        self.unigram = np.zeros(len(vocabulary))    

    def learn(self, sentences):
        # Build bigram transition matrix and unigram probabilities
        tokens_list = [sentence.split() for sentence in sentences]
        for tokens in tokens_list:
            for i in range(len(tokens)):
                if i>=self.n_grams-1:
                    word = tokens[i]
                    grams = tokens[np.maximum(0,i-self.n_grams+1):i]
                    for previous_word in grams:
                        self.transition_matrix[self.vocab_dict[previous_word],self.vocab_dict[word]] += 1
                self.unigram[self.vocabulary.index(tokens[i])] += 1      
            
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix,axis=1).reshape(-1,1)
        self.unigram = self.unigram/np.sum(self.unigram)
        
        return self.unigram, self.transition_matrix

Bi_uni_grams = Bigrams_Unigrams(vocabulary=vocabulary)
unigrams, transition_matrix = Bi_uni_grams.learn(sentences_train)

#### OOV MODULE

def process_word(idx, word, score, proposed_sentence=None):
    if idx==0:    
        return np.log(unigrams[vocabulary_indices[word]])
    else:
        coeff = coeff = argparser.linear_interpolation
        return np.log((1-coeff)*transition_matrix[vocabulary_indices[proposed_sentence[-1]],vocabulary_indices[word]] + \
                      coeff*unigrams[vocabulary_indices[word]])

def propose_words(word, lev_candidates=10, emb_candidates=10, damereau=True):
    proposed_words,k = [], lev_candidates
    while not proposed_words:
        proposed_words = set(find_closest_words_emb(word, k=emb_candidates)).union(set(find_closest_words_lev(word, k=k, damereau=damereau)))
        k+=1
    return proposed_words

def OOV_module(sentence, lev_candidates=2, emb_candidates=20, vocabulary=vocabulary, damereau=False):

    score = 0
    proposed_sentence = []
    
    for idx,word in enumerate(sentence.split()):
        if word in vocabulary:
            score += process_word(idx,word,score,proposed_sentence)
            proposed_sentence = proposed_sentence + [word]
        else:
            proposed_words = propose_words(word, lev_candidates=lev_candidates, emb_candidates=emb_candidates, damereau=damereau)
            couples = []
            for proposed_word in proposed_words:
                couples.append((process_word(idx, proposed_word, score, proposed_sentence),proposed_word))

            best_score_word = sorted(couples)[-1]
            proposed_sentence = proposed_sentence + [best_score_word[1]]
            score += best_score_word[0]
    
    return " ".join(proposed_sentence)

#### PCFG extraction

def transform_parse(parse, to_chomsky=False):
    t = Tree.fromstring(parse)
    
    if to_chomsky:
        t.chomsky_normal_form(horzMarkov=2)
        t.collapse_unary(collapsePOS=True, collapseRoot=True)
    else:
        t.un_chomsky_normal_form()
    
    return " ".join(str(t).split())


def extract_lexicon(corpus):
    tokens_POS = []
    
    for sentence in corpus:
        tree = Tree.fromstring(sentence,remove_empty_top_bracketing=True)
        tree.chomsky_normal_form(horzMarkov=2)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        sentence_lexical_rules = [rule for rule in tree.productions() if rule.is_lexical()]
        tokens_POS.extend(sentence_lexical_rules)
    
    # Count pairs occurencies
    unique, counts = np.unique(tokens_POS, return_counts=True)  
    unique = np.array([[rule.lhs().symbol(), rule.rhs()[0].lower()] for rule in unique])
    counts = counts.astype(np.float64)
    
    # Compute probabilistic lexicon  
    data = np.hstack((unique,counts.reshape(-1,1)))
    
    lexicon_grammar = defaultdict(set)
    for key, val in data[:,:2]:
        lexicon_grammar[key].add(val)
    
    # Convert to matrix
    POS_tags, POS_tags_pos = np.unique(data[:, 0], return_inverse=True)
    words, words_pos = np.unique(data[:, 1], return_inverse=True)
    
    dict_POS_tags = {POS_tag:idx for idx,POS_tag in enumerate(POS_tags)}
    dict_words = {word:idx for idx,word in enumerate(words)}

    lexicon_probabilities = np.zeros((len(POS_tags), len(words)))
    lexicon_probabilities[POS_tags_pos, words_pos] = data[:, 2]
    
    lexicon_probabilities = lexicon_probabilities / np.sum(lexicon_probabilities,axis=1).reshape(-1,1)
    
    return lexicon_probabilities, dict_POS_tags, dict_words, lexicon_grammar


def extract_PCFG(corpus):
    tokens_POS = []
    axioms = set()
    for sentence in corpus:
        tree = Tree.fromstring(sentence, remove_empty_top_bracketing=True)
        tree.chomsky_normal_form(horzMarkov=2)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        rules = tree.productions()
        sentence_lexical_rules = [rule for rule in rules if rule.is_nonlexical()]
        axioms.add(rules[0].lhs().symbol())
        tokens_POS.extend(sentence_lexical_rules)
    
    # Count pairs occurencies
    unique, counts = np.unique(tokens_POS, return_counts=True)
    unique = np.array([[rule.lhs(), rule.rhs()]  for rule in unique])

    counts = counts.astype(np.float64)
    
    # Compute PCFG
    data = np.hstack((unique,counts.reshape(-1,1)))
    
    PCFG_grammar = defaultdict(set)
    for key, val in data[:,:2]:
        if len(val)>1: PCFG_grammar[key.symbol()].add((val[0].symbol(),val[1].symbol()))
        else: PCFG_grammar[key.symbol()].add(val[0].symbol())
    
    # Convert to matrix
    init_non_terminals, row_pos = np.unique(data[:, 0], return_inverse=True)
    target_non_terminals, col_pos = np.unique(data[:, 1], return_inverse=True)

    PCFG = np.zeros((len(init_non_terminals), len(target_non_terminals)))
    PCFG[row_pos, col_pos] = data[:, 2]
    
    PCFG = PCFG / np.sum(PCFG,axis=1).reshape(-1,1)
    
    dict_init_non_terminals = {NT.symbol():idx for idx,NT in enumerate(init_non_terminals)}
    dict_target_non_terminals = {(B.symbol(),C.symbol()):idx for idx,(B,C) in enumerate(target_non_terminals) if len((B,C))>1}
    dict_target_non_terminals.update({NT[0].symbol():idx for idx,NT in enumerate(target_non_terminals) if len(NT)==1})
    
    return PCFG, dict_init_non_terminals, dict_target_non_terminals, PCFG_grammar, list(axioms)

print("Extracting PCFG from the training corpus..","\n")

lexicon_probabilities, dict_POS_tags, dict_words, lexicon_grammar = extract_lexicon(corpus_train)
PCFG, dict_init_non_terminals, dict_target_non_terminals, Grammar, axioms = extract_PCFG(corpus_train)

#### Creating some dictionaries to optimize CYK
unaries_target = set([target for target in dict_target_non_terminals.keys() if np.ndim(target)==0])
binaries_target = set(dict_target_non_terminals.keys()) - set(unaries_target)

Grammar_unaries, Grammar_unaries_POS = defaultdict(set), defaultdict(set)
Grammar_binaries = defaultdict(set)
for nt,target in Grammar.items():
    unaries_tg = target & unaries_target
    unaries_tg_POS = unaries_tg & set(dict_POS_tags)
    binaries_tg = target & binaries_target
    if unaries_tg:
        Grammar_unaries.update({nt:unaries_tg})
        if unaries_tg_POS:
            Grammar_unaries_POS.update({nt:unaries_tg_POS})
    if binaries_tg:
        Grammar_binaries.update({nt:binaries_tg})

Grammar_unaries_inv, Grammar_unaries_POS_inv, Grammar_binaries_inv = defaultdict(set), defaultdict(set), defaultdict(set)

for nt,targets in Grammar_unaries.items():
    for target in targets:
        Grammar_unaries_inv[target].add(nt)
        
for nt,targets in Grammar_unaries_POS.items():
    for target in targets:
        Grammar_unaries_POS_inv[target].add(nt)
        
for nt,targets in Grammar_binaries.items():
    for target in targets:
        Grammar_binaries_inv[target].add(nt)

lexicon_grammar_inv = defaultdict(set)
for POS,words in lexicon_grammar.items():
    for word in words:
        lexicon_grammar_inv[word].add(POS)

left_binaries = set()
right_binaries = set()
for binary in Grammar_binaries_inv:
    left_binaries.add(binary[0])
    right_binaries.add(binary[1])
left_right_binaries = left_binaries.union(right_binaries)

dict_NT = dict_init_non_terminals.copy()
dict_NT.update({NT:idx+len(dict_init_non_terminals) for NT,idx in dict_POS_tags.items()})
dict_NT_inv = {idx:NT for NT,idx in dict_NT.items()}

del unaries_target, binaries_target

binaries = set(Grammar_binaries_inv)

#### CYK Parser

def CYK_parser(sentence, 
               PCFG=PCFG, 
               lexicon_probabilities=lexicon_probabilities, 
               axioms=axioms,
               use_damerau=argparser.use_damereau,
               lev_candidates=argparser.lev_candidates, 
               emb_candidates=argparser.emb_candidates):

    proposed_sentence = OOV_module(sentence=sentence, lev_candidates=lev_candidates, emb_candidates=emb_candidates, damereau=use_damerau)
    
    sentence_tokens = sentence.split()   
    proposed_sentence_tokens = proposed_sentence.split()
    
    n = len(sentence_tokens)
    
    
    score = [[{} for i in range(n+1)] for j in range(n+1)]
    score_sets_left = [[set() for i in range(n+1)] for j in range(n+1)]
    score_sets_right = [[set() for i in range(n+1)] for j in range(n+1)]
    back = np.empty((n, n+1, len(dict_NT)), dtype=object)
    
    
    for i,word in enumerate(proposed_sentence_tokens):
        for POS in lexicon_grammar_inv[word.lower()]:
            score[i][i+1][POS] = lexicon_probabilities[dict_POS_tags[POS], dict_words[word.lower()]]
            if POS in left_binaries:
                score_sets_left[i][i+1].add(POS)
            if POS in right_binaries:
                score_sets_right[i][i+1].add(POS)
                      
    for span in range(2,n+1):
        for begin in range(n-span+1):
            end = begin + span
            for split in range(begin+1, end):
                container_B = score_sets_left[begin][split]
                container_C = score_sets_right[split][end]
                for B,C in binaries & set(itertools.product(container_B,container_C)):
                    for A in Grammar_binaries_inv[(B,C)]:
                        prob = score[begin][split][B] * score[split][end][C] * PCFG[dict_init_non_terminals[A],dict_target_non_terminals[(B,C)]]
                        if prob > score[begin][end].get(A,0):
                            score[begin][end][A] = prob
                            if A in left_binaries:
                                score_sets_left[begin][end].add(A)
                            if A in right_binaries:
                                score_sets_right[begin][end].add(A)
                            back[begin,end,dict_NT[A]] = (split,B,C)
            

    
    def tree(root, children):
        L = [root]
        L.extend(children)
        return str(L).replace(",","").replace("[","(").replace("]",")").replace("'","")
    
    def process_word(word, word2parse=True):
        from_= [",", "(" , ")", "[", "]", "'"]
        to_ = ["$v$i$r$g", "$l$p$", "$r$p$", "$l$b$", "$r$b$", "$a$p$"]
        
        if word2parse:
            for i,j in zip(from_,to_):
                word = word.replace(i,j)
        else:
            for i,j in zip(to_,from_):
                word = word.replace(i,j)
        
        return word
    
    def expand_node(idx_tuple, sentence_tokens):
            back_tuple = back[idx_tuple]

            if isinstance(back_tuple, tuple):
                split, B, C = back_tuple    
                idx_tuple_1, idx_tuple_2 = (idx_tuple[0], split, dict_NT[B]), (split, idx_tuple[1], dict_NT[C])
                return [tree(B,expand_node(idx_tuple_1, sentence_tokens)), tree(C,expand_node(idx_tuple_2, sentence_tokens))]  

            elif back_tuple==None:
                return [process_word(sentence_tokens.pop(0))]

            else :
                final_tuple = idx_tuple[0],idx_tuple[1], dict_NT[back_tuple]
                return [tree(back_tuple,expand_node(final_tuple, sentence_tokens))]

    def buildTree(back, sentence_tokens):

        n = back.shape[0]
        axiom_scores = np.array([score[0][n].get(ax,0) for ax in axioms])
        axiom = axioms[np.argmax(axiom_scores)]
        
        if np.all(axiom_scores==0) :
            children = [tree("NOT_IN_GRAMMAR", [process_word(word, word2parse=True)]) for word in sentence_tokens]
            return process_word(tree("", [tree("NOT_IN_GRAMMAR", children)]), word2parse=False)
        
        if n==1:
            return process_word(tree("",[tree(axiom, [process_word(sentence_tokens[0])])]), word2parse=False)
        
        idx_tuple=(0,n,dict_NT[axiom])
        mytree = process_word(tree("", [tree(axiom, expand_node(idx_tuple, sentence_tokens))]), word2parse=False)
        return mytree
    
    parse_chomsky = buildTree(back, sentence_tokens)
    
    parse_unchomsky = transform_parse(parse_chomsky, to_chomsky=False)

    return parse_unchomsky

#### Parsing

if argparser.written_sentence :
    print("Parsing sentence :",argparser.sent_to_parse)
    parses_results = CYK_parser(sentence = argparser.sent_to_parse)
    print(parses_results)
    
else :
    print("Parsing sentences in :",argparser.file_to_parse)
    if argparser.multiprocess:
        print("Using multiprocessing..")
        
        with io.open(argparser.file_to_parse, encoding='utf-8') as f:
            sentences = [sentence for sentence in f]
        tic = time.time()
        parses_results = multiprocess_func(CYK_parser, sentences, n_jobs=argparser.n_cpus)
        toc = time.time()
        print("Time elapsed :",(toc-tic)//60,"min",np.round((toc-tic)%60,2),"s")
        
    else :
        parses_results = []
        tic = time.time()
        with io.open(argparser.file_to_parse, encoding='utf-8') as f:
            for sentence in tqdm(f):
                parses_results.append(CYK_parser(sentence))
        toc = time.time()
        print("Time elapsed :",(toc-tic)//60,"min",np.round((toc-tic)%60,2),"s")
                
    print("\nWriting output to :",argparser.output_path)
    with open(argparser.output_path, 'w', encoding="utf-8") as f:
        for idx,parse in enumerate(parses_results):
            if idx < len(parses_results)-1:
                f.write(parse+"\n")
            else:
                f.write(parse)
    f.close()