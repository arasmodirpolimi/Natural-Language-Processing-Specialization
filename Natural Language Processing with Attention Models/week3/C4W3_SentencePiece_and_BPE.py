#!/usr/bin/env python
# coding: utf-8

# # SentencePiece and BPE 

# ## Introduction to Tokenization

# In order to process text in neural network models it is first required to **encode** text as numbers with ids, since the tensor operations act on numbers. Finally, if the output of the network is to be words, it is required to **decode** the predicted tokens ids back to text.
# 
# To encode text, the first decision that has to be made is to what level of granularity are we going to consider the text? Because ultimately, from these **tokens**, features are going to be created about them. Many different experiments have been carried out using *words*, *morphological units*, *phonemic units* or *characters* as tokens. For example, 
# 
# - Tokens are tricky. (raw text)
# - Tokens are tricky . ([words](https://arxiv.org/pdf/1301.3781))
# - Token s _ are _ trick _ y . ([morphemes](https://arxiv.org/pdf/1907.02423.pdf))
# - t oʊ k ə n z _ ɑː _ ˈt r ɪ k i. ([phonemes](https://www.aclweb.org/anthology/W18-5812.pdf), for STT)
# - T o k e n s _ a r e _ t r i c k y . ([character](https://www.aclweb.org/anthology/C18-1139/))

# But how to identify these units, such as words, is largely determined by the language they come from. For example, in many European languages a space is used to separate words, while in some Asian languages there are no spaces between words. Compare English and Mandarin.
# 
# - Tokens are tricky. (original sentence)
# - 标记很棘手 (Mandarin)
# - Biāojì hěn jíshǒu (pinyin)
# - 标记 很 棘手 (Mandarin with spaces)
# 
# 
# So, the ability to **tokenize**, i.e. split text into meaningful fundamental units, is not always straight-forward.
# 
# Also, there are practical issues of how large our *vocabulary* of words, `vocab_size`, should be, considering memory limitations vs. coverage. A compromise may be need to be made between: 
# * the finest-grained models employing characters which can be memory intensive and 
# * more computationally efficient *subword* units such as [n-grams](https://arxiv.org/pdf/1712.09405) or larger units.
# 
# In [SentencePiece](https://www.aclweb.org/anthology/D18-2012.pdf) unicode characters are grouped together using either a [unigram language model](https://www.aclweb.org/anthology/P18-1007.pdf) (used in this week's assignment) or [BPE](https://arxiv.org/pdf/1508.07909.pdf), **byte-pair encoding**. We will discuss BPE, since BERT and many of its variants use a modified version of BPE and its pseudocode is easy to implement and understand... hopefully!

# ## SentencePiece Preprocessing
# ### NFKC Normalization

# Unsurprisingly, even using unicode to initially tokenize text can be ambiguous, e.g., 

# In[1]:


eaccent = '\u00E9'
e_accent = '\u0065\u0301'
print(f'{eaccent} = {e_accent} : {eaccent == e_accent}')


# SentencePiece uses the Unicode standard normalization form, [NFKC](https://en.wikipedia.org/wiki/Unicode_equivalence), so this isn't an issue. Looking at the example from above but with normalization:

# In[2]:


from unicodedata import normalize

norm_eaccent = normalize('NFKC', '\u00E9')
norm_e_accent = normalize('NFKC', '\u0065\u0301')
print(f'{norm_eaccent} = {norm_e_accent} : {norm_eaccent == norm_e_accent}')


# Normalization has actually changed the unicode code point (unicode unique id) for one of these two characters.

# In[3]:


def get_hex_encoding(s):
    return ' '.join(hex(ord(c)) for c in s)

def print_string_and_encoding(s):
    print(f'{s} : {get_hex_encoding(s)}') 


# In[4]:


for s in [eaccent, e_accent, norm_eaccent, norm_e_accent]:
    print_string_and_encoding(s)


# This normalization has other side effects which may be considered useful such as converting curly quotes &ldquo; to " their ASCII equivalent. (<sup>*</sup>Although we *now* lose directionality of the quote...)

# ### Lossless Tokenization
# 
# SentencePiece also ensures that when you tokenize your data and detokenize your data the original position of white space is preserved. However, tabs and newlines are converted to spaces.
# 
# To ensure this **lossless tokenization**, SentencePiece replaces white space with _ (U+2581). So that a simple join of the tokens by replacing underscores with spaces can restore the white space, even if there are consecutive symbols. But remember first to normalize and then replace spaces with _ (U+2581).

# In[5]:


s = 'Tokenization is hard.'
sn = normalize('NFKC', s)
sn_ = sn.replace(' ', '\u2581')


# In[6]:


print(get_hex_encoding(s))
print(get_hex_encoding(sn))
print(get_hex_encoding(sn_))


# ## BPE Algorithm
# 
# After discussing the preprocessing that SentencePiece performs, you will get the data, preprocess it, and apply the BPE algorithm. You will see how this reproduces the tokenization produced by training SentencePiece on the example dataset (from this week's assignment).
# 
# ### Preparing our Data
# First, you get the Squad data and process it as above.

# In[7]:


import ast

def convert_json_examples_to_text(filepath):
    example_jsons = list(map(ast.literal_eval, open(filepath))) # Read in the json from the example file
    texts = [example_json['text'].decode('utf-8') for example_json in example_jsons] # Decode the byte sequences
    text = '\n\n'.join(texts)       # Separate different articles by two newlines
    text = normalize('NFKC', text)  # Normalize the text

    with open('example.txt', 'w') as fw:
        fw.write(text)
    
    return text


# In[8]:


text = convert_json_examples_to_text('./data/data.txt')
print(text[:900])


# In the algorithm the `vocab` variable is actually a frequency dictionary of the words. Those words have been prepended with an *underscore* to indicate that they are the beginning of a word. Finally, the characters have been delimited by spaces so that the BPE algorithm can group the most common characters together in the dictionary in a greedy fashion. You will see how that is done shortly.

# In[9]:


from collections import Counter

vocab = Counter(['\u2581' + word for word in text.split()])
vocab = {' '.join([l for l in word]): freq for word, freq in vocab.items()}


# In[10]:


def show_vocab(vocab, end='\n', limit=20):
    """Show word frequencys in vocab up to the limit number of words"""
    shown = 0
    for word, freq in vocab.items():
        print(f'{word}: {freq}', end=end)
        shown +=1
        if shown > limit:
            break


# In[11]:


show_vocab(vocab)


# You check the size of the vocabulary (frequency dictionary) because this is the one hyperparameter that BPE depends on crucially on how far it breaks up a word into SentencePieces. It turns out that for your trained model on the small dataset that 60% of 455 merges of the most frequent characters need to be done to reproduce the upperlimit of a 32K `vocab_size` over the entire corpus of examples.

# In[12]:


print(f'Total number of unique words: {len(vocab)}')
print(f'Number of merges required to reproduce SentencePiece training on the whole corpus: {int(0.60*len(vocab))}')


# ### BPE Algorithm
# Directly from the BPE paper you have the following algorithm. 

# In[13]:


import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_sentence_piece_vocab(vocab, frac_merges=0.60):
    sp_vocab = vocab.copy()
    num_merges = int(len(sp_vocab)*frac_merges)
    
    for i in range(num_merges):
        pairs = get_stats(sp_vocab)
        best = max(pairs, key=pairs.get)
        sp_vocab = merge_vocab(best, sp_vocab)

    return sp_vocab


# To understand what's going on first take a look at the third function `get_sentence_piece_vocab`. It takes in the current `vocab` word-frequency dictionary and the fraction, `frac_merges`, of the total `vocab_size` to merge characters in the words of the dictionary, `num_merges` times. Then for each *merge* operation it `get_stats` on how many of each pair of character sequences there are. It gets the most frequent *pair* of symbols as the `best` pair. Then it merges that pair of symbols (removes the space between them) in each word in the `vocab` that contains this `best` (= `pair`). Consequently, `merge_vocab` creates a new `vocab`, `v_out`. This process is repeated `num_merges` times and the result is the set of SentencePieces (keys of the final `sp_vocab`).

# ### Additional Discussion of BPE Algorithm

# Please feel free to skip the below if the above description was enough.
# 
# In a little more detail you can see in `get_stats` you initially create a list of bigram (two character sequence) frequencies from the vocabulary. Later, this may include trigrams, quadgrams, etc. Note that the key of the `pairs` frequency dictionary is actually a 2-tuple, which is just shorthand notation for a pair.
# 
# In `merge_vocab` you take in an individual `pair` (of character sequences, note this is the most frequency `best` pair) and the current `vocab` as `v_in`. You create a new `vocab`, `v_out`, from the old by joining together the characters in the pair (removing the space), if they are present in a word of the dictionary.
# 
# [Warning](https://regex101.com/): the expression `(?<!\S)` means that either a whitespace character follows before the `bigram` or there is nothing before the bigram (it is the beginning of the word), similarly for `(?!\S)` for preceding whitespace or the end of the word. 

# In[14]:


sp_vocab = get_sentence_piece_vocab(vocab)
show_vocab(sp_vocab) 


# ## Train SentencePiece BPE Tokenizer on Example Data
# ### Explore SentencePiece Model
# First, explore the SentencePiece model provided with this week's assignment. Remember you can always use Python's built in `help` command to see the documentation for any object or method.

# In[15]:


import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='./data/sentencepiece.model')


# In[16]:


# help(sp)


# Try it out on the first sentence of the example text.

# In[17]:


s0 = 'Beginners BBQ Class Taking Place in Missoula!'


# In[18]:


# encode: text => id
print(sp.encode_as_pieces(s0))
print(sp.encode_as_ids(s0))

# decode: id => text
print(sp.decode_pieces(sp.encode_as_pieces(s0)))
print(sp.decode_ids([12847, 277]))


# Notice how SentencePiece breaks the words into seemingly odd parts, but you have seen something similar with BPE. But how close was the model trained on the whole corpus of examples with a `vocab_size` of 32,000 instead of 455? Here you can also test what happens to white space, like '\n'. 
# 
# But first note that SentencePiece encodes the SentencePieces, the tokens, and has reserved some of the ids as can be seen in this week's assignment.

# In[19]:


uid = 15068
spiece = "\u2581BBQ"
unknown = "__MUST_BE_UNKNOWN__"

# id <=> piece conversion
print(f'SentencePiece for ID {uid}: {sp.id_to_piece(uid)}')
print(f'ID for Sentence Piece {spiece}: {sp.piece_to_id(spiece)}')

# returns 0 for unknown tokens (we can change the id for UNK)
print(f'ID for unknown text {unknown}: {sp.piece_to_id(unknown)}')


# In[20]:


print(f'Beginning of sentence id: {sp.bos_id()}')
print(f'Pad id: {sp.pad_id()}')
print(f'End of sentence id: {sp.eos_id()}')
print(f'Unknown id: {sp.unk_id()}')
print(f'Vocab size: {sp.vocab_size()}')


# You can also check what are the ids for the first part and last part of the vocabulary.

# In[21]:


print('\nId\tSentP\tControl?')
print('------------------------')
# <unk>, <s>, </s> are defined by default. Their ids are (0, 1, 2)
# <s> and </s> are defined as 'control' symbol.
for uid in range(10):
    print(uid, sp.id_to_piece(uid), sp.is_control(uid), sep='\t')
    
# for uid in range(sp.vocab_size()-10,sp.vocab_size()):
#     print(uid, sp.id_to_piece(uid), sp.is_control(uid), sep='\t')


# ### Train SentencePiece BPE model with our example.txt

# Finally, train your own BPE model directly from the SentencePiece library and compare it to the results of the implemention of the algorithm from the BPE paper itself.

# In[22]:


spm.SentencePieceTrainer.train('--input=example.txt --model_prefix=example_bpe --vocab_size=450 --model_type=bpe')
sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load('example_bpe.model')

print('*** BPE ***')
print(sp_bpe.encode_as_pieces(s0))


# In[23]:


show_vocab(sp_vocab, end = ', ')


# The implementation of BPE's code from the paper matches up pretty well with the library itself! The differences are probably accounted for by the `vocab_size`. There is also another technical difference in that in the SentencePiece implementation of BPE a priority queue is used to more efficiently keep track of the *best pairs*. Actually, there is a priority queue in the Python standard library called `heapq` if you would like to give that a try below! 

# ## Optionally try to implement BPE using a priority queue below

# In[24]:


from heapq import heappush, heappop


# In[25]:


def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]


# In[26]:


a = [1,4,3,1,3,2,1,4,2]
heapsort(a)


# For a more extensive example consider looking at the [SentencePiece repo](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb). The last few sections of this code were repurposed from that tutorial. Thanks for your participation! Next stop BERT and T5!

# In[ ]:




