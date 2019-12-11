#!/usr/bin/env python3

"""
Generate data used for model training using CFG. We define 5 operations that
can be requested by the user: add class, add attribute, create composition,
create association, create inheritance. A user query can be a direct request,
e.g "create an object A", or it can be indirect (by stating the relationship
between objects and/or attributes). Here are some example query for each
type of operation:

- Add class:
    Create a school.
    Create person.
    Add a class student
- Add attribute:
    Add work in person.
    Student contains a numeric identifier.
    Medicines have an active ingredient.
- Create composition:
    The house is made of rooms.
    A class consists of many students.
- Create association:
    A student can pass exams.
    The cheque is sent to the bank.
- Create inheritance:
    The wife of a man is a woman.
    Students and teachers are human.
"""

from nltk.parse.generate import generate
from nltk import CFG
from nltk.corpus import wordnet
import random

# number of sentences per grammar
NUMBER_OF_SENTENCES = 2000
# word bank and grammars for each intent
BASE_NOUNS = [  ]
BASE_VERBS = [  ]
ADD_CLASS_GRAMMAR = '''
    S   -> VP NP
    VP  -> V | V 'a class'
    NP  -> Det N | N
    Det -> 'a' | 'an' | 'the'
    V   -> 'create' | 'add' | 'construct' | 'make'
'''
ADD_ATTRIBUTE_GRAMMAR = '''
    S   -> QR | DS
    QR  -> VP1 'to' N2
    DS  -> N2 VP2
    VP1 -> V1 NP1
    VP2 -> V2 NP1
    V1  -> 'add'
    V2  -> 'has' | 'contains' | 'is identified by' | 'is characterized by'
    NP1 -> Det JJ 'attribute' N1 | Det JJ N1 | Det N1
    JJ  -> 'text' | 'numeric' | 'boolean' | 'binary'
    Det -> 'a' | 'an' | 'the'
    N1  -> 'id' | 'name' | 'age' | 'value' | 'count' | 'number' | 'state'
'''
CREATE_COMPOSITION_GRAMMAR = '''
    S   -> S_D | S_I
    V_D -> 'contains' | 'has' | 'consists of' | 'is composed of'
    V_I -> 'is part of' | 'are part of' | 'is in' | 'constitutes' | 'make up'
'''
CREATE_ASSOCIATION_GRAMMAR = '''
    S   -> NP VP
    NP  -> Det N | N
    VP  -> V NP
    Det -> 'a' | 'an' | 'the'
'''
CREATE_INHERITANCE_GRAMMAR = '''
    S   -> S_1 | S_2
    V1  -> 'is' | 'is a'
    V2  -> 'can be' | 'can be a'
'''
GRAMMARS = {  }


def create_terminals(lst):
    return ' | '.join([f"'{t}'" for t in lst])


def init_glob():
    '''Initialize the global variables, including list of nouns and the CFGs'''
    global BASE_NOUNS, BASE_VERBS, ADD_CLASS_GRAMMAR, ADD_ATTRIBUTE_GRAMMAR, \
            CREATE_COMPOSITION_GRAMMAR, CREATE_ASSOCIATION_GRAMMAR, \
            CREATE_INHERITANCE_GRAMMAR, GRAMMARS

    # load list of nouns from a text file
    with open('nouns.txt', 'r') as fd:
        raw_text = fd.readlines()
        random.shuffle(raw_text)
        BASE_NOUNS = set([word.strip() for word in raw_text])

    # similarly, load a list of verbs
    with open('verbs.txt', 'r') as fd:
        raw_text = fd.readlines()
        random.shuffle(raw_text)
        BASE_VERBS = set([word.strip() for word in raw_text])

    # add some more terminals for each of the grammar
    # grammar for add class
    list_of_nouns = random.sample(BASE_NOUNS, 500)
    noun_terminals = create_terminals(list_of_nouns)
    rule = f'N -> {noun_terminals}'
    ADD_CLASS_GRAMMAR += rule

    # grammar for add attribute
    list_of_nouns = random.sample(BASE_NOUNS, 50)
    noun_terminals = create_terminals(list_of_nouns)
    rule = f'N2 -> {noun_terminals}'
    ADD_ATTRIBUTE_GRAMMAR += rule

    # grammar for create composition
    list_of_nouns = random.sample(BASE_NOUNS, 500)
    list_vd = []
    list_vi = []
    for noun in list_of_nouns:
        synonyms = wordnet.synsets(noun, pos='n')
        for s in synonyms:
            hs = s.part_holonyms()
            if len(hs) == 0:
                continue
            h = hs[0]
            sname = s.lemma_names()[0].replace('_',' ')
            hname = h.lemma_names()[0].replace('_',' ')
            list_vd.append(f"'{hname}' V_D '{sname}'")
            list_vi.append(f"'{sname}' V_I '{hname}'")
    rule = "S_D -> " + ' | '.join(list_vd) + "\n"
    rule += "S_I -> " + ' | '.join(list_vi) + "\n"
    CREATE_COMPOSITION_GRAMMAR += rule

    # grammar for create association
    list_of_verbs = random.sample(BASE_VERBS, 50)
    list_of_nouns = random.sample(BASE_NOUNS, 50)
    verb_terminals = create_terminals(list_of_verbs)
    noun_terminals = create_terminals(list_of_nouns)
    rule = f'V -> {verb_terminals}\nN -> {noun_terminals}'
    CREATE_ASSOCIATION_GRAMMAR += rule

    # grammar for create inheritance
    list_of_nouns = random.sample(BASE_NOUNS, 500)
    list_isa = []
    list_risa = []
    for noun in list_of_nouns:
        synonyms = wordnet.synsets(noun, pos='n')
        for s in synonyms:
            hs = s.hypernyms()
            if len(hs) == 0:
                continue
            h = hs[0]
            sname = s.lemma_names()[0].replace('_',' ')
            hname = h.lemma_names()[0].replace('_',' ')
            list_isa.append(f"'{sname}' V1 '{hname}'")
            list_risa.append(f"'{hname}' V2 '{sname}'")
    rule = "S_1 -> " + ' | '.join(list_isa) + "\n"
    rule += "S_2 -> " + ' | '.join(list_risa) + "\n"
    CREATE_INHERITANCE_GRAMMAR += rule

    GRAMMARS = {
        'add_class': ADD_CLASS_GRAMMAR,
        'add_attribute': ADD_ATTRIBUTE_GRAMMAR,
        'create_composition': CREATE_COMPOSITION_GRAMMAR,
        'create_association': CREATE_ASSOCIATION_GRAMMAR,
        'create_inheritance': CREATE_INHERITANCE_GRAMMAR
    }


def datagen(operation, raw_grammar):
    '''Actually generate sentences based on CFG and write the data to a text file'''
    output_file = f'{operation}.grammar'

    grammar = CFG.fromstring(raw_grammar)
    all_sentences = [' '.join(sentence) for sentence in generate(grammar)]
    sentences = random.sample(all_sentences, min(len(all_sentences), NUMBER_OF_SENTENCES))

    print(f'Selecting {NUMBER_OF_SENTENCES} out of {len(all_sentences)} generated sentences')
    #print(operation, grammar, sep=': ')
    with open(output_file, 'w') as fd:
        fd.writelines('\n'.join(sentences))
    print(f'Output for operation {operation} written to {output_file}')
    print('---------------------------------------------------------------')


def csv_gen(operation, raw_grammar):
    '''Create the same data as above, but in csv format'''
    output_file = f'{operation}.csv'

    grammar = CFG.fromstring(raw_grammar)
    all_sentences = [' '.join(sentence) for sentence in generate(grammar)]
    sentences = random.sample(all_sentences, min(len(all_sentences), NUMBER_OF_SENTENCES))
    sentences = list(map(lambda x : x + f',{operation}', sentences))

    print(f'Selecting {NUMBER_OF_SENTENCES} out of {len(all_sentences)} generated sentences')
    with open(output_file, 'w') as fd:
        fd.writelines(f'\n'.join(sentences))
    print(f'Output for operation {operation} written to {output_file}')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    init_glob()
    for key, value in GRAMMARS.items():
        datagen(key, value)
        csv_gen(key, value)

