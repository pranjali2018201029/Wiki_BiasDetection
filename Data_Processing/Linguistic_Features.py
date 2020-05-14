
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob

lemmatizer = WordNetLemmatizer()


## Encodings

## POS Tags
## For descriptions of these tags refer: https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/
POS_Tag_Encoding = {"CC" : 0, "CD" : 1, "DT" : 2, "EX" : 3, "FW" : 4, "IN" : 5, "JJ" : 6, "JJR" : 7, "JJS" : 8,
                   "LS" : 9, "MD" : 10, "NN" : 11, "NNS" : 12, "NNP" : 13, "NNPS" : 14, "PDT" : 15, "POS" : 16,
                   "PRP" : 17, "PRP$" : 18, "RB" : 19, "RBR" : 20, "RBS" : 21, "RP" : 22, "TO" : 23, "UH" : 24,
                   "VB" : 25, "VBD" : 26, "VBG" : 27, "VBN" : 28, "VBP" : 29, "VBZ" : 30, "WDT" : 31, "WP" : 32,
                   "WP$" : 33, "WRB" : 34}

## Position in sentence
Sent_Position_Encoding = {"Start" : 0, "Middle" : 1, "End" : 2 }

## Polarity
Polarity_Encoding = {"Negative" : 0, "Positive" : 1, "Neutral" : 2}

## Gramatical relation : Not Implemented
# Grammatical_Rel_Encoding = {}

## Hedge, Factive Verb, Assertive Verb, Implicative Verb, Report Verb, Entailment, Strong Subjective, Weak Subjective,
## Positive word, Negative Word, Bias Lexicon
## For all above : 1 --> Present in the corresponding lexicon list, 0 --> Not present in the corresponding lexicon list
Hedges = []
Factive_Verbs = []
Assertive_Verbs = []
Implicative_Verbs = []
Report_Verbs = []
Entailments = []
Strong_Subjectives = []
Weak_Subjectives = []
Positive_Words = []
Negative_Words = []
Bias_Lexicons = []

Lexicon_Folder_Path = "/Users/pranjali/Downloads/Wiki_BiasDetection/Lexicons/"

Feature_Columns = ["word", "lemma", "POS", "POS_Prev", "POS_Next", "Sent_Position", 
                  "Hedge", "Hedge_Context", "Factive", "Factive_Context", "Assertive", "Assertive_Context",
                  "Implicative", "Implicative_Context", "Report", "Report_Context", 
                   "Entailment", "Entailment_Context", "StrongSub", "StrongSub_Context", 
                   "WeakSub", "WeakSub_Context", "Polarity", "Positive", "Positive_Context", 
                   "Negative", "Negative_Context", "Bias_Lexicon"]


def Load_Lexicon_File(Filename):
    
    with open(Lexicon_Folder_Path + Filename, "r") as file:
        Lines = file.readlines()
    
    Lexicon_List = []
    for line in Lines:
        Lexicon_List.append(line[:-1])
        
    return Lexicon_List


def Load_Lexicons():
    
    global Hedges, Factive_Verbs, Assertive_Verbs, Implicative_Verbs, Report_Verbs, Entailments
    global Strong_Subjectives, Weak_Subjectives, Positive_Words, Negative_Words, Bias_Lexicons
    
    Hedges = Load_Lexicon_File("Hyland_Hedges_2005.txt")
    Factive_Verbs = Load_Lexicon_File("Hooper_Factives_1975.txt")
    Assertive_Verbs = Load_Lexicon_File("Hooper_Assertives_1975.txt")
    Implicative_Verbs = Load_Lexicon_File("Karttunen_Implicatives_1971.txt")
    Report_Verbs = Load_Lexicon_File("Report_Verbs.txt")
    Entailments = Load_Lexicon_File("Berant_Entailments_2012.txt")
    Strong_Subjectives = Load_Lexicon_File(" Wiebe_Riloff_Strong_Subjectives_2003.txt")
    Weak_Subjectives = Load_Lexicon_File("Wiebe_Riloff_Weak_Subjectives_2003.txt")
    Positive_Words = Load_Lexicon_File("Liu_Positive_Words_2005.txt")
    Negative_Words = Load_Lexicon_File("Liu_Negative_Words_2005.txt")
    Bias_Lexicons = Load_Lexicon_File("NPOV_Edits.txt")

## As lemma is a string value, not sure how to use it ( Can't use its embedding, 
## Can we calculate other  linguistic features on lemma instead of original word?)
def Get_Lemma(word):
    lemma = lemmatizer.lemmatize(word)
    return lemma


## Input: Sentence, Output: List of tuples (word, POS_Tag) for all words in the sentence
def Get_POS_Tag(sentence):
    sent_words = word_tokenize(sentence)
    POS_Tag_List = nltk.pos_tag(sent_words)
    return POS_Tag_List


## Returns position of the word in the sentence. (Start->0, Middle->1, End->2)
def Get_Sentence_Position(Word_Index, Sent_Length):
    
    part_size = int(Sent_Length/3)
    
    if Word_Index < part_size:
        return 0
    elif Word_Index < 2*part_size:
        return 1
    else:
        return 2

## Ideally, The polarity of word according to Riloff and Wiebe's paper need to be calculated
## But I have used TextBlob.
def Get_Polarity(word):
    
    Polarity = TextBlob(word).sentiment.polarity
    
    if Polarity < 0:
        return 0
    elif Polarity > 0:
        return 1
    else:
        return 2


## Not implemented yet
def Get_Grammatical_Rel(word, sentence):
    return GR_Val


## Not implemented yet : We can use NPOV edits data that we have 
def Get_Collaborative_Feature(word):
    return CF_Val


## Input : Sentence
## Output : Pandas DataFrame where each row represents linguistic features of word in the sentence, 
##          Columns names list is mentioned at the start of the code

def Get_Sent_Linguistic_Features(Sentence):
    
    ## Word_Features: List representing linguistic features of the word in the sentence
    ## Sentence_Features : List of Word_Features lists 
    
    Sentence_Features = []
    
    Sent_Words = word_tokenize(Sentence)
    Sent_Words = [w for w in Sent_Words if w.isalpha()]
    Sent_Length = len(Sent_Words)
    
    Sent_POS_Tags = Get_POS_Tag(Sentence)
    Sent_POS_Tags = [t for t in Sent_POS_Tags if t[0].isalpha()]
    
    for Word_Index in range(len(Sent_Words)):
        
        Word = Sent_Words[Word_Index]
        
        ## Context Words
        if Word_Index > 0:
            Prev_Word = Sent_Words[Word_Index-1]
        else:
            Prev_Word = -1
            
        if Word_Index < Sent_Length-1:
            Next_Word = Sent_Words[Word_Index+1]
        else:
            Next_Word = -1
        
        ## Feature 1: Word (string)
        Word_Features = [Word]

        ## Feature 2: Lemma (string)
        word_lemma = Get_Lemma(Word)
        Word_Features.append(word_lemma)

        ## Feature 3: POS Tag (Tag encoded into int)
        POS_Tag = Sent_POS_Tags[Word_Index][1]
        POS_Tag_Val = POS_Tag_Encoding[POS_Tag]
        Word_Features.append(POS_Tag_Val)

        ## Feature 4: POS Tag of previous word (Tag encoded into int)
        if Word_Index > 0:
            POS_Tag = Sent_POS_Tags[Word_Index-1][1]
            POS_Tag_Val = POS_Tag_Encoding[POS_Tag]
        else:
            POS_Tag_Val = -1
        Word_Features.append(POS_Tag_Val)

        ## Feature 5: POS Tag of next word (Tag encoded into int)
        if Word_Index < Sent_Length-1:
            POS_Tag = Sent_POS_Tags[Word_Index+1][1]
            POS_Tag_Val = POS_Tag_Encoding[POS_Tag]
        else:
            POS_Tag_Val = -1
        Word_Features.append(POS_Tag_Val)

        ## Feature 6: Position of word in the sentence (Encoded into int)
        Sent_Position_Val = Get_Sentence_Position(Word_Index, Sent_Length)
        Word_Features.append(Sent_Position_Val)

        ## Feature 7: Hedge (1 or 0)
        if Word in Hedges:
            Hedge_Val = 1
        else:
            Hedge_Val = 0
        Word_Features.append(Hedge_Val)

        ## Feature 8: Hedge Context i.e. if Hedge is present in the context (1 or 0)
        Prev_Hedge_Val = 0
        if Prev_Word in Hedges:
            Prev_Hedge_Val = 1
        
        Next_Hedge_Val = 0
        if Next_Word in Hedges:
            Next_Hedge_Val = 1
                
        if Prev_Hedge_Val or Next_Hedge_Val:
            Hedge_Val = 1
        else: 
            Hedge_Val = 0
            
        Word_Features.append(Hedge_Val)

        ## Feature 9: Factive Verb (1 or 0)
        if Word in Factive_Verbs:
            Factive_Verb_Val = 1
        else:
            Factive_Verb_Val = 0
        Word_Features.append(Factive_Verb_Val)

        ## Feature 10: Factive Verb Context i.e. if Factive Verb is present in the context (1 or 0)
        Prev_Factive_Verb_Val = 0
        if Prev_Word in Factive_Verbs:
            Prev_Factive_Verb_Val = 1
        
        Next_Factive_Verb_Val = 0
        if Next_Word in Factive_Verbs:
            Next_Factive_Verb_Val = 1
                
        if Prev_Factive_Verb_Val or Next_Factive_Verb_Val:
            Factive_Verb_Val = 1
        else: 
            Factive_Verb_Val = 0
            
        Word_Features.append(Factive_Verb_Val)
        
        ## Feature 11: Assertive Verb (1 or 0)
        if Word in Assertive_Verbs:
            Assertive_Verb_Val = 1
        else:
            Assertive_Verb_Val = 0
        Word_Features.append(Assertive_Verb_Val)

        ## Feature 12: Assertive Verb Context i.e. if Assertive Verb is present in the context (1 or 0)
        Prev_Assertive_Verb_Val = 0
        if Prev_Word in Assertive_Verbs:
            Prev_Assertive_Verb_Val = 1
        
        Next_Assertive_Verb_Val = 0
        if Next_Word in Assertive_Verbs:
            Next_Assertive_Verb_Val = 1
                
        if Prev_Assertive_Verb_Val or Next_Assertive_Verb_Val:
            Assertive_Verb_Val = 1
        else: 
            Assertive_Verb_Val = 0
            
        Word_Features.append(Assertive_Verb_Val)
        
        ## Feature 13: Implicative Verb (1 or 0)
        if Word in Implicative_Verbs:
            Implicative_Verb_Val = 1
        else:
            Implicative_Verb_Val = 0
        Word_Features.append(Implicative_Verb_Val)

        ## Feature 14: Implicative Verb Context i.e. if Implicative Verb is present in the context (1 or 0)
        Prev_Implicative_Verb_Val = 0
        if Prev_Word in Implicative_Verbs:
            Prev_Implicative_Verb_Val = 1
        
        Next_Implicative_Verb_Val = 0
        if Next_Word in Implicative_Verbs:
            Next_Implicative_Verb_Val = 1
                
        if Prev_Implicative_Verb_Val or Next_Implicative_Verb_Val:
            Implicative_Verb_Val = 1
        else: 
            Implicative_Verb_Val = 0
            
        Word_Features.append(Implicative_Verb_Val)
        
        ## Feature 15: Report Verb (1 or 0)
        if Word in Report_Verbs:
            Report_Verb_Val = 1
        else:
            Report_Verb_Val = 0
        Word_Features.append(Report_Verb_Val)

        ## Feature 16: Report Verb Context i.e. if Report Verb is present in the context (1 or 0)
        Prev_Report_Verb_Val = 0
        if Prev_Word in Report_Verbs:
            Prev_Report_Verb_Val = 1
        
        Next_Report_Verb_Val = 0
        if Next_Word in Report_Verbs:
            Next_Report_Verb_Val = 1
                
        if Prev_Report_Verb_Val or Next_Report_Verb_Val:
            Reporte_Verb_Val = 1
        else: 
            Report_Verb_Val = 0
            
        Word_Features.append(Report_Verb_Val)
        
        ## Feature 17: Entailment (1 or 0)
        if Word in Entailments:
            Entailment_Val = 1
        else:
            Entailment_Val = 0
        Word_Features.append(Entailment_Val)

        ## Feature 18: Entailment Context i.e. if Entailment is present in the context (1 or 0)
        Prev_Entailment_Val = 0
        if Prev_Word in Entailments:
            Prev_Entailment_Val = 1
        
        Next_Entailment_Val = 0
        if Next_Word in Entailments:
            Next_Entailment_Val = 1
                
        if Prev_Entailment_Val or Next_Entailment_Val:
            Entailment_Val = 1
        else: 
            Entailment_Val = 0
            
        Word_Features.append(Entailment_Val)
        
        ## Feature 19: Strong Subjective (1 or 0)
        if Word in Strong_Subjectives:
            Strong_Subjective_Val = 1
        else:
            Strong_Subjective_Val = 0
        Word_Features.append(Strong_Subjective_Val)

        ## Feature 20: Strong Subjective Context i.e. if Strong Subjective is present in the context (1 or 0)
        Prev_Strong_Subjective_Val = 0
        if Prev_Word in Strong_Subjectives:
            Prev_Strong_Subjective_Val = 1
        
        Next_Strong_Subjective_Val = 0
        if Next_Word in Strong_Subjectives:
            Next_Strong_Subjective_Val = 1
                
        if Prev_Strong_Subjective_Val or Next_Strong_Subjective_Val:
            Strong_Subjective_Val = 1
        else: 
            Strong_Subjective_Val = 0
            
        Word_Features.append(Strong_Subjective_Val)
        
        ## Feature 21: Weak Subjective (1 or 0)
        if Word in Weak_Subjectives:
            Weak_Subjective_Val = 1
        else:
            Weak_Subjective_Val = 0
        Word_Features.append(Weak_Subjective_Val)

        ## Feature 22: Weak Subjective Context i.e. if Weak Subjective is present in the context (1 or 0)
        Prev_Weak_Subjective_Val = 0
        if Prev_Word in Weak_Subjectives:
            Prev_Weak_Subjective_Val = 1
        
        Next_Weak_Subjective_Val = 0
        if Next_Word in Weak_Subjectives:
            Next_Weak_Subjective_Val = 1
                
        if Prev_Weak_Subjective_Val or Next_Weak_Subjective_Val:
            Weak_Subjective_Val = 1
        else: 
            Weak_Subjective_Val = 0
            
        Word_Features.append(Weak_Subjective_Val)
        
        ## Feature 23: Polarity of the word (0, 1, 2) 
        Polarity_Val = Get_Polarity(Word)
        Word_Features.append(Polarity_Val)
        
        ## Feature 24: Positive Word (1 or 0)
        if Word in Positive_Words:
            Positive_Word_Val = 1
        else:
            Positive_Word_Val = 0
        Word_Features.append(Positive_Word_Val)

        ## Feature 25: Positive Word Context i.e. if Positive Word is present in the context (1 or 0)
        Prev_Positive_Word_Val = 0
        if Prev_Word in Positive_Words:
            Prev_Positive_Word_Val = 1
        
        Next_Positive_Word_Val = 0
        if Next_Word in Positive_Words:
            Next_Positive_Word_Val = 1
                
        if Prev_Positive_Word_Val or Next_Positive_Word_Val:
            Positive_Word_Val = 1
        else: 
            Positive_Word_Val = 0
            
        Word_Features.append(Positive_Word_Val)
        
        ## Feature 26: Negative Word (1 or 0)
        if Word in Negative_Words:
            Negative_Word_Val = 1
        else:
            Negative_Word_Val = 0
        Word_Features.append(Negative_Word_Val)

        ## Feature 27: Negative Word Context i.e. if Negative Word is present in the context (1 or 0)
        Prev_Negative_Word_Val = 0
        if Prev_Word in Negative_Words:
            Prev_Negative_Word_Val = 1
        
        Next_Negative_Word_Val = 0
        if Next_Word in Negative_Words:
            Next_Negative_Word_Val = 1
                
        if Prev_Negative_Word_Val or Next_Negative_Word_Val:
            Negative_Word_Val = 1
        else: 
            Negative_Word_Val = 0
            
        Word_Features.append(Negative_Word_Val)
        
        ## Feature 28: Bias Lexicon (1 or 0)
        if Word in Bias_Lexicons:
            Negative_Word_Val = 1
        else:
            Negative_Word_Val = 0
        Word_Features.append(Negative_Word_Val)
        
        
        ## Add Word feature vector to Sentence_Features
        Sentence_Features.append(Word_Features)
        
    Sentence_Features_DF = pd.DataFrame(Sentence_Features, columns = Feature_Columns)
    return Sentence_Features_DF

Load_Lexicons()


## Sample Call to the function

# from Linguistic_Features import*
# Sent_DF = Get_Sent_Linguistic_Features("it was rather unfortunate that he vehemently opposed the budding indian scientist subrahmanyan chandrasekhar about his theory on the maximum mass of stars known as white dwarfs, the mass above which the star collapses and becomes a neutron star, quark star or black hole.")