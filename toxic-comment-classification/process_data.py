import pickle
import re
import numpy as np
import pandas as pd
from collections import Counter
from string import punctuation
from sklearn.model_selection import train_test_split


#Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers = re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    text = special_character_removal.sub('',text)
    
    #Replace Numbers
    text = replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


def main(comments, y):
    # =========== prepare word dict ================
    word_cnt = Counter()
    for doc in comments:
        doc_cnt = Counter(w for w in doc.split())
        word_cnt.update(doc_cnt)

    word2id = {'<PAD>': 0, '<UNK>': 1}
    for w, c in word_cnt.most_common():
        if c > 30:
            word2id[w] = len(word2id)

    id2word = {i: w for w, i in word2id.items()}

    # =========== prepare word embedding ================
    with open("./crawl-300d-2M.pkl", 'rb') as f:
        # load word2vec
        fb_w2v = pickle.load(f)

    emb_weights = np.zeros((len(word2id), 300))
    print('Embedding Shape', emb_weights.shape)

    cnt = 0
    for w, i in word2id.items():
        embedding_vector = fb_w2v.get(w)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            emb_weights[i] = embedding_vector
            cnt += 1
    print('Words with pre-trained embedding: {}'.format(cnt))

    # =========== prepare train and val data ================
    inputs = []
    for doc in comments:
        inputs.append([word2id.get(w, 1) for w in doc.split()])

    x_train, x_val, y_train, y_val = train_test_split(inputs, y, test_size=0.07, random_state=42)

    # =========== saving data ================
    process_data = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 
                    'word2id': word2id, 'id2word': id2word}
    with open('./process_data.pkl', 'wb') as f:
        pickle.dump(process_data, f)
    
    np.save('EmbeddingMatrix.npy', emb_weights)


if __name__ == '__main__':
    train_df = pd.read_csv('./train.csv')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    comments = []
    for text in list_sentences_train:
        comments.append(text_to_wordlist(text))

    # test_df = pd.read_csv('./test.csv')
    # list_sentences_test = test_df["comment_text"].fillna("NA").values
    # test_comments = []
    # for text in list_sentences_test:
    #     test_comments.append(text_to_wordlist(text))

    main(comments, y)