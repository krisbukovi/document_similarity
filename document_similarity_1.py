# author: Kris Bukovi
# last modified: September 12, 2018
# purpose: Given a PDF document, calculate the nearest n documents 

import io
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd 
import re 
import seaborn as sns 
import pdfminer
import csv
import string

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords

# hide info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# global constant variable for path to tensorboard files
LOG_DIR = 'graphs'

OUTPUT = []

class SimilarityMap:

    # instance variable for list of paths to each pdf
    def __init__(self, pdfList):

        # initialize variable for list of paths 
        self.pdfList = pdfList

    def pdf_to_txt(self, tempPaths):

        txtList = []

        # create instance of pdf resource manager
        rm = PDFResourceManager()

        # create an io file object (file function doesn't work here with pdfminer.six)
        pdf = io.StringIO()

        codec = 'utf-8'

        laparams = LAParams()

        # create text converter
        converter = TextConverter(rm, pdf, codec=codec, laparams=laparams)

        # create instance of page interpreter 
        pg_interp = PDFPageInterpreter(rm, converter)

        password = ""

        maxpages = 0

        pagenos = set()

        # loop through all files
        for i in range(len(tempPaths)):

            # open file in read mode
            with open(tempPaths[i], 'rb') as inFile:

                for page in PDFPage.get_pages(inFile, pagenos, maxpages=maxpages, password=password, caching = True, check_extractable = True):

                    # pdf to text
                    pg_interp.process_page(page)

                # store text from pdf in variable 
                text = pdf.getvalue()

            # close converter and io file object
            inFile.close()

            if text:
                txtList.append(text)
            else:
                print("file %i is empty", i)
        
        converter.close()
        pdf.close()

        return txtList

    # clean data
    def cleanData(self, docList):

        tempList = []

        # go through text split where ending puncutation occurs to obtain sentences, store in vector
        for doc in docList:

            # remove stopwords
            stop_words = set(stopwords.words('english'))

            # split document into sentences
            # turn all letters in sentences to lower case
            sentences = [s.strip() for s in sent_tokenize(doc)]

            processed_doc = []

            for sent in sentences:

                temp_sent = []

                # remove dashes from end of pages, as well as et al  
                sent1 = sent.replace("-  ", "")
                sent2 = sent1.replace("- ", "")
                sent3 = sent2.replace("-", "")
                sent4 = sent3.replace("et al.", "")

                # split sentence into words 
                tokens = [w.lower() for w in word_tokenize(sent4)]

                for word in tokens:
                    
                    # create list without stop words, punctuation, numbers and words shorter than 4 letters
                    if (word not in stop_words) and (word not in string.punctuation) and (word.isalpha()) and (len(word) > 3):

                        temp_sent.append(word)

                # join all words back together into a sentence
                final_sent = " ".join(temp_sent)

                """ print(final_sent)
                print("--------------------------------------------------------") """

                # append processed sentence to list representing the document
                processed_doc.append(final_sent)
            

            # add processed document to list 
            tempList.append(processed_doc)

        # create a tsv file with all sentences for labeling
        with open(LOG_DIR + '/labels.tsv', "w") as tempFile:
            #tsv_out = csv.writer(tempFile, delimiter='\t')
            #tsv_out.writerow(tempList[1])

            tempFile.write('/t'.join(tempList[2]))

        # return list
        return tempList


    def createVector_use(self, tempVec):

        # create list object
        embeddings = []
        ave = []
        output = []

        # create an Universal Sentence Encoder object 
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
 
        for i in range(len(tempVec)):

            embeddings.append(embed(tempVec[i]))

            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                mdVector = session.run(embeddings[i])    
                
            # calculate the average of all sentence vectors for each document
            #ave.append(np.reshape(np.mean(mdVector, axis=0), (1, 512)))
            ave.append(mdVector)

            #ave = tf.convert_to_tensor(ave)

            if i == 1:
                # grab vectors for temporary tensorboard display from second doc in list
                OUTPUT.append(mdVector)

        # return vector
        return ave

    def createVector_doc2vec(self, tempDoc):

        # hyperparamters (need to adjust)
        max_epochs = 100
        vec_size = 50
        alpha = 0.025

        # create instance of Doc2Vec model 
        # PV-DM model
        model = Doc2Vec(size=vec_size,
                        alpha=alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)
        
        # build model with training based on tempDoc (need to train on more than one doc)
        model.build_vocab(tempDoc)

        # cycle through number of epochs 
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tempDoc,
            total_examples=model.corpus_count,
            epochs=model.iter)

            # decrease the learning rate for each step in gradient descent 
            model.alpha -= 0.0002

            # [fix the learning rate, no decay]
            # not sure what this does
            # model.min_alpha = model.alpha

        # save model
        model.save("d2v.model")

        # load model
        model= Doc2Vec.load("d2v.model")

        # to find the vector of a document which is not in training data
        temp_paths = []
        temp_paths.append("t_sne.pdf")

        sm = SimilarityMap(temp_paths)

        test_docs = sm.pdf_to_txt(temp_paths)

        v1 = model.infer_vector(test_docs)

        tensor_vec = tf.reshape(tf.convert_to_tensor(v1), (50, 1))

        print("V1_infer", tensor_vec)

        """ print("MOST SIMILAR DOC")
        # to find most similar doc using tags
        similar_doc = model.docvecs.most_similar('1')
        print(similar_doc) """


        # print vector of document at index 1 in training data
        # print(model.docvecs['1'])

        return tensor_vec


if __name__ == '__main__':

    # create list to store file paths for all PDFs
    paths = []

    # create a list to store text of files after conversion
    textList = []

    paths.append("universal_sentence_encoder.pdf")
    paths.append("barnes_hut_sne.pdf")
    paths.append("abstractive_text_summarization.pdf")

    for p in paths:
        # create a tsv file with all file names for labeling
        with open(LOG_DIR + '/labels.tsv', 'w', newline='') as tempFile:
            tsv_out = csv.writer(tempFile, delimiter='\t')
            tsv_out.writerow(p) 

    # create simMap object
    simMap = SimilarityMap(paths)

    # store text list with processed pdf data
    textList = simMap.pdf_to_txt(paths)

    # process the text list from file
    clean_data = simMap.cleanData(textList)



    # Universal Sentence Encoder Model Implementation #

    # get 512 dimension vectors representing each of the documents
    use_output = simMap.createVector_use(clean_data)
    
    #print(use_output)



    # Doc2Vec Model implementation #
    
    full_text = []

    for d in clean_data:

        full_text.append(" ".join(d))

    tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[paths[i]]) for i, _d in enumerate(full_text)]

    d2v_output = simMap.createVector_doc2vec(tagged_data)

    sentence_test_output = tf.convert_to_tensor(np.reshape(OUTPUT, (582, 512) ))

    print("what shape?")
    print(sentence_test_output)


    # Visualize in Tensorboard
    with tf.Session() as sess:

        # create path to labels/metadata 
        #metadata = os.path.join('labels.tsv')

        # create embedding variable for output
        embedding_var = tf.Variable(sentence_test_output, name="document_similarity")

        # initialize session
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # store metadata path in variable
        #embedding.metadata_path = metadata

        # create writer and projector
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        projector.visualize_embeddings(writer, config)

        # saver for checkpoints
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(LOG_DIR, "output.ckpt"))

    









    





