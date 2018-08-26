# author: Kris Bukovi
# last modified: August 26, 2018
# purpose: Given 2 PDF documents, calculate the document similarity using the Universal Sentence Encoder
# 
#   

import tensorflow_hub as hub 

def importDoc():
    # convert pdf to txt file


    # open document 
    

    # read document
    
    # store text in a df 

    # return df  

#clean data
def (df):
    # go through text split where ending puncutation occurs to obtain sentences, store in vector

    # turn all letters in sentences to lower case

    # return vector


def createVector(v):

    # create an Universal Sentence Encoder object 
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    # 
    embeddings = embed(v)

    print session.run(embeddings)



    # return vector

# calculate distance between two vectors
def calculateDistance(v1, v2):





