# author: Kris Bukovi
# last modified: August 26, 2018
# purpose: Given 2 PDF documents, calculate the document similarity using the Universal Sentence Encoder
# 
#   

import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import tensorflow_hub as hub 

class SimilarityMap:

    def importDoc(path):

        # create instance of pdf resource manager
        rm = PDFResourceManager()

        pdf = io.StringIO

        converter = TextConverter(rm, pdf)

        pg_interp = PDFPageInterpreter(rm, converter)

        # convert pdf to txt file
        


        # open document 
        

        # read document
        
        # store text in a df 

        # return df  

    #clean data
    def cleanData(df):
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


if __name__ == '__main__':

    # create list to store file paths for all PDFs
    list = []

    # get path to first file
    s = String(input("Please enter the path for first file: "))





    





