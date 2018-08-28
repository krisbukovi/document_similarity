# author: Kris Bukovi
# last modified: August 26, 2018
# purpose: Given 2 PDF documents, calculate the document similarity using the Universal Sentence Encoder

import io
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd 
import re 
import seaborn as sns 

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from nltk.tokenize import sent_tokenize, word_tokenize

class SimilarityMap:

    # instance variable for list of paths to each pdf
    def __init__(self, pdfList):

        # initialize variable for list of paths 
        self.pdfList = pdfList

    def pdf2txt(self, size):

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

        # loop through all files (can i ge rid of this loop?)
        for i in range(size):

            # open file in read mode
            with open(self.pdfList[i], 'rb') as inFile:

                for page in PDFPage.get_pages(inFile, pagenos, maxpages=maxpages, password=password, caching = True, check_extractable = True):

                    # pdf to text
                    pg_interp.process_page(page)

                # store text from pdf in variable 
                text = pdf.getvalue()

            # close converter and io file object
            inFile.close()
            converter.close()
            pdf.close()

            if text:
                txtList.append(text)
            else:
                print("file %i is empty", i)

            return txtList

    # clean data
    def cleanData(self, docList):

        tempList = []

        # go through text split where ending puncutation occurs to obtain sentences, store in vector
        for doc in docList:
            tempList.append(sent_tokenize(doc))
            

        # turn all letters in sentences to lower case

        # return list
        return tempList


    def createVector(v):

        # create an Universal Sentence Encoder object 
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
 
        for sentence in v:
            embeddings = embed(v)

        print session.run(embeddings)



        # return vector

    # calculate distance between two vectors
    #def calculateDistance(v1, v2):


if __name__ == '__main__':

    # create list to store file paths for all PDFs
    paths = ["Don't cry because it's over, smile because it happened. Be yourself; everyone else is already taken. You know you're in love when you can't fall asleep because reality is finally better than your dreams."]

    # get path to first file
    #paths.append(str(input("Please enter the path for first file: ")))

    simMap = SimilarityMap(paths)

    # simMap.pdf2txt(len(paths))

    print(paths)

    print(simMap.cleanData(paths))









    





