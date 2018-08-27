# Document Similarity
Current methods for document similarity all include processing the words in the document to change their representation into that of a vector. The distance between these two vectors is then calculated using one of the many difference equations.

In my first attempt I will be using the Universal Sentence Encoder (research paper found here: https://arxiv.org/pdf/1803.11175.pdf) to create a fixed dimensional embedding representation of the sentences in the document to be processed. To process the pdfs to be text files I will be using PDFminer.six, the version of PDFminer compatible with Python 3. (**note: had to run sed $'s/\r//' -i $(which pdf2txt.py) to fix error in pdf2txt.py file**)

