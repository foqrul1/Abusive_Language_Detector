from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import xlsxwriter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import *
import csv
import tsv


print('\n\n\t\tScraping Data to text file...\n\n')
data = open('E:/Freelancing Project/abusive/Abusive_Language_Detector/Abusive_Language_Detector/data/text.txt','r',encoding='utf8').read()
data = data.replace('।','.')
print('Scraped Data: ',data)

blob = TextBlob(data)

workbook = xlsxwriter.Workbook('../output/output.xlsx')
worksheet = workbook.add_worksheet()

writer = tsv.TsvWriter(open("../data/test.tsv", "w"))

writer.line("test_id","comment")
worksheet.write(0,0, 'Content ID')
worksheet.write(0,1, 'Bangla Text')
worksheet.write(0,2, 'English Text')
worksheet.write(0,3, 'Prediction')

print('\n\n\t\tSplitting Data to .tsv file...\n\n')

i=1
print('For each sentence in paragraph:\n\n')
for sentence in blob.sentences:
    en = str(sentence.translate(to='en'))
    print(i,'. ',sentence,' - ',en)
    worksheet.write(i,0,i)
    worksheet.write(i,1,str(sentence))
    worksheet.write(i,2,en)
    
    writer.line(i,en) #tsv

    i += 1

writer.close()

print('\n\t\tTest file ready!(TSV format)\n\n')

print('\n\n\t\tTraining Data...please wait...\n\n')
# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('clf', MultinomialNB()),])
text_clf = text_clf.fit(df_train.comment, df_train.label)

######## Training complete ########
print('\n\n\t\tTraining Completed!\n\n')

predicted = text_clf.predict(df_train.comment)
# metrics on training data
print('\n\n\t\tMetrics on training data:\n\n')
print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
print('precision : {0}'.format(precision_score(df_train.label, predicted)))
print('recall : {0}'.format(recall_score(df_train.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))


print('\n\n\t\tGetting prediction...')
predicted = text_clf.predict(df_test.comment)

j=1
# writing results to a file
for item in predicted:
    if(item == 1):
        worksheet.write(j,3,'Abusive')
    elif(item == 0):
        worksheet.write(j,3,'Safe Sentence')
    j += 1
    if(i == j):
        break


workbook.close()
print('\n\n\t\tProcess Completed - File is ready to open!')
