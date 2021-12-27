import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter

df = pd.read_csv('spam.csv')
df.drop_duplicates(inplace=True)
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
# plt.pie(df['Category'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
# plt.show()
nltk.download('punkt')
df['mail_len_charac'] = df['Message'].apply(len)
df['mail_len_words'] = df['Message'].apply(lambda x: len(nltk.word_tokenize(x)))
df['mail_len_sentences'] = df['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))


# plt.figure(figsize=(12,8))
# sns.histplot(df[df['Category'] == 0]['mail_len_charac'])
# sns.histplot(df[df['Category'] == 1]['mail_len_charac'],color='red')
# plt.show()

# plt.figure(figsize=(12,8))
# sns.histplot(df[df['Category'] == 0]['mail_len_words'])
# sns.histplot(df[df['Category'] == 1]['mail_len_words'],color='red')
# plt.show()

# sns.pairplot(df,hue = 'Category')
# plt.show()

# sns.heatmap(df.corr(),annot = True)

def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


df['mail_transformed'] = df['Message'].apply(transform_text)
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(df[df['Category'] == 1]['mail_transformed'].str.cat(sep=" "))
ham_wc = wc.generate(df[df['Category'] == 0]['mail_transformed'].str.cat(sep=" "))
#plt.imshow(spam_wc)
#plt.show()
#plt.imshow(ham_wc)
#plt.show()
spam_corpus = []
for mail in df[df['Category'] == 1]['mail_transformed'].tolist():
    for word in mail.split():
        spam_corpus.append(word)
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
#plt.xticks(rotation = 'vertical')
#plt.show()
ham_corpus = []
for mail in df[df['Category'] == 0]['mail_transformed'].tolist():
    for word in mail.split():
        ham_corpus.append(word)
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0], pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
#plt.xticks(rotation = 'vertical')
#plt.show()
cv = CountVectorizer()
#tfidf = TfidfVectorizer() improvement of features
tfidf = TfidfVectorizer(max_features=3000)
#x = cv.fit_transform(df['mail_transformed']).toarray()
x = tfidf.fit_transform(df['mail_transformed']).toarray()
y = df['Category'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
#gnb.fit(x_train, y_train)
#y_pred1 = gnb.predict(x_test)
#print(accuracy_score(y_test, y_pred1))
#print(confusion_matrix(y_test,y_pred1))
#print(precision_score(y_test,y_pred1))
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
#bnb.fit(x_train, y_train)
#y_pred3 = bnb.predict(x_test)
#print(accuracy_score(y_test, y_pred3))
#print(confusion_matrix(y_test,y_pred3))
#print(precision_score(y_test,y_pred3))

# tfidf  --> MNB
svc = SVC(kernel ='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver=50, random_state=2)
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'xgb' : xgb
}

def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    return accuracy, precision

train_classifier(svc,x_train,y_train,x_test,y_test)

accuracy_score = []
precision_score = []

for name,clf in clfs.items():

    current_accuracy, current_precision = train_classifier(clf, x_train, y_train, x_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("precision - ", current_precision)

    accuracy_score.append(current_accuracy)
    precision_score.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(), 'Accuracy':accuracy_score, 'Precision':precision_score}).sort_values('Accuracy', ascending=False)

performance_df

sns.catplot(x = 'Algorithm', y='value',
            hue = 'variable', data=performance_df, kind='bar', height=5)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()

##model improve
#-change the max features parameters of tfIdf

temp_df = pd.DataFrame({'Algorithm':clfs.keys(), 'Accuracy_max_ft_3000':accuracy_score, 'Precision_max_ft_3000':precision_score}).sort_values('Accuracy', ascending=False)

performance_df.merge(temp_df, on='Algorithm')

scaler = MinMaxScaler()
x = scaler.fit_trasform(x)

y = df['target'].values

X = tfidf.fit_transform(df['transformed_text']).toarray()

##appending the num_chatracter col to X
X = np.hstack((X, df['num_characters'].values.reshape(-1, 1)))

y = df['target'].values


import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
