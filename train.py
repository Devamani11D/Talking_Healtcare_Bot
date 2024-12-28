import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np

nltk.download('wordnet')
from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

symptomsFile=json.loads(open("D:\Desktop_top\deeplearning\dl_project1\intents.json").read())

words=[]
classes=[]
documents=[]

ignoreLetters=["!",".","..",",","?"]

# print(symptomsFile["intents"][0]["tag"])

for intent in symptomsFile["intents"]:
    for pattern in intent["patterns"]:
        word_list=tokenizer.tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words=[lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]

words=sorted(set(words))
classes=sorted(set(classes))

pickle.dump(words,open('D:\Desktop_top\deeplearning\dl_project1\words.pkl','wb'))
pickle.dump(classes,open('D:\Desktop_top\deeplearning\dl_project1\classes.pkl','wb'))

data=[]
template=[0]*len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(template)
    output_row[classes.index(document[1])]=1
    print(len(bag),len(output_row))
    data.append([bag,output_row])


random.shuffle(data)
data=np.array(data,dtype="object")

x_train=list(data[:,0])
y_train=list(data[:,1])
    
# model 

model=Sequential()

model.add(Dense(256,input_shape=(len(x_train[0]),),activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(y_train[0]),activation='softmax'))

sgd=SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist=model.fit(np.array(x_train),np.array(y_train),epochs=200,batch_size=5,verbose=1)

model.save("D:\Desktop_top\deeplearning\dl_project1\chatbotModel.h5",hist)

print("Training done!")