import speech_recognition as sr
import pyttsx3
import time

import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random
import json
# nltk.download('wordnet')
from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

lemmatizer=WordNetLemmatizer()
intents=json.loads(open("D:\Desktop_top\deeplearning\dl_project1\intents.json").read())

words= pickle.load(open('D:\Desktop_top\deeplearning\dl_project1\words.pkl','rb'))
classes=pickle.load(open('D:\Desktop_top\deeplearning\dl_project1\classes.pkl','rb'))
model= load_model('D:\Desktop_top\deeplearning\dl_project1\chatbotModel.h5')

def cleanup_sentence(sentence):
    sentence_words=tokenizer.tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence):
    sentence_words=cleanup_sentence(sentence)
    bag=[0]*len(words)

    for w in sentence_words:
        for  i , word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bag=bag_of_words(sentence)
    result=model.predict(np.array([bag]))[0]
    print(f"result {result}")

    error_threshold=0.25

    results=[[i ,r ] for i,r in enumerate(result) if r>error_threshold]

    results.sort(key=lambda x:x[1],reverse=True)


    return_list=[]

    for r in results:
        return_list.append({'intent': classes[r[0]],'probability':str(r[1])})

    print(f"return list: f{return_list[0]}")
    return return_list

def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']

    result=''

    for i in list_of_intents:
        if i['tag']== tag:
            result=random.choice(i['responses'])
            break

    return result

def calling_the_bot(text):
    global result
    predict=predict_class(text)
    print(f"Prediction: f{predict}")
    result=get_response(predict,intents)

    engine.say("found it "+result)

    engine.runAndWait()


if __name__ == '__main__':
    print("Bot is Running")

    recognizer= sr.Recognizer()
    mic=sr.Microphone()

    engine=pyttsx3.init()
    rate=engine.getProperty('rate')
    print(f"rate : {rate}")
    engine.setProperty('rate',190)

    volume=engine.getProperty('volume')
    engine.setProperty('volume',0.5)

    voices=engine.getProperty('voices')

    engine.say("Hello USER, I am your personal Talking HealthCare BOT.")

    engine.runAndWait()

    engine.say(
        "IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE SAY MALE.\
        OTHERWISE SAY FEMALE.")
    engine.runAndWait()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source,duration=0.1)
        audio=recognizer.listen(source)

    audio=recognizer.recognize_google(audio)

    print(audio)

    if audio=="Female".lower():
        engine.setProperty('voice', voices[1].id)
        print("You have chosen to continue with Female Voice")
    else: 
        engine.setProperty('voice',voices[0].id)
        print("You have chosen to continue with Male Voice")

    while True or final.lower()=='True':
        with mic as symptom:
            print("Say your Symptoms..")
            engine.say("You may tell me your symptoms now...")
            engine.runAndWait()

            try:
                recognizer.adjust_for_ambient_noise(symptom,duration=0.1)
                symp=recognizer.listen(symptom)
                text=recognizer.recognize_google(symp)
                engine.say("You said {}".format(text))
                engine.runAndWait()
                
                time.sleep(1)

                calling_the_bot(text)

            except sr.UnknownValueError:
                engine.say("Sorry, your symptoms are unclear to me..")
                engine.runAndWait()
                print("Sorry your symptoms are unclear to me")

            finally:
                engine.say("If you want to continue please say Continue otherwise say I want to Exit")
                engine.runAndWait()

        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans,duration=0.1)
            voice=recognizer.listen(ans)
            final=recognizer.recognize_google(voice)

        if final.lower() == 'no' or final.lower()=="please exit":
            engine.say("Thank you. Shutting down now..")
            engine.runAndWait()
            print("Bot has been stopped by the user")
            exit(0)