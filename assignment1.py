## This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import os
import re
import numpy as np
import seaborn as sns
import math
nltk.download('stopwords')
from nltk.corputs import stopwords
from pandas.config import reser_option
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import dot
from numpy.linalg import norm
# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def read_file(path):
    with open(path) as data_file:
        data = json.load(data_file)
        return data
    
def score_convert(score):
    score = score.split(",")
    left = int(scort[0])
    right = int(score[1])
    return left+right

def box_plot(data):
    fig1,ax1 = plt.subplots()
    ax1.set_title("boxplot for total goals")
    ax1.set_xlabel("total goals")
    ax1.boxplot(data,filerprops = dict(markerfacecolor = "g", marker = "0"), whis = 1.5)
    plt.savefig("task4.png")
    
def bar_plot(data):
    fig1,ax1 = plt.subplots(figsize =(9,6))
    ax1.set_title("barplot for total goals")
    ax1.set_xlabel("club_name")
    ax1.set_ylabel("number_of_mentions")
    ax1.bar(data ["club_name"],data["number_of_mentions"])   
    fig1.autofmt_xdate(rotation = 60)
    plt.savefig("task5.png")
    
def heat_map(data):
    fig1,ax1 =plt.subplots()
    ax1,set_title("Heat map for club similarity")
    ax1.set_xable("club name")
    ax1.set_ylabel("club name")
    sns.heatmap(data)
    plt.savefig("task6.png")
    
def scatter_plot(data1,data2):
    fig1,ax1 = plt.subplots()
    ax1.set_title("Scatter plot for mentioning vs score")
    ax1.set_xlabel("Number of atticles mentioning")
    ax1.set_ylabel("Total number of goal scored")
    ax1.scatter(data1,data2,alpha = 0.6)  
    plt.savefig("task7.png")       

def cosin_sim(v1,v2):
    return dot(v1,v2)/(norm(v1)* norm(v2))

def task1():
    data = read_file(datafilepath)
    return sorted(data(["teams_codes"])
    
def task2():
    data = read_file(datafilepath)
    nclubs = len(data["clubs"])
                  
    code = []
    scored = []
    against = []
    
    for i in range(nclubs):
        club = data["clubs][i]
        code.append(data["clubs"[i]["club_code])
        scored.append(data["clubs"][i]["goals_scored"])
        against.append(data["clubs][i]["goals_conceded"])
        
    df = pd.DataFrame({"team code":code,"goals scored by team":scored,"goals scored against team":against)}
    df = df.sort_values(by = ["team code"])
    df.to_csv("task2.csv", index = False)                 
    return None
      
def task3():
    filenames = os.listdir(articlespath)
    pattern = r"\b[0-9]{1,2}\-[0-9]{1,2}\b"
    filename_list = []
    total_score -values= []
    for file in filenames:
        filename_list.append(file)
        f = open(articlespath + "/" +file, "r")
        string = f.read()              
        re_result = re.findall(pattern,string)
        
        if len(re_sult) ==0:
            total_score.append(0)
        else:
            toral_score.append(max(score_convert(scort) for score in re_result]))
    
    df = pd.DataFrame({"filemname":filename_list,"total_goals":total_score})
    df = df.sort_values(by=["filename"])
    df.to_csv("task.csv",index = False                  
    return  None

def task4():
    df = pd.read_csv("task3.csv")
    box_plot(df["total_goals"])
    return None
    
def task5():
    data = read_file(datafilepath)
    clubname =[data["clubs"][i]["name"] for i in range(len(data["clubs"]))]
    d = {club:0 for club in clubname)
    flienames = os.listdir(articlespath)
     
    for file in filenames:
         f = open(articlespath +"/" +file,"r")
         for name in clubname:
            re_result = re.search(name,string,re.IGNORECASE)
            
            if result:
                d[name]+=1
         
    df = pd.DataFrame({"club_name": d.keys(),"number_of_mentions":d.values()})
    df = df.sort_values(by=[club_name"])
    df.to_csv("task5.csv",index = False)
    bar_plot(df,sort=True)          
    return None
    
def task6():
    data = read_file(datafilepath)
    clubname = [data["clubs][i]["name"] for i in range(len(data["clubs"]))]
    d = {club:[] for club in clubname}
    filenames = os.listdir(articlespath)
    for dile in filenames:
         f = open(articlespath + "/" + file,"r")
         string = f.read()
         for name in clubname:
            re_resule = re.search(name,string,re.IGNORECASE)
            if re_result:
                d[name].append(file)
                     
    for club1 in d.keys():
       clubsimilarity =[]
       for club2 in d.keys()
           len1 = len(d[club1])
           len2 = len(d[club2])        
           intersection = len(np.intersect1d(d[club1],d[club2]))
           if intersection >0:
               clubsimilarity.append(2*intersection / (len1+len2))
           else:
               clubsimilarity.append(0)
       heatmap.append(clubsimilarity)
    heat_map(heatmap)
    return None
    
def task7():
    articles = pd.read_csv("task5.csv")
    score = pd.read_csv("task2.csv")
    scatter_plot(articles["number_of_mentions"],score["goals scored by team"])
    return None
    
def task8(filename):
    f = open(filename, "r")
    string = " ".join(f.read().splitlines()).lower()
    string = re.sub(r'[^a-z]',' ',string)   
    string = " ".join(string.split())
    wordlist = nltk.word_tokenize(string)
    stopWords = list(set(stopwords.words('english')))
    filteredList = [w for w in wordlist if not w in stopWords and len(w)!= 1]
    return filteredList
    
def task9():
    corpus = []
    filenames = [file for file in os.listdir(articlespath)]  
    for file in filenames[:200 ]:
        path = articlespath + "/" +file
        corpus.append(" ".join(task8(path)))
    vectorizer = CountVectorizer()
    term_counts = vectorizer.fit_transform(corpus)
    term_counts = term_counts.toarrary()                
    
    transform = TfidfTransformer()                
    tfidf = transformer.fit_transform(term_counts)
    doc_tfidf = tfidf.toarrary()                
                     
    first_article = []
    second_article = [] 
    score = []                 
    for article_count_index in range(term_counts.shape[0]):
        q_unit = doc_tfidf[article_count_index]             
        sims = [cosin_sim(q_unit,doc_tfidf[d_id]) for d_id in range(doc_tfidf.shape[0])]           
        second_large = sorted(sims)[-2]
        index = sims.index(second_large)             
        first_article.append(filenames[article_count_index])             
        second_article.append(filenames[article_count_index]             
        score.append(second_large)             
    df = pd.DataFrame(list(zip(first_article, second_article,score)), columns = ['article1', 'article2','similarity'])          
    df.to_csv("task9.csv, index = False)                 
    return None

