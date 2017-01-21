import openpyxl
import string
import enchant
import time
import re
from nltk.stem.porter import *
import test_v5 as PosNeg

def main():
    # Start time taken to execute code
    start_time = time.time()
    
    # Fetch Data from the Excel file
    wb = openpyxl.load_workbook('training-Obama-Romney-tweets.xlsx')
    sheetsinfile = wb.get_sheet_names()
    
    tweets = []
    t_scores = []

    for i in range(0,len(sheetsinfile)):
        if i<2:
            sheet = wb.get_sheet_by_name(sheetsinfile[i])
            l1 = (list(sheet.columns)[3])
            l2 = (list(sheet.columns)[4])
            for j in range(0,len(l1)):
                if j!=0 and j!=1:
                    tweets.append((l1[j]).value)
                    t_scores.append((l2[j]).value)
        else:
            break
    o_tweets = []
    o_t_scores = []         
    o_sheet = wb.get_sheet_by_name(sheetsinfile[0])
    o_tweets, o_t_scores = fetch_tweet_and_scores(o_tweets, o_t_scores, (list(o_sheet.columns)[3]), (list(o_sheet.columns)[4]))
    
    r_tweets = []
    r_t_scores = []     
    r_sheet = wb.get_sheet_by_name(sheetsinfile[1])
    r_tweets, r_t_scores = fetch_tweet_and_scores(r_tweets, r_t_scores, (list(r_sheet.columns)[3]), (list(r_sheet.columns)[4]))
    
    all_tweets_clean = []
    all_tscore_clean = []
    otweets_clean = []
    otscore_clean = []
    rtweets_clean = []
    rtscore_clean = []
    
    #all_tweets_clean, all_tscore_clean = clean_tweets_and_scores(all_tweets_clean,all_tscore_clean, tweets, t_scores)
    otweets_clean, otscore_clean, onos = clean_tweets_and_scores(otweets_clean, otscore_clean, o_tweets, o_t_scores)
    rtweets_clean, rtscore_clean, rnos = clean_tweets_and_scores(rtweets_clean, rtscore_clean, r_tweets, r_t_scores)
    
    pos,neg,nu = PosNeg.posneg()
    
    otscore_clean = lexi(otweets_clean, otscore_clean, pos, neg, nu)
    
    rtscore_clean = lexi(rtweets_clean, rtscore_clean, pos, neg, nu)
    
    with open("oclean.txt", "w") as text_file:
        for i in range(0,len(otweets_clean)):
            for j in range(0, len(otweets_clean[i])):
                if j!=len(otweets_clean[i])-1:
                    print(otweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(otweets_clean[i][j])+"\t",end="",file=text_file)

            print(otscore_clean[i],file=text_file)
            
    with open("rclean.txt", "w") as text_file:
        for i in range(0,len(rtweets_clean)):
            for j in range(0, len(rtweets_clean[i])):
                if j!=len(rtweets_clean[i])-1:
                    print(rtweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(rtweets_clean[i][j])+"\t",end="",file=text_file)

            print(rtscore_clean[i],file=text_file)
            
    with open("oclean2.txt", "w") as text_file:
        for i in range(0,len(otweets_clean)):
            for j in range(0, len(otweets_clean[i])):
                if j!=len(otweets_clean[i])-1:
                    print(otweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(otweets_clean[i][j])+"\t",end="",file=text_file)

            print(otscore_clean[i],file=text_file)
            
    with open("rclean2.txt", "w") as text_file:
        for i in range(0,len(rtweets_clean)):
            for j in range(0, len(rtweets_clean[i])):
                if j!=len(rtweets_clean[i])-1:
                    print(rtweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(rtweets_clean[i][j])+"\t",end="",file=text_file)

            print(rtscore_clean[i],file=text_file)
            
    with open("onos.txt", "w") as text_file:
        for i in range(0,len(onos)):
            for j in range(0, len(onos[i])):
                if j!=len(onos[i])-1:
                    print(onos[i][j],'',end="",file=text_file)
                else:
                    print(str(onos[i][j])+"\n",end="",file=text_file)
    
    with open("rnos.txt", "w") as text_file:
        for i in range(0,len(rnos)):
            for j in range(0, len(rnos[i])):
                if j!=len(rnos[i])-1:
                    print(rnos[i][j],'',end="",file=text_file)
                else:
                    print(str(rnos[i][j])+"\n",end="",file=text_file)
                    
    flare = do_for_test()
    
    if flare == 1:
        print ("MISSION ACCOMPLISHED!!")
            
    print("--- %s seconds ---" % (time.time() - start_time))
    
def do_for_test():
    # Fetch Data from the Excel file
    wb = openpyxl.load_workbook('testing-Obama-Romney-tweets.xlsx')
    sheetsinfile = wb.get_sheet_names()
    
    tweets = []
    t_scores = []

    for i in range(0,len(sheetsinfile)):
        if i<2:
            sheet = wb.get_sheet_by_name(sheetsinfile[i])
            l1 = (list(sheet.columns)[0])
            l2 = (list(sheet.columns)[4])
            for j in range(0,len(l1)):
                if j!=0 and j!=1:
                    tweets.append((l1[j]).value)
                    t_scores.append((l2[j]).value)
        else:
            break
    o_tweets = []
    o_t_scores = []         
    o_sheet = wb.get_sheet_by_name(sheetsinfile[0])
    print (o_sheet)
    o_tweets, o_t_scores = fetch_tweet_and_scores(o_tweets, o_t_scores, (list(o_sheet.columns)[0]), (list(o_sheet.columns)[4]))
    
    r_tweets = []
    r_t_scores = []     
    r_sheet = wb.get_sheet_by_name(sheetsinfile[1])
    print (r_sheet)
    r_tweets, r_t_scores = fetch_tweet_and_scores(r_tweets, r_t_scores, (list(r_sheet.columns)[0]), (list(r_sheet.columns)[4]))
    
    all_tweets_clean = []
    all_tscore_clean = []
    otweets_clean = []
    otscore_clean = []
    rtweets_clean = []
    rtscore_clean = []
    
    #all_tweets_clean, all_tscore_clean = clean_tweets_and_scores(all_tweets_clean,all_tscore_clean, tweets, t_scores)
    otweets_clean, otscore_clean, onos = clean_tweets_and_scores(otweets_clean, otscore_clean, o_tweets, o_t_scores)
    rtweets_clean, rtscore_clean, rnos = clean_tweets_and_scores(rtweets_clean, rtscore_clean, r_tweets, r_t_scores)
    
    with open("oclean_test.txt", "w") as text_file:
        for i in range(0,len(otweets_clean)):
            for j in range(0, len(otweets_clean[i])):
                if j!=len(otweets_clean[i])-1:
                    print(otweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(otweets_clean[i][j])+"\t",end="",file=text_file)

            print(otscore_clean[i],file=text_file)
            
    with open("rclean_test.txt", "w") as text_file:
        for i in range(0,len(rtweets_clean)):
            for j in range(0, len(rtweets_clean[i])):
                if j!=len(rtweets_clean[i])-1:
                    print(rtweets_clean[i][j],'',end="",file=text_file)
                else:
                    print(str(rtweets_clean[i][j])+"\t",end="",file=text_file)

            print(rtscore_clean[i],file=text_file)
            
    return 1
    
def lexi(tw,sc,po,ne,nu):
    sumall = []
    for i in tw:
        sum_ = [0,0,0]
        if (type(i) != type(0.0)):
            for j in i:
                if j in po:
                    sum_[0] = sum_[0] + 1
                if j in ne:
                    sum_[1] = sum_[1] + 1
                if j in nu:
                    sum_[2] = sum_[2] + 1
        sumall.append(sum_)
    for i in range(0,len(sc)):
        maxi = max(sumall[i])
        if maxi !=0:
            if (sumall[i]).index(maxi) == 0 and (sc[i] == -1 or sc[i] == 0) and maxi >=3 and sumall[i][0]-sumall[i][1] >= 3 and sumall[i][0]-sumall[i][2] >= 3:
                sc[i] = 1
            if (sumall[i]).index(maxi) == 1 and (sc[i] == 1 or sc[i] == 0) and maxi >=3 and sumall[i][1]-sumall[i][0] >= 2 and sumall[i][1]-sumall[i][2] >= 2:
                sc[i] = -1
            if (sumall[i]).index(maxi) == 2 and (sc[i] == 1 or sc[i] == -1) and maxi >=3 and sumall[i][2]-sumall[i][1] >= 2 and sumall[i][2]-sumall[i][0] >= 2:
                sc[i] = 0
    return sc
    
def clean_tweets_and_scores(finaltweets, finalscores, tweets, t_scores):
    no_senti = []
    if len(tweets)==len(t_scores):
        for i in range(0,len(tweets)):
            if tweets[i]!=None and t_scores[i]!=None:
                if t_scores[i]==0 or t_scores[i]==1 or t_scores[i]==-1:
                    finaltweets.append(tweets[i])
                    finalscores.append(t_scores[i])
                else:
                    no_senti.append(tweets[i])
        
        
        clean_final = do_clean(finaltweets)
        
        no_senti_final = do_clean(no_senti)
        
        return clean_final, finalscores, no_senti_final
    
    else:
        print ("SOMETHING WENT WORNG")
        return [],[]
    
    
def do_clean(finaltweets):
    stemmer = PorterStemmer()
    stop=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'rt']
    clean_final = []
    for i in finaltweets:
        i=cleanhtml(i)
        ts=i.split(" ")
        for w in range(0,len(ts)):
            if "\t" in ts[w]:
                ts[w] = ts[w].replace("\t","")
            ts[w]=ts[w].lower()
            if (ts[w]!='' and ts[w][0]=='@'):
                ts[w]='USER'
            if (ts[w]!='' and len(ts[w])>4  and ts[w][0]=='w' and ts[w][1]=='w' and ts[w][2]=='w' and ts[w][3]=='.'):
                ts[w]='URL'
            if (ts[w]!='' and len(ts[w])>7  and ts[w][0]=='h' and ts[w][1]=='t' and ts[w][2]=='t' and ts[w][3]=='p' and
                ts[w][4]==':' and ts[w][5]=='/' and ts[w][6]=='/'):
                ts[w]='URL'
            if (ts[w]!='' and len(ts[w])>8  and ts[w][0]=='h' and ts[w][1]=='t' and ts[w][2]=='t' and ts[w][3]=='p' and
                ts[w][4]=='s' and ts[w][5]==':' and ts[w][6]=='/' and ts[w][7]=='/'):
                ts[w]='URL'
            for i in range(0,len(ts[w])):
                flag=0
                if len(ts[w])>i+2 and ts[w][i]==ts[w][i+1] and ts[w][i]==ts[w][i+2]:
                    for j in range(i+2,len(ts[w])):
                        if ts[w][i]==ts[w][j]:
                            flag=1
                            if len(ts[w])>j+1 and ts[w][i]!=ts[w][j+1]:
                                break
                        if j==(len(ts[w]))-1 and ts[w][i]==ts[w][j]:
                            flag=1
                            break
                    if flag==1:
                        ts[w]=ts[w].replace(ts[w][i:j+1],ts[w][i])
            if (ts[w]!='' and ts[w][0].isdigit()):
                ts[w]=''
            if 'z' in ts[w]:
                ts[w]=ts[w].replace('z','s')
            if "'" in ts[w]:
                ts[w] = ts[w].replace("'", "")
            if '"' in ts[w]:
                ts[w] = ts[w].replace('"', '')
            ts[w] = ts[w].rstrip()
            exclude = set(string.punctuation)
            ts[w] = ''.join(ch for ch in ts[w] if ch not in exclude)
            #ts[w] = stemmer.stem(ts[w])
                
        for i in range(0,len(stop)):
            if stop[i] in ts:
                ts = remove_values_from_list(ts, stop[i])
        cleaned_list=[word.strip(string.punctuation) for word in ts]
        while '' in cleaned_list:
            cleaned_list.remove('')
        clean_final.append(cleaned_list)
    return clean_final
    
def dospellcheck(ts):
    d=enchant.Dict("en_US")
    if (len(ts))>3 and ts!='' and (d.check(ts))==False:
        sug=d.suggest(ts)
        if len(sug)>=3:
            for i in range(0, len(sug)):
                if (len(ts) == len(sug[i]) or len(ts) == len(sug[i])+1 or len(ts) == len(sug[i])-1):
                    dist=edit(ts,sug[i])
                    if dist<3:
                        return sug[i]
    return ts

def edit(s1, s2):
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]
    

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

    
def fetch_tweet_and_scores(t_list, s_list, tweet_list, score_list):
    for j in range(0,len(tweet_list)):
        if j!=0 and j!=1:
            t_list.append((tweet_list[j]).value)
    for k in range(0,len(score_list)):
        if k!=0 and k!=1:
            s_list.append((score_list[k]).value)
    return t_list, s_list


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
    
            
if __name__ == '__main__':
    main()
    