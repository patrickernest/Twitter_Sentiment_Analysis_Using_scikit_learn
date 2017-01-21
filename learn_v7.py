import numpy as np
import pandas as pd
import sklearn.cross_validation
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
import test_v5 as PosNeg
names = ['text', 'label']
data_test_o = pd.read_table("oclean_test.txt", sep="\t", names=names)
data_test_r = pd.read_table("rclean_test.txt", sep="\t", names=names)

def test_challenge(train,test):
    
    X_train_tfidf, X_new_tfidf = vect_cal(train,test)

    model_NB = MultinomialNB().fit(X_train_tfidf,train['label'])
    
    pred_NB = model_NB.predict(X_new_tfidf)
    
    model_LR = linear_model.LogisticRegression(multi_class='multinomial', solver='sag').fit(X_train_tfidf,train['label'])
    
    pred_LR = model_LR.predict(X_new_tfidf)
    
    model_SVM = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,train['label'])
    
    pred_SVM = model_SVM.predict(X_new_tfidf)
    
    print ("NAIVE BAYES CLASSIFIER TEST RESULTS")
    
    print ("NB Test accuracy:", np.mean(pred_NB == test['label']))
    
    cm_NB = (metrics.confusion_matrix(test['label'], pred_NB, labels = [1,-1,0]))
    
    pospres_NB,posrec_NB,posf1_NB,negpres_NB,negrec_NB,negf1_NB = data_comp(cm_NB)
    
    print ("1.Positive class: Precision:",pospres_NB," | Recall:",posrec_NB,"| F-score:",posf1_NB)
    print ("1.Negative class: Precision:",negpres_NB," | Recall:",negrec_NB,"| F-score:",negf1_NB)
    
    print ("LOGISTIC REGRESSION CLASSIFIER TEST RESULTS")
    
    print ("LR Test accuracy:", np.mean(pred_LR == test['label']))
    
    cm_LR = (metrics.confusion_matrix(test['label'], pred_LR, labels = [1,-1,0]))
    
    pospres_LR,posrec_LR,posf1_LR,negpres_LR,negrec_LR,negf1_LR = data_comp(cm_LR)
    
    print ("1.Positive class: Precision:",pospres_LR," | Recall:",posrec_LR,"| F-score:",posf1_LR)
    print ("1.Negative class: Precision:",negpres_LR," | Recall:",negrec_LR,"| F-score:",negf1_LR)
    
    print ("SUPPORT VECTOR MACHINE TEST RESULTS")
    
    print ("SVM Test accuracy:", np.mean(pred_SVM == test['label']))
    
    cm_SVM = (metrics.confusion_matrix(test['label'], pred_SVM, labels = [1,-1,0]))
    
    pospres_SVM,posrec_SVM,posf1_SVM,negpres_SVM,negrec_SVM,negf1_SVM = data_comp(cm_SVM)
    
    print ("1.Positive class: Precision:",pospres_SVM," | Recall:",posrec_SVM,"| F-score:",posf1_SVM)
    print ("1.Negative class: Precision:",negpres_SVM," | Recall:",negrec_SVM,"| F-score:",negf1_SVM)
    
    print ("----------------------------------------------------------------------------")
    
    

def main():
    names = ['text', 'label']
    data1 = pd.read_table("oclean.txt", sep="\t", names=names)
    
    ###############################################################################
    # Test models for NORMAL Method
    print ("---------------------------NORMAL PREDICTION TEST---------------------------")
    print ("FOR OBAMA")
    test_challenge(data1,data_test_o)
    ###############################################################################
    
    dfList = np.array_split(data1, 10)
    call_classify(dfList,"OBAMA")
    
    print ("Obama Count")
    print (Counter(data1['label']))
    
    #######################
    
    data2 = pd.read_table("rclean.txt", sep="\t", names=names)
    
    ###############################################################################
    # Test models for NORMAL Method
    print ("---------------------------NORMAL PREDICTION TEST---------------------------")
    print ("FOR ROMNEY")
    test_challenge(data2,data_test_r)
    ###############################################################################
    
    dfList2 = np.array_split(data2, 10)
    call_classify(dfList2,"ROMNEY")
    
    print ("Romneys Count")
    print (Counter(data2['label']))
    
    letsdoadvanceclassify(data1,"OBAMA")
    
    letsdoadvanceclassify(data2,"ROMNEY")
        
    ##########################################################
    
    # Let us try semi superwised learning
    
    names_nosent = ['text']
    data_nosent1 = pd.read_table("onos.txt", names=names_nosent)
    
    semisuplearning(data1,data_nosent1,"OBAMA")
    
    names_nosent = ['text']
    data_nosent2 = pd.read_table("rnos.txt", names=names_nosent)
    
    semisuplearning(data2,data_nosent2,"ROMNEY")
    
def letsdoadvanceclassify(data,dsstr):
    
    print ("LEXICON LEARNING FOR THE "+dsstr+" DATASET")
    
    pos,neg,nu = PosNeg.posneg()
    posw = []
    negw = []
    nuw = []
    for i in list(data['text']):
        for j in i.split(" "):
            if j in pos:
                posw.append(j)
            if j in neg:
                negw.append(j)
            if j in nu:
                nuw.append(j)
                
    if dsstr == "OBAMA":
        with open("oclean2.txt", "a") as myfile:
            print("\n", file=myfile)
            if posw != []:
                for i in range(0, len(posw)):
                    print(str(posw[i])+"\t"+str(1), file=myfile)
            print("\n", file=myfile)
            if negw != []:
                for i in range(0, len(negw)):
                    print(str(negw[i])+"\t"+str(-1), file=myfile)
            print("\n", file=myfile)
            if nuw != []:
                for i in range(0, len(nuw)):
                    print(str(nuw[i])+"\t"+str(0), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("oclean2.txt", sep="\t", names=names)
        
        
        ###############################################################################
        # Test models for LEXICON Method
        print ("---------------------------LEXICON PREDICTION TEST---------------------------")
        print ("FOR OBAMA")
        test_challenge(data1,data_test_o)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        names = ['text', 'label']
        data_ = pd.read_table("oclean.txt", sep="\t", names=names)
        names_nosent = ['text']
        data_nosent1 = pd.read_table("onos.txt", names=names_nosent)
        semisuplearning_l(data_,data_nosent1,"OBAMA")
        
    elif dsstr == "ROMNEY":
        with open("rclean2.txt", "a") as myfile:
            print("\n", file=myfile)
            if posw != []:
                for i in range(0, len(posw)):
                    print(str(posw[i])+"\t"+str(1), file=myfile)
            print("\n", file=myfile)
            if negw != []:
                for i in range(0, len(negw)):
                    print(str(negw[i])+"\t"+str(-1), file=myfile)
            print("\n", file=myfile)
            if nuw != []:
                for i in range(0, len(nuw)):
                    print(str(nuw[i])+"\t"+str(0), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("rclean2.txt", sep="\t", names=names)
        
        ###############################################################################
        # Test models for LEXICON Method
        print ("---------------------------LEXICON PREDICTION TEST---------------------------")
        print ("FOR ROMNEY")
        test_challenge(data1,data_test_r)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        names = ['text', 'label']
        data__ = pd.read_table("rclean.txt", sep="\t", names=names)   
        names_nosent = ['text']
        data_nosent2 = pd.read_table("rnos.txt", names=names_nosent)
        semisuplearning_l(data__,data_nosent2,"ROMNEY")
    
            
    
def call_classify(dfList,dsstr):
    NB_acca = []
    NB_pospresa = []
    NB_posreca  = []
    NB_posf1a= []
    NB_negpresa = []
    NB_negreca = []
    NB_negf1a = []
    SVM_acca = []
    SVM_pospresa = []
    SVM_posreca = []
    SVM_posf1a = []
    SVM_negpresa = []
    SVM_negreca = []
    SVM_negf1a = []
    LR_acca = []
    LR_pospresa = []
    LR_posreca = []
    LR_posf1a = []
    LR_negpresa = []
    LR_negreca = []
    LR_negf1a = []
    en_acca = []
    en_pospresa = []
    en_posreca = []
    en_posf1a = []
    en_negpresa  = []
    en_negreca = []
    en_negf1a  = []
    c=0
    for i in range(0, len(dfList)):
        test = dfList[i]
        train_arr = []
        for j in range(0, len(dfList)):
            if i!=j:
                train_arr.append(dfList[j])
        train = pd.concat(train_arr)
        NB_acc,NB_pospres,NB_posrec,NB_posf1,NB_negpres,NB_negrec,NB_negf1,SVM_acc,SVM_pospres,SVM_posrec,SVM_posf1,SVM_negpres,SVM_negrec,SVM_negf1,LR_acc,LR_pospres,LR_posrec,LR_posf1,LR_negpres,LR_negrec,LR_negf1,en_acc,en_pospres,en_posrec,en_posf1,en_negpres,en_negrec,en_negf1 = do_classify(train, test)
        c=c+1
        NB_acca.append(NB_acc)
        NB_pospresa.append(NB_pospres)
        NB_posreca.append(NB_posrec)
        NB_posf1a.append(NB_posf1)
        NB_negpresa.append(NB_negpres)
        NB_negreca.append(NB_negrec)
        NB_negf1a.append(NB_negf1)
        SVM_acca.append(SVM_acc)
        SVM_pospresa.append(SVM_pospres)
        SVM_posreca.append( SVM_posrec)
        SVM_posf1a.append(SVM_posf1)
        SVM_negpresa.append(SVM_negpres)
        SVM_negreca.append(SVM_negrec)
        SVM_negf1a.append(SVM_negf1)
        LR_acca.append(LR_acc)
        LR_pospresa.append(LR_pospres)
        LR_posreca.append(LR_posrec)
        LR_posf1a.append(LR_posf1)
        LR_negpresa.append(LR_negpres)
        LR_negreca.append(LR_negrec)
        LR_negf1a.append(LR_negf1)
        en_acca.append(en_acc)
        en_pospresa.append(en_pospres)
        en_posreca.append(en_posrec)
        en_posf1a.append(en_posf1)
        en_negpresa.append(en_negpres)
        en_negreca.append(en_negrec)
        en_negf1a.append(en_negf1)
    print ("")
    print ("FOR "+dsstr+" TWEETS")
    print ("")
    print ("FOR NAIVE BAYES CLASSIFIER")
    print ("")
    print ("1.Positive class(Averages): Precision:",np.mean(NB_pospresa)," | Recall:",np.mean(NB_posreca),"| F-score:",np.mean(NB_posf1a))
    print ("1.Negative class(Averages): Precision:",np.mean(NB_negpresa)," | Recall:",np.mean(NB_negreca),"| F-score:",np.mean(NB_negf1a))
    print ("Overall Accuracy(Average):",np.mean(NB_acca)*100,"%")
    print ("")
    print ("FOR SVM CLASSIFIER")
    print ("")
    print ("1.Positive class(Averages): Precision:",np.mean(SVM_pospresa)," | Recall:",np.mean(SVM_posreca),"| F-score:",np.mean(SVM_posf1a))
    print ("1.Negative class(Averages): Precision:",np.mean(SVM_negpresa)," | Recall:",np.mean(SVM_negreca),"| F-score:",np.mean(SVM_negf1a))
    print ("Overall Accuracy(Average):",np.mean(SVM_acca)*100,"%")
    print ("")
    print ("FOR LR CLASSIFIER")
    print ("")
    print ("1.Positive class(Averages): Precision:",np.mean(LR_pospresa)," | Recall:",np.mean(LR_posreca),"| F-score:",np.mean(LR_posf1a))
    print ("1.Negative class(Averages): Precision:",np.mean(LR_negpresa)," | Recall:",np.mean(LR_negreca),"| F-score:",np.mean(LR_negf1a))
    print ("Overall Accuracy(Average):",np.mean(LR_acca)*100,"%")
    print ("")
    print ("FOR EN CLASSIFIER")
    print ("")
    print ("1.Positive class(Averages): Precision:",np.mean(en_pospresa)," | Recall:",np.mean(en_posreca),"| F-score:",np.mean(en_posf1a))
    print ("1.Negative class(Averages): Precision:",np.mean(en_negpresa)," | Recall:",np.mean(en_negreca),"| F-score:",np.mean(en_negf1a))
    print ("Overall Accuracy(Average):",np.mean(en_acca)*100,"%")
    print ("")
    print ("")
    
    
def vect_cal(data1,data2):
    names = ['text', 'label']
    train_data = pd.DataFrame(data1, columns=names)
    test_data = pd.DataFrame(data2, columns=names)
    # Compute TF-IDF Vectors
    
    count_vect = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    X_train_counts = count_vect.fit_transform(train_data['text'].values.astype('U'))
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    X_new_counts = count_vect.transform(test_data['text'].values.astype('U'))
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    return X_train_tfidf, X_new_tfidf
            
def do_classify(train,test):  
    names = ['text', 'label']
    train_data, test_data = pd.DataFrame(train, columns=names), pd.DataFrame(test, columns=names)
    
    # Compute TF-IDF Vectors
    
    X_train_tfidf, X_new_tfidf = vect_cal(train_data,test_data)
    
    ###############################
    
    # Train Model NB
    
    predicted ,NB_acc ,NB_cm ,NB_pospres,NB_posrec,NB_posf1,NB_negpres,NB_negrec,NB_negf1 ,NB_probs = NB(X_train_tfidf,train_data['label'],X_new_tfidf,test_data['label'],1)
    
    ##################################
    
    # Train Model SVM
    
    predicted_2 ,SVM_acc ,SVM_cm ,SVM_pospres,SVM_posrec,SVM_posf1,SVM_negpres,SVM_negrec,SVM_negf1,SVM_probs = SGD(X_train_tfidf,train_data['label'],X_new_tfidf,test_data['label'],1)
    
    ##################################
    
    # Train Model LR
    
    predicted_3 ,LR_acc ,LR_cm ,LR_pospres,LR_posrec,LR_posf1,LR_negpres,LR_negrec,LR_negf1 ,LR_probs = LR(X_train_tfidf,train_data['label'],X_new_tfidf,test_data['label'],1)
    
    #################

    # ENSEMBLE
    
    en_pred = []
    
    for i in range(0, len(predicted)):
        if predicted[i] == predicted_2[i] == predicted_3[i]:
            en_pred.append(predicted[i])   
        elif predicted_2[i] == predicted_3[i]:
            en_pred.append(predicted_2[i])
        elif predicted[i] == predicted_3[i]:
            en_pred.append(predicted[i])
        else:
            en_pred.append(predicted[i])
                
    en_acc = (np.mean(en_pred == test_data['label']))

    en_cm = (metrics.confusion_matrix(test_data['label'], en_pred, labels = [1,-1,0]))
    
    en_pospres,en_posrec,en_posf1,en_negpres,en_negrec,en_negf1 = data_comp(en_cm)
    
    return (NB_acc,NB_pospres,NB_posrec,NB_posf1,NB_negpres,NB_negrec,NB_negf1,
            SVM_acc,SVM_pospres,SVM_posrec,SVM_posf1,SVM_negpres,SVM_negrec,SVM_negf1,
            LR_acc,LR_pospres,LR_posrec,LR_posf1,LR_negpres,LR_negrec,LR_negf1,
            en_acc,en_pospres,en_posrec,en_posf1,en_negpres,en_negrec,en_negf1)
    
def NB(train_tfidf,train_label,test_tfidf,test_label,flag):

    clf = MultinomialNB().fit(train_tfidf,train_label)
    
    predicted ,NB_acc ,NB_cm ,NB_pospres,NB_posrec,NB_posf1,NB_negpres,NB_negrec,NB_negf1 ,NB_probs = comp_vars(clf,test_tfidf,test_label)
    
    if flag == 0:
        return NB_acc
    else:
        return predicted ,NB_acc ,NB_cm ,NB_pospres,NB_posrec,NB_posf1,NB_negpres,NB_negrec,NB_negf1 ,NB_probs
    
def SGD(train_tfidf,train_label,test_tfidf,test_label,flag):
    
    clf_2 = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(train_tfidf,train_label)
    
    predicted_2 ,SVM_acc ,SVM_cm ,SVM_pospres,SVM_posrec,SVM_posf1,SVM_negpres,SVM_negrec,SVM_negf1,SVM_probs = comp_vars(clf_2,test_tfidf,test_label)
    
    if flag == 0:
        return SVM_acc
    else:
        return predicted_2 ,SVM_acc ,SVM_cm ,SVM_pospres,SVM_posrec,SVM_posf1,SVM_negpres,SVM_negrec,SVM_negf1,SVM_probs

def LR(train_tfidf,train_label,test_tfidf,test_label,flag):
    clf_3 = linear_model.LogisticRegression(multi_class='multinomial', solver='sag').fit(train_tfidf,train_label)
    
    predicted_3 ,LR_acc ,LR_cm ,LR_pospres,LR_posrec,LR_posf1,LR_negpres,LR_negrec,LR_negf1 ,LR_probs = comp_vars(clf_3,test_tfidf,test_label)
    
    if flag == 0:
        return LR_acc
    else:
        return predicted_3 ,LR_acc ,LR_cm ,LR_pospres,LR_posrec,LR_posf1,LR_negpres,LR_negrec,LR_negf1 ,LR_probs

def comp_vars(clf, test_tfidf, test_label):
    
    predicted = clf.predict(test_tfidf)
    
    acc = np.mean(predicted == test_label)
    
    cm = (metrics.confusion_matrix(test_label, predicted, labels = [1,-1,0]))
    
    pospres,posrec,posf1,negpres,negrec,negf1 = data_comp(cm)
    
    probs = clf.predict_proba(test_tfidf)
    
    return predicted, acc, cm, pospres, posrec, posf1, negpres, negrec, negf1, probs

def semisuplearning(data,nosent,dsstr):
    print ("")
    print ("SEMI-SUPERVISED LEARNING FOR THE "+dsstr+" DATASET")
    print ("")
    names = ['text', 'label']
    train_data = pd.DataFrame(data, columns=names)
    names_nosent = ['text']
    nosent_data = pd.DataFrame(nosent, columns=names_nosent)
    # Compute TF-IDF Vectors
    
    count_vect = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    #count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data['text'].values.astype('U'))
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    #print (X_train_tf.shape)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print (X_train_tfidf.shape)
    
    X_new_counts_2 = count_vect.transform(nosent_data['text'].values.astype('U'))
    X_new_tfidf_2 = tfidf_transformer.transform(X_new_counts_2)
    
    ######################################################################################################
    
    clf_4 = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, train_data['label'])
    
    predicted_4 = clf_4.predict(X_new_tfidf_2)
    
    pred = list(predicted_4)
    text = list(nosent_data['text'])
    
    pos,neg,nu = PosNeg.posneg()
    
    pred = lexi(text, pred, pos, neg, nu)
    
    if dsstr == "OBAMA":
        with open("oclean.txt", "a") as myfile:
            print("\n", file=myfile)
            for i in range(0, len(text)):
                print(str(text[i])+"\t"+str(pred[i]), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("oclean.txt", sep="\t", names=names)
        
        
        ###############################################################################
        # Test models for SEMI Method
        print ("---------------------------SEMI-SUPERVISED PREDICTION TEST---------------------------")
        print ("FOR "+dsstr)
        test_challenge(data1,data_test_o)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        
    elif dsstr == "ROMNEY":
        with open("rclean.txt", "a") as myfile:
            print("\n", file=myfile)
            for i in range(0, len(text)):
                print(str(text[i])+"\t"+str(pred[i]), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("rclean.txt", sep="\t", names=names)
        
        
        ###############################################################################
        # Test models for SEMI Method
        print ("---------------------------SEMI-SUPERVISED PREDICTION TEST---------------------------")
        print ("FOR "+dsstr)
        test_challenge(data1,data_test_r)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        
def semisuplearning_l(data,nosent,dsstr):
    print ("")
    print ("SEMI-SUPERVISED WITH LEXICON LEARNING FOR THE "+dsstr+" DATASET")
    print ("")
    names = ['text', 'label']
    train_data = pd.DataFrame(data, columns=names)
    names_nosent = ['text']
    nosent_data = pd.DataFrame(nosent, columns=names_nosent)
    # Compute TF-IDF Vectors
    
    count_vect = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    #count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data['text'].values.astype('U'))
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    X_new_counts_2 = count_vect.transform(nosent_data['text'].values.astype('U'))
    X_new_tfidf_2 = tfidf_transformer.transform(X_new_counts_2)
    
    ######################################################################################################
    
    clf_4 = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, train_data['label'])
    
    predicted_4 = clf_4.predict(X_new_tfidf_2)
    
    pred = list(predicted_4)
    text = list(nosent_data['text'])
    
    pos,neg,nu = PosNeg.posneg()
    
    pred = lexi(text, pred, pos, neg, nu)
    
    if dsstr == "OBAMA":
        with open("oclean2.txt", "a") as myfile:
            print("\n", file=myfile)
            for i in range(0, len(text)):
                print(str(text[i])+"\t"+str(pred[i]), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("oclean2.txt", sep="\t", names=names)
        
        
        ###############################################################################
        # Test models for LEXICON SEMI Method
        print ("---------------------------LEXICON WITH SEMI PREDICTION TEST---------------------------")
        print ("FOR "+dsstr)
        test_challenge(data1,data_test_o)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        
    elif dsstr == "ROMNEY":
        with open("rclean2.txt", "a") as myfile:
            print("\n", file=myfile)
            for i in range(0, len(text)):
                print(str(text[i])+"\t"+str(pred[i]), file=myfile)
                
        names = ['text', 'label']
        data1 = pd.read_table("rclean2.txt", sep="\t", names=names)
        
        ###############################################################################
        # Test models for LEXICON SEMI Method
        print ("---------------------------LEXICON WITH SEMI PREDICTION TEST---------------------------")
        print ("FOR "+dsstr)
        test_challenge(data1,data_test_r)
        ###############################################################################
        
        dfList = np.array_split(data1, 10)
        call_classify(dfList,dsstr)
        
def lexi(tw,sc,po,ne,nu):
    sumall = []
    for i in tw:
        sum_ = [0,0,0]
        if (type(i) != type(0.0)):
            for j in i.split(" "):
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

def data_comp(cm):
    pres_pos = cm[0][0]/(cm[0][0]+cm[1][0]+cm[2][0])
    pres_neg = cm[1][1]/(cm[0][1]+cm[1][1]+cm[2][1])
    
    rec_pos = cm[0][0]/(cm[0][0]+cm[0][1]+cm[0][2]) 
    rec_neg = cm[1][1]/(cm[1][0]+cm[1][1]+cm[1][2])
    
    f1_pos = (2*pres_pos*rec_pos)/(pres_pos+rec_pos)
    f1_neg = (2*pres_neg*rec_neg)/(pres_neg+rec_neg)
    
    return pres_pos, rec_pos, f1_pos, pres_neg, rec_neg, f1_neg   
    
if __name__ == '__main__':
    main()