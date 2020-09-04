# -*- coding: utf-8 -*-
"""

Frustration Intensity Prediction in Customer Support Dialog Texts
python3 code for experimentation

"""

"""
Instruction to run the program:
    1) Dialog input file should exist in the current directory named "dprocessed.txt"
        (example of a file given in the repository)
    2) The parameters should be set as global variables:
        * USER_KEYWORD_COUNT
        * SUPPORT_KEYWORD_COUNT
        (see releated research paper "Frustration Intensity Prediction in Customer Support Dialog Texts")
    2a) Optional parameters as localal variables can be also set:
        * HIDDEN_COUNT
        * EPOCHS
    3) python3 file to be run without parameters
    4) output is written into "output.txt"
"""

USER_KEYWORD_COUNT = 500
SUPPORT_KEYWORD_COUNT = 200


import sys
from datetime import datetime

import tensorflow as tf
from collections import Counter
from statistics import median
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

import nltk
#nltk.download('punkt')

agrstat=[0,0,0,0] # expert agreement stat
gradestat = [0,0,0,0,0] # stat of 5 different grades

def add_dialogue_info(ddata,dgrades,dialogue,key):
    dialogue2 = []
    grades2 = []
    for tnum,turn in enumerate(dialogue):
        gr = []
        if turn[:5]=="USER:":
            pos1 = turn.find('{')
            pos2 = turn.find('}')
            if pos1>=0 and pos2>=0:
                gradetext = turn[pos1+1:pos2].strip()
                turn = turn[:pos1].rstrip()
                gr = gradetext.split(',')
        dialogue2.append(turn)
        grades2.append(gr)
    if len(dialogue2)>0:
        ddata[key] = dialogue2
        dgrades[key] = grades2

def analyze_annotation_file(fname):
    user_turns = 0
    supp_turns = 0
    ddata = {}
    dgrades = {}
    key = ""
    dialogue = []
    for line in open(fname,"r",encoding="utf-8"):
        if line.lstrip()[1:2]=="-":
            pos1 = line.find('[')
            pos2 = line.find(']')
            if pos1>=0 and pos2>=0:
                key = line[pos1:pos2+1]
        elif line[:5] in ("USER:","SUPP:"):
            if line[:5]=='USER:':
                user_turns += 1
            else:
                supp_turns += 1
            if key!="": 
                line = line.replace('&amp;','&')
                line = line.replace('&gt;','>')
                line = line.replace('&lt;','<')
                line = line.replace(';',' .,')
                dialogue.append(line)
        else:
            add_dialogue_info(ddata,dgrades,dialogue,key)
            key = ""
            dialogue = []
    add_dialogue_info(ddata,dgrades,dialogue,key)
    print('dialogues',len(ddata))
    print('user_turns',user_turns)
    print('supp_turns',supp_turns)
    print('avg dialogue length',(user_turns+supp_turns)/len(ddata))
    return ddata,dgrades

def compute_final_grade_median3(gr):
    if len(gr)==0:
        return -1
    res = []
    for g in gr:
        if g=="":
            return -1
        elif g.lower()=="n":
            return -2
        elif g.isdigit():
            res.append(int(g))
        else:
            return -1
    assert len(res)==3
    global agrstat
    med = median(res)
    agrstat[3]+=1
    for i in range(3):
        if med==res[i]:
            agrstat[i]+=1
    return med

def dialogue_grade_stats(ddata,dgrades,key):
    di = 0
    dy = 0
    for i,line in enumerate(ddata[key]):
        if line[:5]=="USER:":
            di += 1
            gry = compute_final_grade_median3(dgrades[key][i])
            if gry>=0:
                dy += gry
            else:
                dy = gry
                break
    else:
        dy /= di
    return dy

def gradedicttolist(gd):
    glist = [0 for _ in range(7)]
    for g in range(-2,5):
        glist[g] = gd[g]
    return glist

def gradedicttolist_supp(gd):
    glist = [0 for _ in range(11)]
    for g in range(-2,9):
        glist[g] = gd[g]
    return glist

def countposaverage(glist):
    cnt = 0
    tot = 0.0
    for g in glist:
        if g>=0:
            cnt += glist[g]
            tot += g*glist[g]
    if cnt==0: return -1.0
    else: return tot / cnt

def countposstdev(glist):
    avg = countposaverage(glist)
    cnt = 0
    tot = 0.0
    for g in glist:
        if g>=0:
            cnt += glist[g]
            tot += glist[g]*(g-avg)**2
    if cnt==0: return 99
    if cnt==1: return 0.0
    return (tot / cnt)**0.5

def evaluate_outcomes_accuracy_2D_01_special0(a,b,coeffs=(1.0,)):
    ret = 0.0
    amax = np.argmax(a)
    bmax = np.argmax(b)
    diff = abs(amax-bmax)
#    print("diff",diff,amax,bmax)
    if diff<len(coeffs):
        ret += coeffs[diff]
    return ret

def evaluate_outcomes_accuracy_2D_01_special(actual,test,coeffs=(1.0,)):
    ret = 0.0
    for i in range(len(actual)):
        a = actual[i]
        b = test[i]
        amax = np.argmax(a)
        bmax = np.argmax(b)
        diff = abs(amax-bmax)
        if diff<len(coeffs):
            ret += coeffs[diff]
    return ret / len(actual)

if __name__ == '__main__':
    usrvocab = Counter()
    suppvocab = Counter()
    usrworddict = {}
    suppworddict = {}
#    usrwordlist = []
#    suppwordlist = []
    usrgradevocaby = {}
    suppgradevocaby = {}
    grydeltasum = 0
    grydeltacount = 0
    grydeltastats = [0 for _ in range(9)]
    
    CODE_TEST = False
    CODE_TEST_EXP_COUNT = 1
    OUT_TO_FILE = True
    if OUT_TO_FILE:
        fout = open("output.txt","a",encoding="utf-8")
    else:
        fout = sys.stdout
    print("\n\n============= EXPERIMENT =============",datetime.now(),file=fout)
    if CODE_TEST:
        print('************** Debug mode **************')
        print('************** Debug mode **************',file=fout)

    
    # A. read dialogues from file
    ddata,dgrades=analyze_annotation_file("dprocessed.txt")
    print(len(ddata),len(dgrades))

    # B. process dialogues #1
    for key in ddata: # for all dialogues
        dy = dialogue_grade_stats(ddata,dgrades,key)
        # dialogue main processing
        turnlist_supp = []
        gry = -1
        gryprev = -1
        iusr = 0
        for i,line in enumerate(ddata[key]):
            # printing A details
            turn = line[5:].strip().replace(' /// ',' ')
            turnlist = nltk.word_tokenize(turn)
            if line[:5]=="USER:":
                iusr += 1
                gr = ",".join(dgrades[key][i])
                gryprev = gry
                gry = compute_final_grade_median3(dgrades[key][i])
                # USER stats
                for word in turnlist:
                    word = word.lower()
                    if word not in usrgradevocaby:
                        usrgradevocaby[word] = Counter()
                    usrvocab[word] += 1
                    usrgradevocaby[word][gry] +=1
                # SUPP stats
                if iusr>=2:
                    if gry==-1 or gryprev==-1:
                        grydelta = -1
                    elif gry==-2 or gryprev==-2:
                        grydelta = -2
                    else:
                        grydeltasum += gry - gryprev
                        grydeltacount += 1
                        grydelta = gry - gryprev + 4
                        grydeltastats[grydelta] += 1
                    for word in turnlist_supp:
                        word = word.lower()
                        if word not in suppgradevocaby:
                            suppgradevocaby[word] = Counter()
                        suppvocab[word] += 1
                        suppgradevocaby[word][grydelta] +=1
            elif line[:5]=="SUPP:":
                takelist_supp = turnlist
            else:
                print('@#@ Illegal turn:',line)
                assert False

    # C. process dialogue statistics
    print("usrgradevocaby",len(usrgradevocaby),file=fout)
    MIN_FREQ_STD_MEDIAN = 3
    i = 0
    for w in sorted(usrgradevocaby.items(),key=lambda x: (countposstdev(x[1]),-sum(gradedicttolist(x[1])[:-2])), reverse=False):
        word  = w[0]
        gd = w[1]
        if sum(gradedicttolist(gd)[:-2])>=MIN_FREQ_STD_MEDIAN:
            i += 1
            if i<= USER_KEYWORD_COUNT:
                usrworddict[word] = i-1
    
    print("suppgradevocaby",len(suppgradevocaby),file=fout)
    MIN_FREQ_STD_MEDIAN_SUPP = 3
    i = 0
    for w in sorted(suppgradevocaby.items(),
                    key=lambda x: (countposstdev(x[1]),-sum(gradedicttolist_supp(x[1])[:-2])), reverse=False):
        word  = w[0]
        gd = w[1]
        if sum(gradedicttolist(gd)[:-2])>=MIN_FREQ_STD_MEDIAN_SUPP:
            i += 1
            if i<= SUPPORT_KEYWORD_COUNT:
                suppworddict[word] = i-1
    
    print("agreementstat (exp1, exp2, exp3, total)",agrstat,file=fout)
    print("agreementstat (exp1-ratio, exp2-ratio, exp3-ratio)",
          agrstat[0]/agrstat[3],agrstat[1]/agrstat[3],agrstat[2]/agrstat[3],file=fout)

    # D. process dialogues #2
    GRADE_COUNT = 5
    OUTPUT_SIZE = GRADE_COUNT
    INPUT_SIZE = 0
    X_all_list = []
    y_all_list = []
    y_all_num = []
    X1_all_list = []
    y1_all_list = []
    y1_all_num = []
    X2_all_list = []
    y2_all_list = []
    y2_all_num = []
    y22_all_list = []
    y22_all_num = []
    
    HOT_OUT = False
    if HOT_OUT:
        hotusr = open("data/hot-usr-data.csv","w",encoding="utf-8")
        hotusrplus = open("data/hot-usrplus-data.csv","w",encoding="utf-8")
        print(";".join(["usr-hot"+str(i) for i in range(USER_KEYWORD_COUNT)]),
              "take-grade","dialogue-key",sep=';',file=hotusr)
        print(";".join(["usr1-hot"+str(i) for i in range(USER_KEYWORD_COUNT)]),
              "usr-take1-grade",
              ";".join(["supp-hot"+str(i) for i in range(SUPPORT_KEYWORD_COUNT)]),
              ";".join(["usr2-hot"+str(i) for i in range(USER_KEYWORD_COUNT)]),
              "usr-take2-grade","dialogue-key",sep=';',file=hotusrplus)
    for key in ddata: # for all dialogues
        supp_hot = []
        prev_hot = []
        supp_hot_string = []
        prev_hot_string = []
        supp_goodwords = []
        prev_goodwords = []
        supp_take = ""
        prev_take = ""
        gry = -1
        gryprev = -1
        iusr = 0
        for i,line in enumerate(ddata[key]):
            take = line[5:].strip().replace(' /// ',' ')
            takelist = nltk.word_tokenize(take)
            if line[:5]=="USER:":
                iusr += 1
                gryprev = gry
                gry = compute_final_grade_median3(dgrades[key][i])
                goodwords = set()
                curr_hot = [0 for _ in range(USER_KEYWORD_COUNT)]
                for word in takelist:
                    if word in usrworddict:
                        goodwords.add(word)
                        curr_hot[usrworddict[word]] = 1
                curr_hot_string = [str(i) for i in curr_hot]
                curr_goodwords = list(goodwords)
                if gry>=0:
                    if HOT_OUT:
                        print(";".join(curr_hot_string),gry,key,sep=';',file=hotusr)
                    gradelist = list(to_categorical(gry,OUTPUT_SIZE))
                    X_all_list.append(curr_hot)
                    y_all_list.append(gradelist)
                    y_all_num.append(gry)
                    gradestat[gry] += 1
                    if iusr>=2:
                        if gryprev>=0:
                            grydiff = max(min(gry-gryprev+OUTPUT_SIZE//2,OUTPUT_SIZE-1),0)
                            if HOT_OUT:
                                print(";".join(prev_hot_string),gryprev,";".join(supp_hot_string),
                                      ";".join(curr_hot_string),gry,key,sep=';',file=hotusrplus)
                            prev_gradelist = list(to_categorical(gryprev,OUTPUT_SIZE))
                            gradelist = list(to_categorical(gry,OUTPUT_SIZE))
                            gradedifflist = list(to_categorical(grydiff,OUTPUT_SIZE))
                            X1_all_list.append(prev_hot)
                            y1_all_list.append(prev_gradelist)
                            y1_all_num.append(gryprev)
                            X2_all_list.append(prev_hot+supp_hot)
                            y2_all_list.append(gradelist)
                            y2_all_num.append(gry)
                            y22_all_list.append(gradedifflist)
                            y22_all_num.append(grydiff)
                prev_goodwords = curr_goodwords
                prev_take = take
                prev_hot = curr_hot
                prev_hot_string = curr_hot_string
            elif line[:5]=="SUPP:":
                goodwords = set()
                curr_hot = [0 for _ in range(SUPPORT_KEYWORD_COUNT)]
                for word in takelist:
                    if word in suppworddict:
                        goodwords.add(word)
                        curr_hot[suppworddict[word]] = 1
                curr_hot_string = [str(i) for i in curr_hot]
                supp_goodwords = list(goodwords)
                supp_take = take
                supp_hot = curr_hot
                supp_hot_string = curr_hot_string

    print("gradestat",gradestat,file=fout)
    print("grydeltainfo (grydeltasum,grydeltacount,grydeltasum/grydeltacount)",
          grydeltasum,grydeltacount,grydeltasum/grydeltacount,file=fout)
    print("grydeltastats",grydeltastats,file=fout)

    if HOT_OUT:
        hotusr.close()
        hotusrplus.close()
    
    # E. Modeling
#    ref_output = np.argmax(gradestat)
    HIDDEN_COUNT = 64
    EPOCHS = 50
    INPUT_SIZE = len(X_all_list[0])
    
    print("\n  === USR EXP ===",datetime.now(),file=fout)
    print("HIDDEN_COUNT",HIDDEN_COUNT,file=fout)
    print("EPOCHS",EPOCHS,file=fout)
    print("INPUT_SIZE",INPUT_SIZE,file=fout)

    accuracy = 0 
    refaccuracy = 0 
    accuracyplus = 0 
    refaccuracyplus = 0 
    daccuracy = [0.0 for _ in range(OUTPUT_SIZE)] 
    ddaccuracy = [[0 for _ in range(OUTPUT_SIZE)] for _ in range(OUTPUT_SIZE)] 
    daccuracyplus = [0.0 for _ in range(OUTPUT_SIZE)] 
    daccstat = [0 for _ in range(OUTPUT_SIZE)] 
    
    for e in range(len(X_all_list)):
        X_train = np.array(X_all_list[:e] + X_all_list[e+1:]).astype(np.float32)
        y_train = np.array(y_all_list[:e] + y_all_list[e+1:]).astype(np.float32)
        X_test = np.array(X_all_list[e:e+1]).astype(np.float32)
        y_test = np.array(y_all_list[e:e+1]).astype(np.float32)
        y_correct_num = y_all_num[e]
        print("exp1",e+1,len(X_all_list),y_correct_num)
        model = keras.Sequential()
        model.add(Flatten(input_shape=(INPUT_SIZE,)))
        model.add(Dense(HIDDEN_COUNT, activation=tf.nn.relu))
        model.add(Dense(OUTPUT_SIZE, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=EPOCHS, verbose=False)
        y_pred = model.predict(X_test)
        y_pred_num = np.argmax(y_pred)
        acc0 = evaluate_outcomes_accuracy_2D_01_special0(y_pred[0],y_test[0],coeffs=(1.0,))
        refacc0 = evaluate_outcomes_accuracy_2D_01_special0(gradestat,y_test[0],coeffs=(1.0,))
        acc00 = evaluate_outcomes_accuracy_2D_01_special0(y_pred[0],y_test[0],coeffs=(1.0,1.0))
        refacc00 = evaluate_outcomes_accuracy_2D_01_special0(gradestat,y_test[0],coeffs=(1.0,1.0))
        accuracy += acc0
        accuracyplus += acc00
        daccstat[y_correct_num] += 1
        ddaccuracy[y_correct_num][y_pred_num] += 1
        daccuracy[y_correct_num] += acc0
        daccuracyplus[y_correct_num] += acc00
        refaccuracy += refacc0
        refaccuracyplus += refacc00
        if CODE_TEST:
            if e+1==CODE_TEST_EXP_COUNT:
                break
    
    data_count = sum(daccstat)
    accuracy /= data_count
    accuracyplus /= data_count
    refaccuracy /= data_count
    refaccuracyplus /= data_count
    for i in range(len(daccstat)):
        if daccstat[i]>0:
            daccuracy[i] /= daccstat[i]
            daccuracyplus[i] /= daccstat[i]
    print("accuracy,refaccuracy",accuracy,refaccuracy,file=fout)
    print("accuracyplus,refaccuracyplus",accuracyplus,refaccuracyplus,file=fout)
    print("daccstat",daccstat,file=fout)
    print("daccuracy (accuracy by correct grade)",daccuracy,file=fout)
    print("daccuracyplus (accuracy-plus by correct grade)",daccuracyplus,file=fout)
    print("ddaccuracy (confusion matrix)\n",np.array(ddaccuracy),file=fout)
    
    HIDDEN_COUNT = 64
    EPOCHS = 50
    INPUT_SIZE_1 = len(X1_all_list[0])
    INPUT_SIZE_2 = len(X2_all_list[0])
    
    print("\n  === USRPLUS EXP ===",datetime.now(),file=fout)
    print("HIDDEN_COUNT",HIDDEN_COUNT,file=fout)
    print("EPOCHS",EPOCHS,file=fout)
    print("INPUT_SIZE_1",INPUT_SIZE_1,file=fout)
    print("INPUT_SIZE_2",INPUT_SIZE_2,file=fout)

    accuracy = 0 
    accuracyplus = 0 
    refaccuracy = 0 
    refaccuracyplus = 0 
    rrefaccuracy = 0 
    rrefaccuracyplus = 0 
    ddaccuracy1 = [[0 for _ in range(OUTPUT_SIZE)] for _ in range(OUTPUT_SIZE)] 
    ddaccuracy2 = [[0 for _ in range(OUTPUT_SIZE)] for _ in range(OUTPUT_SIZE)] 
    daccuracy = [0.0 for _ in range(OUTPUT_SIZE)] 
    daccuracyplus = [0.0 for _ in range(OUTPUT_SIZE)] 
    daccstat = [0 for _ in range(OUTPUT_SIZE)] 

    for e in range(len(X2_all_list)):
        X1_train = np.array(X1_all_list[:e] + X1_all_list[e+1:]).astype(np.float32)
        y1_train = np.array(y1_all_list[:e] + y1_all_list[e+1:]).astype(np.float32)
        X2_train = np.array(X2_all_list[:e] + X2_all_list[e+1:]).astype(np.float32)
        y2_train = np.array(y2_all_list[:e] + y2_all_list[e+1:]).astype(np.float32)
        y22_train = np.array(y22_all_list[:e] + y22_all_list[e+1:]).astype(np.float32)
        X1_test = np.array(X1_all_list[e:e+1]).astype(np.float32)
        X2_test = np.array(X2_all_list[e:e+1]).astype(np.float32)
        y2_test = np.array(y2_all_list[e:e+1]).astype(np.float32)
        y2_correct_num = y2_all_num[e]
        y22_test = np.array(y22_all_list[e:e+1]).astype(np.float32)
        y22_correct_num = y22_all_num[e]
        print("exp2",e+1,len(X2_all_list),y2_correct_num)

        refmodel = keras.Sequential()
        refmodel.add(Flatten(input_shape=(INPUT_SIZE_1,)))
        refmodel.add(Dense(HIDDEN_COUNT, activation=tf.nn.relu))
        refmodel.add(Dense(OUTPUT_SIZE, activation=tf.nn.softmax))
        refmodel.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        refmodel.fit(X1_train, y1_train, epochs=EPOCHS, verbose=False)
        y1_pred = refmodel.predict(X1_test)
        y1_pred_num = np.argmax(y1_pred)

        rrefmodel = keras.Sequential()
        rrefmodel.add(Flatten(input_shape=(INPUT_SIZE_1,)))
        rrefmodel.add(Dense(HIDDEN_COUNT, activation=tf.nn.relu))
        rrefmodel.add(Dense(OUTPUT_SIZE, activation=tf.nn.softmax))
        rrefmodel.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        rrefmodel.fit(X1_train, y2_train, epochs=EPOCHS, verbose=False)
        y2_pred_ref = rrefmodel.predict(X1_test)
        y2_pred_num_ref = np.argmax(y2_pred_ref)

        model = keras.Sequential()
        model.add(Flatten(input_shape=(INPUT_SIZE_2,)))
        model.add(Dense(HIDDEN_COUNT, activation=tf.nn.relu))
        model.add(Dense(OUTPUT_SIZE, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X2_train, y2_train, epochs=EPOCHS, verbose=False)
        y2_pred = model.predict(X2_test)
        y2_pred_num = np.argmax(y2_pred)

        acc0 = evaluate_outcomes_accuracy_2D_01_special0(y2_pred[0],y2_test[0],coeffs=(1.0,))
        acc00 = evaluate_outcomes_accuracy_2D_01_special0(y2_pred[0],y2_test[0],coeffs=(1.0,1.0))
        refacc0 = evaluate_outcomes_accuracy_2D_01_special0(y1_pred[0],y2_test[0],coeffs=(1.0,))
        refacc00 = evaluate_outcomes_accuracy_2D_01_special0(y1_pred[0],y2_test[0],coeffs=(1.0,1.0))
        rrefacc0 = evaluate_outcomes_accuracy_2D_01_special0(y2_pred_ref[0],y2_test[0],coeffs=(1.0,))
        rrefacc00 = evaluate_outcomes_accuracy_2D_01_special0(y2_pred_ref[0],y2_test[0],coeffs=(1.0,1.0))
        accuracy += acc0
        accuracyplus += acc00
        daccstat[y2_correct_num] += 1
        ddaccuracy1[y2_correct_num][y1_pred_num] += 1
        ddaccuracy2[y2_correct_num][y2_pred_num] += 1
        daccuracy[y2_correct_num] += acc0
        daccuracyplus[y2_correct_num] += acc00
        refaccuracy += refacc0
        refaccuracyplus += refacc00
        rrefaccuracy += rrefacc0
        rrefaccuracyplus += rrefacc00

        if CODE_TEST:
            if e+1==CODE_TEST_EXP_COUNT:
                break

    data_count = sum(daccstat)
    accuracy /= data_count
    accuracyplus /= data_count
    refaccuracy /= data_count
    refaccuracyplus /= data_count
    rrefaccuracy /= data_count
    rrefaccuracyplus /= data_count
    for i in range(len(daccstat)):
        if daccstat[i]>0:
            daccuracy[i] /= daccstat[i]
            daccuracyplus[i] /= daccstat[i]
    print("accuracy,refaccuracy,rrefaccuracy",
          accuracy,refaccuracy,rrefaccuracy,file=fout)
    print("accuracyplus,refaccuracyplus,rrefaccuracyplus",
          accuracyplus,refaccuracyplus,rrefaccuracyplus,file=fout)
    print("daccstat",daccstat,file=fout)
    print("daccuracy (accuracy by correct grade)",daccuracy,file=fout)
    print("daccuracyplus (accuracy-plus by correct grade)",daccuracyplus,file=fout)
    print("ddaccuracy1 (confusion matrix usr)\n",np.array(ddaccuracy1),file=fout)
    print("ddaccuracy2 (confusion matrix usrplus)\n",np.array(ddaccuracy2),file=fout)

    print("Experiment end time:",datetime.now(),file=fout)
    if OUT_TO_FILE:
        fout.close()

