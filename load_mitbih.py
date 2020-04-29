# -*- coding: utf-8 -*-

import os
import csv

import numpy as np
from scipy.signal import medfilt
from features_ecg import *

# Load the data with the configuration and features selected
# Params:
# - leads_flag = [MLII, V1] set the value to 0 or 1 to reference if that lead is used
# - reduced_DS = load DS1, DS2 patients division (Chazal) or reduced version, 
#                i.e., only patients in common that contains both MLII and V1 
def load_mit_db(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag):
    print("Loading MIT BIH arr (" + DS + ") ...")

    # ML-II
    if reduced_DS == False:
        DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
         #DS1 = [101, 106]
         #DS2 = [100, 103]
    # ML-II + V1
    else:
        DS1 = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        DS2 = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
         #DS1 = [101, 106]
         #DS2 = [105, 111]
    if DS == 'DS1':
        my_db = load_signal(DS1, winL, winR, do_preprocess)
        labels = np.array(sum(my_db.class_ID, [])).flatten()
        #np.savetxt('FusionResults_adaboost/' + 'DS2_traininglabels.csv', labels.astype(int), '%.0f')   
    else:
        my_db = load_signal(DS2, winL, winR, do_preprocess)
        labels = np.array(sum(my_db.class_ID, [])).flatten()
        #np.savetxt('FusionResults_adaboost/' + 'DS2_testinglabels.csv', labels.astype(int), '%.0f')
    
    features = np.array([], dtype=float)
    labels = np.array([], dtype=np.int32)

    # This array contains the number of beats for each patient (for cross_val)
    patient_num_beats = np.array([], dtype=np.int32)
    for p in range(len(my_db.beat)):
        patient_num_beats = np.append(patient_num_beats, len(my_db.beat[p]))
        
    f1 = np.array([],dtype=float)
    f2 = np.array([],dtype=float)
    f3 = np.array([],dtype=float)
    f4 = np.array([],dtype=float)


   

    # Compute RR features
    if use_RR or norm_RR:
        if DS == 'DS1':
            RR = [RR_intervals() for i in range(len(DS1))]
        else:
            RR = [RR_intervals() for i in range(len(DS2))]

        print("Computing RR intervals ...")

        for p in range(len(my_db.beat)):
            if maxRR:
                RR[p] = compute_RR_intervals(my_db.R_pos[p])
            else:
                RR[p] = compute_RR_intervals(my_db.orig_R_pos[p])
                
            RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
            RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
            RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
            RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]


    if use_RR:
        f_RR = np.empty((0,4))
        for p in range(len(RR)):
            row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
            f_RR = np.vstack((f_RR, row))

        f1 = np.column_stack((f1, f_RR)) if f1.size else f_RR
    
    if norm_RR:
        f_RR_norm = np.empty((0,4))
        for p in range(len(RR)):
            # Compute avg values!
            avg_pre_R = np.average(RR[p].pre_R)
            avg_post_R = np.average(RR[p].post_R)
            avg_local_R = np.average(RR[p].local_R)
            avg_global_R = np.average(RR[p].global_R)

            row = np.column_stack((RR[p].pre_R / avg_pre_R, RR[p].post_R / avg_post_R, RR[p].local_R / avg_local_R, RR[p].global_R / avg_global_R))
            f_RR_norm = np.vstack((f_RR_norm, row))

        f1 = np.column_stack((f1, f_RR_norm))  if f1.size else f_RR_norm

    #########################################################################################
    # Compute morphological features
   

    num_leads = np.sum(leads_flag)


    # Wavelets
    if (True):
        print("Wavelets ...")

        f_wav = np.empty((0, 23 * num_leads))

        for p in range(len(my_db.beat)):
            for b in my_db.beat[p]:
                f_wav_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_wav_lead.size == 1:
                            f_wav_lead =  compute_wavelet_descriptor(b[s], 'db1', 3)
                        else:
                            f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
                #print(f_wav.shape)
                #print(f_wav_lead.shape)
                f_wav = np.vstack((f_wav, f_wav_lead))
                #f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))
                
        f2 = f_wav
        #f2 = np.column_stack((f2, f_wav))  if f2.size else f_wav
        
        print(str(f2.shape))

    # HOS
    if (True):
        print("HOS ...")
        n_intervals = 6
        lag = int(round( (winL + winR )/ n_intervals))

        f_HOS = np.empty((0, (n_intervals-1) * 2 * num_leads))
        for p in range(len(my_db.beat)):
            for b in my_db.beat[p]:
                f_HOS_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_HOS_lead.size == 1:
                            f_HOS_lead =  compute_hos_descriptor(b[s], n_intervals, lag)
                        else:
                            f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
                f_HOS = np.vstack((f_HOS, f_HOS_lead))
                #f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))

        #f3 = np.column_stack((f3, f_HOS))  if f3.size else f_HOS
        f3 = f_HOS
        print(str(f3.shape))

    # My morphological descriptor
    if (True):
        print("My Descriptor ...")
        f_myMorhp = np.empty((0,4 * num_leads))
        for p in range(len(my_db.beat)):
            for b in my_db.beat[p]:
                f_myMorhp_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_myMorhp_lead.size == 1:
                            f_myMorhp_lead =  compute_my_own_descriptor(b[s], winL, winR)
                        else:
                            f_myMorhp_lead = np.hstack((f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
                f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                #f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))
        f4 = f_myMorhp
        #f4 = np.column_stack((f4, f_myMorhp))  if f4.size else f_myMorhp
        print(str(f4.shape))
    
    labels = np.array(sum(my_db.class_ID, [])).flatten()
    print("labels")

    # Set labels array!
    #print(str(features) + " " + str(labels) + " " + str(patient_num_beats))
    #print(my_db.beat)
    with open("RR-features.txt", "w") as txt_file:
        for line in f1:
            txt_file.write(" ".join(str(v) for v in line) + "\n")
            
    with open("wavelet-features.txt", "w") as txt_file:
        for line in f2:
            txt_file.write(" ".join(str(v) for v in line) + "\n")
            
    with open("HOS.txt", "w") as txt_file:
        for line in f3:
            txt_file.write(" ".join(str(v) for v in line) + "\n")
            
    with open("morphological-features.txt", "w") as txt_file:
        for line in f4:
            txt_file.write(" ".join(str(v) for v in line) + "\n")
            
    return f1, f2, f3, f4, labels, patient_num_beats


# DS: contains the patient list for load
# winL, winR: indicates the size of the window centred at R-peak at left and right side
# do_preprocess: indicates if preprocesing of remove baseline on signal is performed
def load_signal(DS, winL, winR, do_preprocess):

    class_ID = [[] for i in range(len(DS))]
    beat = [[] for i in range(len(DS))] # record, beat, lead
    R_poses = [ np.array([]) for i in range(len(DS))]
    Original_R_poses = [ np.array([]) for i in range(len(DS))]   
    valid_R = [ np.array([]) for i in range(len(DS))]
    my_db = mit_db()
    patients = []

    # Lists 
    # beats = []
    # classes = []
    # valid_R = np.empty([])
    # R_poses = np.empty([])
    # Original_R_poses = np.empty([])

    size_RR_max = 20

    pathDB = '../input/mitbih-database/'
    #pathDB2 = 'annotations'
    DB_name = ''
    fs = 360
    jump_lines = 1

    # Read files: signal (.csv )  annotations (.txt)    
    fRecords = list()
    fAnnotations = list()

    lst1 = os.listdir(pathDB)
    #lst2 = os.listdir(pathDB2 + DB_name)
    lst1.sort()
    #lst2.sort()
    for file in lst1:
         if file.endswith(".csv"):
            if int(file[0:3]) in DS:
                fRecords.append(file)
         if file.endswith(".txt"):
            if int(file[0:3]) in DS:
                fAnnotations.append(file)

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = [['N'], ['L'], ['R'], ['V'], ['A']]

    RAW_signals = []
    r_index = 0

    #for r, a in zip(fRecords, fAnnotations):
    for r in range(0, len(fRecords)):

        print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

        # 1. Read signalR_poses
        filename = pathDB + DB_name + fRecords[r]
        print(filename)
        f = open(filename, 'r')
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip first line!
        MLII_index = 1
        V1_index = 2
        if int(fRecords[r][0:3]) == 114:
            MLII_index = 2
            V1_index = 1

        MLII = []
        V1 = []
        for row in reader:
            MLII.append((int(row[MLII_index])))
            V1.append((int(row[V1_index])))
            #print(MLII)
            #print(V1)
        f.close()


        RAW_signals.append((MLII, V1)) ## NOTE a copy must be created in order to preserve the original signal
        # display_signal(MLII)

        # 2. Read annotations
        filename = pathDB + DB_name + fAnnotations[r]
        print(filename)
        f = open(filename, 'rb')
        next(f) # skip first line!

        annotations = []
        for line in f:
            annotations.append(line)
            #print(annotations)
        f.close()
        # 3. Preprocessing signal!
        if do_preprocess:
            #scipy.signal
            # median_filter1D
            baseline = medfilt(MLII, 71) 
            baseline = medfilt(baseline, 215) 

            # Remove Baseline
            for i in range(0, len(MLII)):
                MLII[i] = MLII[i] - baseline[i]

            # TODO Remove High Freqs

            # median_filter1D
            baseline = medfilt(V1, 71) 
            baseline = medfilt(baseline, 215) 

            # Remove Baseline
            for i in range(0, len(V1)):
                V1[i] = V1[i] - baseline[i]


        # Extract the R-peaks from annotations
        for a in annotations:
            aS = a.split()
            pos = int(aS[1])
            originalPos = int(aS[1])
            classAnttd = aS[2]
            if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
                index, value = max(enumerate(MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
                pos = (pos - size_RR_max) + index

            peak_type = 0
            if classAnttd.decode("utf-8") in MITBIH_classes:
                if(pos > winL and pos < (len(MLII) - winR)):
                    beat[r].append( (MLII[pos - winL : pos + winR], V1[pos - winL : pos + winR]))
                    for i in range(0,len(AAMI_classes)):
                        if classAnttd.decode("utf-8") in AAMI_classes[i]:
                            class_AAMI = i
                            break #exit loop
                    #convert class
                    class_ID[r].append(class_AAMI)

                    valid_R[r] = np.append(valid_R[r], 1)
                else:
                    valid_R[r] = np.append(valid_R[r], 0)
            else:
                valid_R[r] = np.append(valid_R[r], 0)
            
            R_poses[r] = np.append(R_poses[r], pos)
            Original_R_poses[r] = np.append(Original_R_poses[r], originalPos)
        
        #R_poses[r] = R_poses[r][(valid_R[r] == 1)]
        #Original_R_poses[r] = Original_R_poses[r][(valid_R[r] == 1)]

    # Set the data into a bigger struct that keep all the records!
    my_db.filename = fRecords

    my_db.raw_signal = RAW_signals
    my_db.beat = beat # record, beat, lead
    my_db.class_ID = class_ID
    my_db.valid_R = valid_R
    my_db.R_pos = R_poses
    my_db.orig_R_pos = Original_R_poses
    return my_db


