import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.stats import pearsonr as pcc
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import argparse
import sys
print(sys.version)
import os
from F_get_indxSets_515seq_NoVal_all100Rand import F_get_indxSets
import pdb
import UTD2017 as UTD
import common
import refEstimationMethod as ref


#fileDir             = os.path.dirname(os.path.realpath('__file__'))
#DATABASE_DIR        = os.path.join(fileDir, '../Dataset')
#parser              = argparse.ArgumentParser()
#parser.add_argument("NUM_HIDDEN", type=int)
#parser.add_argument("TestT", type=int) #Number of randomized test times
#parser.add_argument("flag_rand", type=int)
#parser.add_argument("IdxMeArr", type=int)
#args                = parser.parse_args()
#num_of_hidden       = args.NUM_HIDDEN
#TestT               = args.TestT
#flag_rand           = args.flag_rand
#IdxMeArr            = args.IdxMeArr

############################### READ DATABASE (INPUT and EXPECTED OUTPUT) and NORMALIZE Data_woFlag %%%%%%%%%%%%%%%%%%%%%%%%

List_head_trace     = common.List_head_trace
Num_head_trace      = len(List_head_trace)
Num_sess_per_trace  = 300
Num_metrics         = 2 # Number of features
Num_sess            = Num_sess_per_trace * Num_head_trace
Num_sess_train      = int(Num_sess * common.train_percent)
Num_sess_test       = int(Num_sess - Num_sess_train)
Num_seg_max         = common.NUM_INPUT_FRAME # Number of history frames
flag_rand           = 0
TestT               = 1 # Number of randomized test times
EST_WIN_LIST        = common.EST_WIN_LIST # Number of frames to be predicted


########## Architecture of LSTM ################
num_of_hidden       = common.num_of_hidden
learning_rate       = common.learning_rate
training_step       = common.training_step
display_step = common.display_step
keep_rate_DROPOUT = 1
num_run = common.num_run # Number of runs per setting
# 
num_hidden = num_of_hidden
num_input = Num_metrics
timesteps = Num_seg_max
batch_size = Num_sess_train
# 
for VIDEO_ID in common.VIDEO_ID_LIST:
    # List_head_trace = common.List_head_trace_all_video[VIDEO_ID,0:common.NUM_HTRACE]
    for TMP_HTRACE_ID in range(common.Num_head_trace[VIDEO_ID]):
        # print('HTRACE_ID:', common.List_head_trace_id[VIDEO_ID][1])
        HTRACE_ID = common.List_head_trace_id[VIDEO_ID][TMP_HTRACE_ID]
        htrace_name = "dat/xyz_vid_" + str(VIDEO_ID) + "_uid_" + str(HTRACE_ID) + ".txt"
        List_head_trace = [htrace_name]
        print('Test head trace:')
        print(List_head_trace)
        # Reference head traces
        List_head_trace_ref = []
        for trace_id in range(common.Num_head_trace[VIDEO_ID]):
            if trace_id != TMP_HTRACE_ID:
                htrace_name = "dat/xyz_vid_" + str(VIDEO_ID) + "_uid_" + str(common.List_head_trace_id[VIDEO_ID][trace_id]) + ".txt"
                List_head_trace_ref = np.concatenate((List_head_trace_ref, [htrace_name]), axis=0)
        print('Ref. head trace:')
        Ref_trace_num = len(List_head_trace_ref)
        print(List_head_trace_ref)
        print(Ref_trace_num)
        # quit()
        # continue
        # 
        file_result_all = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID) + "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref4_v3.txt"
        print(file_result_all)
        # if os.path.exists(file_result_all):
        #     result_file_all = open(file_result_all, 'a')
        #     print("Appending")
        # else:
        #     result_file_all = open(file_result_all, 'w')
        result_file_all = open(file_result_all, 'w')
        #
        for Num_seg_max in common.NUM_INPUT_FRAME_LIST:
            # 
            for EST_WIN in EST_WIN_LIST:

                phi = np.zeros([EST_WIN], dtype='f')
                theta = np.zeros([EST_WIN], dtype='f')
                est_phi = np.zeros([EST_WIN], dtype='f')
                est_theta = np.zeros([EST_WIN], dtype='f')

            ################### START TRAINNING AND TESTING #####################
            ###### CREATE TRAINNING,  AND TEST SETS %%%%%%%%%%%%%%


            # EST_WIN_LIST = np.array([1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
            # EST_WIN_LIST = np.array([30])
                DELAY_LIST = common.DELAY_LIST
                # 
                for DELAY in DELAY_LIST:
                    # Load test data
                    Video_data, Label = ref.load_phi_from_file(List_head_trace, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    Video_data_theta, Label_theta = ref.load_theta_from_file(List_head_trace, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    # Ref. head traces data
                    print('Load ref trace data:')
                    # data
                    Ref_data_phi = np.zeros([Ref_trace_num, Num_sess_per_trace, Num_seg_max, 1])
                    Ref_data_theta = np.zeros([Ref_trace_num, Num_sess_per_trace, Num_seg_max, 1])
                    # label
                    Ref_label_phi = np.zeros([Ref_trace_num, Num_sess_per_trace, EST_WIN])
                    Ref_label_theta = np.zeros([Ref_trace_num, Num_sess_per_trace, EST_WIN])
                    # 
                    for ref_htrace_id in range(Ref_trace_num):
                        ref_htrace = List_head_trace_ref[ref_htrace_id]
                        print(ref_htrace)
                        Ref_data_phi[ref_htrace_id], Ref_label_phi[ref_htrace_id] = ref.load_phi_from_file([ref_htrace], Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                        Ref_data_theta[ref_htrace_id], Ref_label_theta[ref_htrace_id] = ref.load_theta_from_file([ref_htrace], Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)

                    # quit()
                    # print(len(Video_data))
                    # print(len(Label))
                    # print(Label[0])
                    # quit()
                    for idx_Test in range(0,TestT):
                        if flag_rand == 1:
                            shuffle_sess    = np.random.permutation(Num_sess)
                            id_train        = shuffle_sess[0:Num_sess_train]
                            id_test         = shuffle_sess[Num_sess_train:]
                        elif flag_rand == 2: ## 
                            id_train,id_test = F_get_indxSets(idx_Test)
                            id_train =np.array(id_train)
                            id_test  =np.array(id_test)
                        else:
                            id_train = np.arange(0,Num_sess_train)
                            id_test  = np.arange(Num_sess_train,Num_sess)


                        video_test  = Video_data
                        Label_test  = Label

                        video_test_theta  = Video_data_theta
                        Label_test_theta  = Label_theta

                        Num_sess_test = Num_sess_per_trace

                        # print(video_train.shape)
                        # print(video_test.shape)


                        # est_VP_ref1, err_VP_ref1, avg_err_ref1, rmse_ref1 = ref.lastViewportAll(video_test, Label_test, Num_sess_test, Num_seg_max)
                        # print('ref1: rmse: ', rmse_ref1, ', avg_err: ', avg_err_ref1)
                        # quit()

                        # viewport estimation
                        # phi
                        est_VP_phi, err_VP_phi = ref.LRAll_phi(video_test, Label_test, Num_sess_test, Num_seg_max, DELAY, EST_WIN)
                        # theta
                        est_VP_theta, err_VP_theta = ref.LRAll_theta(video_test_theta, Label_test_theta, Num_sess_test, Num_seg_max, DELAY, EST_WIN)
                        # 
                        rmse_phi, avg_err_phi, CI_phi, std_phi = ref.calc_per_metric_2(err_VP_phi)
                        rmse_theta, avg_err_theta, CI_theta, std_theta = ref.calc_per_metric_2(err_VP_theta)

                        # combine
                        # result_full
                        file_result_per_frame = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref4_HISWIN_" + str(Num_seg_max) + "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "_per_frame.txt"
                        print(file_result_per_frame)
                        result_text_per_frame = open(file_result_per_frame, 'w')
                        result_text_per_frame.write("phi\ttheta\test_phi\test_theta\terr_phi\terr_theta\terr\n")
                        # result_short
                        file_result_per_frame_short = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref4_HISWIN_" + str(Num_seg_max) + "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "_per_frame_short.txt"
                        print(file_result_per_frame_short)
                        result_text_per_frame_short = open(file_result_per_frame_short, 'w')
                        result_text_per_frame_short.write("phi\ttheta\test_phi\test_theta\terr_phi\terr_theta\terr\n")
                        err_VP = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        err_tile = np.zeros([Num_sess_test, EST_WIN+1], dtype=np.uint8)
                        black_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        black_tile_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        redun_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        visi_tile_BR_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        est_visi_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        redun_tile_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        k_nearest_vp = np.zeros([Num_sess_test, EST_WIN, common.K_NEAREST, 2], dtype='int')
                        for ses_id in range(Num_sess_test):
                            for fid in range(EST_WIN):
                                phi[fid] =  float(ref.denorm_phi(Label_test[ses_id][fid]))
                                theta[fid] = float(ref.denorm_theta(Label_theta[ses_id][fid]))
                                est_phi[fid] = float(ref.denorm_phi(est_VP_phi[ses_id][fid]))
                                est_theta[fid] = float(ref.denorm_theta(est_VP_theta[ses_id][fid]))
                                err_phi = err_VP_phi[ses_id][fid]
                                err_theta = err_VP_theta[ses_id][fid]

                                # print(phi,',', theta, ',', est_phi, ',', est_theta)
                                # a = common.ang_dist(np.deg2rad(phi), np.deg2rad(theta), np.deg2rad(est_phi), np.deg2rad(est_theta))
                                err_VP[ses_id][fid] = common.ang_dist(np.deg2rad(phi[fid]), np.deg2rad(theta[fid]), np.deg2rad(est_phi[fid]), np.deg2rad(est_theta[fid]))
                                if np.isnan(err_VP[ses_id][fid]):
                                    # print('sess_id:',ses_id,', fid:', fid)
                                    err_VP[ses_id][fid] = 0
                                # print(err_VP[ses_id][fid])
                                # quit()
                            # black_tile_num[ses_id], black_tile_ratio[ses_id], redun_tile_num[ses_id], visi_tile_BR_ratio[ses_id] = common.calc_err_tile(phi, theta, est_phi, est_theta)
                            # a = Ref_label_phi[:,ses_id,:]
                            # print(a.shape)
                            # quit()
                            black_tile_num[ses_id], black_tile_ratio[ses_id], redun_tile_num[ses_id], visi_tile_BR_ratio[ses_id], est_visi_tile_num[ses_id],k_nearest_vp[ses_id] = common.calc_err_tile_KNN(phi, theta, est_phi, est_theta, Ref_label_phi[:,ses_id,:], Ref_label_theta[:,ses_id,:], common.K_NEAREST)
                            for fid in range(EST_WIN):
                                result_text_per_frame.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(phi[fid], theta[fid], est_phi[fid], est_theta[fid], err_VP_phi[ses_id][fid], err_VP_theta[ses_id][fid], err_VP[ses_id][fid]))
                                if ses_id % 6 == 0:
                                    result_text_per_frame_short.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t" %(phi[fid], theta[fid], est_phi[fid], est_theta[fid], err_VP_phi[ses_id][fid], err_VP_theta[ses_id][fid], err_VP[ses_id][fid]))
                                    for k in range(common.K_NEAREST):
                                        result_text_per_frame_short.write("%d\t%d\t" %(k_nearest_vp[ses_id][fid][k][0], k_nearest_vp[ses_id][fid][k][1]))
                                    result_text_per_frame_short.write("\n")
                            redun_tile_ratio[ses_id] = redun_tile_num[ses_id]/est_visi_tile_num[ses_id]
                            # print(black_tile_num[ses_id])
                            # quit()
                        result_text_per_frame.close()
                        result_text_per_frame_short.close()
                        # quit()
                        # 
                        rmse, avg_err, CI, std = ref.calc_per_metric_2(err_VP)
                        result_file_all.write("%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(Num_seg_max, EST_WIN, DELAY, rmse_phi, avg_err_phi, CI_phi, std_phi, rmse_theta, avg_err_theta, CI_theta, std_theta, rmse, avg_err, CI, std, np.mean(black_tile_ratio), np.mean(visi_tile_BR_ratio), np.mean(redun_tile_num)/common.NO_TILE, np.mean(redun_tile_ratio)))
                        result_file_all.flush()
        result_file_all.close()
            # quit()


