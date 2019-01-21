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
os.system("mkdir Training_Result")
os.system("mkdir Training_Result/"+"Hidden_" + str(num_of_hidden))
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
        file_result_all = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID) + "_HTRACE_ID_" + str(HTRACE_ID)  + "_Ref5_v3.txt"
        print(file_result_all)
        if os.path.exists(file_result_all):
            result_file_all = open(file_result_all, 'a')
            print("Appending")
        else:
            result_file_all = open(file_result_all, 'w')
        #
        for Num_seg_max in common.NUM_INPUT_FRAME_LIST:
            timesteps = Num_seg_max
            for EST_WIN in EST_WIN_LIST:
                # 
                tf.reset_default_graph()
                num_classes = 2 * EST_WIN

                X = tf.placeholder("float", [None, timesteps, num_input])
                Y = tf.placeholder("float", [None, num_classes])

                weights = {
                    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
                }
                biases = {
                    'out': tf.Variable(tf.random_normal([num_classes]))
                }

                def RNN(x, weights, biases):
                    # Prepare data shape to match `rnn` function requirements
                    # Current data input shape: (batch_size, timesteps, num_input)
                    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
                    # Unstack to get a list of 'timesteps' tensor of shape (batch_size, n_input)
                    x = tf.unstack(x, timesteps, 1)
                    # Define a lstm cell with tensorflow
                    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
                    lstm_cell = rnn.DropoutWrapper(lstm_cell,input_keep_prob=keep_rate_DROPOUT, output_keep_prob=keep_rate_DROPOUT, state_keep_prob=keep_rate_DROPOUT)
                    # Get lstm cell output
                    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                    # Linear activation, using rnn inner loop last output
                    return tf.matmul(outputs[-1], weights['out']) + biases['out']

                prediction = RNN(X, weights, biases)

                Prediction_MOS = tf.add(tf.multiply(prediction, 1), 0) 
                Label_MOS = tf.add(tf.multiply(Y, 1), 0)

                LOSS = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Prediction_MOS, Label_MOS))))
                PCC = tf.contrib.metrics.streaming_pearson_correlation(labels=Prediction_MOS, predictions=Label_MOS, name='pearson_r')

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(LOSS)
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

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
                    # Load train data
                    train_data_phi, train_label_phi = UTD.load_phi_from_file_3(List_head_trace_ref, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    train_data_theta, train_label_theta = UTD.load_theta_from_file_3(List_head_trace_ref, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    # Load test data
                    test_data_phi, test_label_phi = UTD.load_phi_from_file_3(List_head_trace, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    test_data_theta, test_label_theta = UTD.load_theta_from_file_3(List_head_trace, Num_sess_per_trace, Num_seg_max, DELAY, EST_WIN)
                    Num_sess_test = test_data_phi.shape[0]

                    for idx_Test in range(0,TestT):
                        for run_id in range(num_run):
                            print('Predicting phi ...')
                            #################### predict phi #####################
                            file_result = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref5_v3_HISWIN_" + str(Num_seg_max) +  "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "train_log_phi.txt"
                            result_text = open(file_result, 'w')
                            print("Number of videos:                 ", Num_sess, file=result_text)
                            print("Number of segments:               ", Num_seg_max, file=result_text)
                            print("Number of training videos:        ", Num_sess_train, file=result_text)
                            print("Number of test videos:            ", Num_sess_test, file=result_text)
                            print("Number of metrics:                ", Num_metrics, file=result_text)

                            ###################################################################

                            print("Learning rate:                   ", learning_rate, file=result_text)
                            print("Training steps:                  ", training_step, file=result_text)
                            print("Number of hidden:                ", num_hidden, file=result_text)
                            print("Number of input(Metrics):        ", num_input, file=result_text)
                            print("Time steps:                      ", timesteps, file=result_text)
                            print("Batch size:                      ", batch_size, file=result_text)
                            print("Keep Rate DROPOUT:               ", keep_rate_DROPOUT, file=result_text)
                            result_text.write("run_id\tEpoch\tMEAN_err_test\tRMSE_test\tMEAN_err_train\tRMSE_train\tnorm_RMSE_test\tnorm_RMSE_train\n")
                            with tf.Session() as seg:
                                # Run the initializer
                                seg.run(init)
                                # TRAIN
                                for step in range(1, training_step + 1):
                                    # batch_train = train_data_phi
                                    # batch_input = batch_train.reshape((batch_size, timesteps, num_input))
                                    seg.run(train_op, feed_dict={X: train_data_phi, Y: train_label_phi})
                                    if step % display_step == 0 or step == 1:
                                    # if step == training_step:
                                        # print("-----------------------------------------------------------")
                                        # print("Step " ,step,": ")
                                        preMOS_train_phi, lbMOS_train_phi, rmse_train_phi = seg.run([Prediction_MOS, Label_MOS, LOSS], 
                                            feed_dict={X: train_data_phi, Y: train_label_phi})
                                        # print(preMOS_train.shape, lbMOS_train.shape)
                                        # pcc_train = pcc(preMOS_train, lbMOS_train)[0]
                                        pcc_train = 0
                                        rmse2_train_phi, avg_err_train_phi, CI_train_phi, std_err_train_phi = UTD.calc_per_metric_2_phi(preMOS_train_phi, lbMOS_train_phi, len(lbMOS_train_phi), EST_WIN)

                                        # pcc_train = 0
                                        # print("RMSE train: ", rmse_train, ", PCC: ", pcc_train, ", rmse: ", rmse2_train, ", avg_err_train: ", avg_err_train)
                                        # TEST
                                        # print("TESTING...")
                                        batch_test = test_data_phi
                                        batch_test.reshape((-1, timesteps, num_input))
                                        preMOS_test_phi, lbMOS_test_phi, rmse_test_phi = seg.run([Prediction_MOS, Label_MOS, LOSS], 
                                            feed_dict = {X: batch_test, Y: test_label_phi})
                                        # pcc_test = pcc(preMOS_test, lbMOS_test)[0]
                                        pcc_test = 0
                                        rmse2_test_phi, avg_err_test_phi, CI_test_phi, std_err_test_phi = UTD.calc_per_metric_2_phi(preMOS_test_phi, lbMOS_test_phi, len(lbMOS_test_phi), EST_WIN)
                                        # pcc_test = 0
                                        print(run_id, ',', step, ',', avg_err_test_phi,',',rmse2_test_phi,',',avg_err_train_phi, ',', rmse2_train_phi,',', rmse_train_phi, ',', rmse_test_phi)
                                        print(CI_train_phi, ',', CI_test_phi)
                                        result_text.write("%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(run_id, step, rmse2_test_phi, avg_err_test_phi, CI_test_phi, std_err_test_phi, rmse2_train_phi, avg_err_train_phi, CI_train_phi, std_err_train_phi))
                                        result_text.flush()
                            result_text.close()
                            # quit()
                            print('Predicting theta ...')
                            ###################### predict theta ###########################
                            file_result = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref5_v3_HISWIN_" + str(Num_seg_max) +  "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "train_log_theta.txt"
                            result_text = open(file_result, 'w')
                            print("Number of videos:                 ", Num_sess, file=result_text)
                            print("Number of segments:               ", Num_seg_max, file=result_text)
                            print("Number of training videos:        ", Num_sess_train, file=result_text)
                            print("Number of test videos:            ", Num_sess_test, file=result_text)
                            print("Number of metrics:                ", Num_metrics, file=result_text)

                            ###################################################################

                            print("Learning rate:                   ", learning_rate, file=result_text)
                            print("Training steps:                  ", training_step, file=result_text)
                            print("Number of hidden:                ", num_hidden, file=result_text)
                            print("Number of input(Metrics):        ", num_input, file=result_text)
                            print("Time steps:                      ", timesteps, file=result_text)
                            print("Batch size:                      ", batch_size, file=result_text)
                            print("Keep Rate DROPOUT:               ", keep_rate_DROPOUT, file=result_text)
                            result_text.write("run_id\tEpoch\tMEAN_err_test\tRMSE_test\tMEAN_err_train\tRMSE_train\tnorm_RMSE_test\tnorm_RMSE_train\n")
                            with tf.Session() as seg:
                                # Run the initializer
                                seg.run(init)
                                # TRAIN
                                for step in range(1, training_step + 1):
                                    # batch_train = train_data_phi
                                    # batch_input = batch_train.reshape((batch_size, timesteps, num_input))
                                    seg.run(train_op, feed_dict={X: train_data_theta, Y: train_label_theta})
                                    if step % display_step == 0 or step == 1:
                                    # if step == training_step:
                                        # print("-----------------------------------------------------------")
                                        # print("Step " ,step,": ")
                                        preMOS_train_theta, lbMOS_train_theta, rmse_train_theta = seg.run([Prediction_MOS, Label_MOS, LOSS], 
                                            feed_dict={X: train_data_theta, Y: train_label_theta})
                                        # print(preMOS_train.shape, lbMOS_train.shape)
                                        # pcc_train = pcc(preMOS_train, lbMOS_train)[0]
                                        pcc_train = 0
                                        rmse2_train_theta, avg_err_train_theta, CI_train_theta, std_err_train_theta = UTD.calc_per_metric_2_theta(preMOS_train_theta, lbMOS_train_theta, len(lbMOS_train_theta), EST_WIN)

                                        # pcc_train = 0
                                        # print("RMSE train: ", rmse_train, ", PCC: ", pcc_train, ", rmse: ", rmse2_train, ", avg_err_train: ", avg_err_train)
                                        # TEST
                                        # print("TESTING...")
                                        batch_test = test_data_theta
                                        batch_test.reshape((-1, timesteps, num_input))
                                        preMOS_test_theta, lbMOS_test_theta, rmse_test_theta = seg.run([Prediction_MOS, Label_MOS, LOSS], 
                                            feed_dict = {X: batch_test, Y: test_label_theta})
                                        # pcc_test = pcc(preMOS_test, lbMOS_test)[0]
                                        pcc_test = 0
                                        rmse2_test_theta, avg_err_test_theta, CI_test_theta, std_err_test_theta = UTD.calc_per_metric_2_theta(preMOS_test_theta, lbMOS_test_theta, len(lbMOS_test_theta), EST_WIN)
                                        # pcc_test = 0
                                        print(run_id, ',', step, ',', avg_err_test_theta,',',rmse2_test_theta,',',avg_err_train_theta, ',', rmse2_train_theta,',', rmse_train_theta, ',', rmse_test_theta)
                                        print(CI_train_theta, ',', CI_test_theta)
                                        result_text.write("%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(run_id, step, rmse2_test_theta, avg_err_test_theta, CI_test_theta, std_err_test_theta, rmse2_train_theta, avg_err_train_theta, CI_train_theta, std_err_train_theta))
                                        result_text.flush()
                            result_text.close()

                            # quit()
                        # 
            #             rmse_phi, avg_err_phi, CI_phi, std_phi = ref.calc_per_metric_2(err_VP_phi)
            #             rmse_theta, avg_err_theta, CI_theta, std_theta = ref.calc_per_metric_2(err_VP_theta)

                        # combine
                        # FULL
                        file_result_per_frame = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref5_v3_HISWIN_" + str(Num_seg_max) +  "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "_per_frame.txt"
                        print(file_result_per_frame)
                        result_text_per_frame = open(file_result_per_frame, 'w')
                        result_text_per_frame.write("phi\ttheta\test_phi\test_theta\terr_phi\terr_theta\terr\n")
                        # SHORT
                        file_result_per_frame_short = "Training_Result/"+"Hidden_" + str(num_of_hidden) +"/VID_" + str(VIDEO_ID)+ "_HTRACE_ID_" + str(HTRACE_ID) + "_Ref5_v3_HISWIN_" + str(Num_seg_max) +  "_DELAY_" + str(DELAY) + "_ESTWIN_" + str(EST_WIN) + "_per_frame_short.txt"
                        print(file_result_per_frame_short)
                        result_text_per_frame_short = open(file_result_per_frame_short, 'w')
                        result_text_per_frame_short.write("phi\ttheta\test_phi\test_theta\terr_phi\terr_theta\terr\n")
                        # 
                        err_VP = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        black_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        black_tile_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        redun_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        visi_tile_BR_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        est_visi_tile_num = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        redun_tile_ratio = np.zeros([Num_sess_test, EST_WIN], dtype='f')
                        for ses_id in range(Num_sess_test):
                            for fid in range(EST_WIN):
                                phi[fid] = UTD.loc_2_angle2_phi(test_label_phi[ses_id][fid*2:(fid+1)*2])
                                theta[fid] = UTD.loc_2_angle2_theta(test_label_theta[ses_id][fid*2:(fid+1)*2])
                                est_phi[fid] = UTD.loc_2_angle2_phi(preMOS_test_phi[ses_id][fid*2:(fid+1)*2])
                                est_theta[fid] = UTD.loc_2_angle2_theta(preMOS_test_theta[ses_id][fid*2:(fid+1)*2])
                                # phi[fid] =  float(ref.denorm_phi(test_label_phi[ses_id][fid]))
                                # theta[fid] = float(ref.denorm_theta(test_label_theta[ses_id][fid]))
                                # est_phi[fid] = float(ref.denorm_phi(preMOS_test_phi[ses_id][fid]))
                                # est_theta[fid] = float(ref.denorm_theta(preMOS_test_theta[ses_id][fid]))
                                # err_phi = err_phi_test[ses_id][fid]
                                # err_theta = err_theta_test[ses_id][fid]
                                err_phi = 0
                                err_theta = 0

                                # print(phi,',', theta, ',', est_phi, ',', est_theta)
                                # a = common.ang_dist(np.deg2rad(phi), np.deg2rad(theta), np.deg2rad(est_phi), np.deg2rad(est_theta))
                                err_VP[ses_id][fid] = common.ang_dist(np.deg2rad(phi[fid]), np.deg2rad(theta[fid]), np.deg2rad(est_phi[fid]), np.deg2rad(est_theta[fid]))
                                if np.isnan(err_VP[ses_id][fid]):
                                    # print('sess_id:',ses_id,', fid:', fid)
                                    err_VP[ses_id][fid] = 0

                                result_text_per_frame.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(phi[fid], theta[fid], est_phi[fid], est_theta[fid], err_phi, err_theta, err_VP[ses_id][fid]))
                                if ses_id % 6 == 0:
                                    result_text_per_frame_short.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(phi[fid], theta[fid], est_phi[fid], est_theta[fid], err_phi, err_theta, err_VP[ses_id][fid]))
                                # print(err_VP[ses_id][fid])
                                # quit()
                            # black_tile_num[ses_id], black_tile_ratio[ses_id], redun_tile_num[ses_id], visi_tile_BR_ratio[ses_id] = common.calc_err_tile(phi, theta, est_phi, est_theta)
                            # quit()
                            black_tile_num[ses_id], black_tile_ratio[ses_id], redun_tile_num[ses_id], visi_tile_BR_ratio[ses_id],est_visi_tile_num[ses_id] = common.calc_err_tile(phi, theta, est_phi, est_theta)
                            redun_tile_ratio[ses_id] = redun_tile_num[ses_id]/est_visi_tile_num[ses_id]
                            # print(black_tile_num[ses_id])
                            # quit()
                        result_text_per_frame.close()
                        result_text_per_frame_short.close()
                        # 
                        rmse, avg_err, CI, std = ref.calc_per_metric_2(err_VP)

                        result_file_all.write("%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %(Num_seg_max, EST_WIN, DELAY, rmse2_test_phi, avg_err_test_phi, CI_test_phi, std_err_test_phi,  rmse2_test_theta, avg_err_test_theta, CI_test_theta, std_err_test_theta, rmse, avg_err, CI, std, np.mean(black_tile_ratio), np.mean(visi_tile_BR_ratio), np.mean(redun_tile_num)/common.NO_TILE, np.mean(redun_tile_ratio[ses_id])))
                        result_file_all.flush()
                        # quit()
        result_file_all.close()
            # quit()


