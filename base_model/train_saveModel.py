import os
import argparse
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model

from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.lib.io import file_io


def run_experiment(hparams):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    random.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    train_batch_size = hparams.train_batch_size
    test_batch_size = hparams.test_batch_size

    with file_io.FileIO(",".join(hparams.train_files), 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    best_auc = 0.0
    def calc_auc(raw_arr):
        """Summary

        Args:
            raw_arr (TYPE): Description

        Returns:
            TYPE: Description
        """
        # sort by pred value, from small to big
        arr = sorted(raw_arr, key=lambda d:d[2])

        auc = 0.0
        fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
        for record in arr:
            fp2 += record[0] # noclick
            tp2 += record[1] # click
            auc += (fp2 - fp1) * (tp2 + tp1)
            fp1, tp1 = fp2, tp2

        # if all nonclick or click, disgard
        threshold = len(arr) - 1e-3
        if tp2 > threshold or fp2 > threshold:
            return -0.5

        if tp2 * fp2 > 0.0:  # normal auc
            return (1.0 - auc / (2.0 * tp2 * fp2))
        else:
            return None

    def _auc_arr(score):
      score_p = score[:,0]
      score_n = score[:,1]
      #print "============== p ============="
      #print score_p
      #print "============== n ============="
      #print score_n
      score_arr = []
      for s in score_p.tolist():
        score_arr.append([0, 1, s])
      for s in score_n.tolist():
        score_arr.append([1, 0, s])
      return score_arr
    def _eval(sess, model, best_auc):
      auc_sum = 0.0
      score_arr = []
      for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
      test_gauc = auc_sum / len(test_set)
      Auc = calc_auc(score_arr)
      if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, hparams.job_dir)
      return test_gauc, Auc

    def _export(sess, uij, lr, logits_all, path):

        # saver = tf.train.Saver()
        # saver.restore(sess, save_path=path)

        tf.saved_model.simple_save(
            sess,
            hparams.export_dir,
            inputs={
                'u': tf.convert_to_tensor(uij[0], dtype=tf.float16),
                'hist_i': tf.convert_to_tensor(uij[3], dtype=tf.float16),
                'sl': tf.convert_to_tensor(lr, dtype=tf.float16)
            },
            outputs={'logits_all': tf.convert_to_tensor(logits_all, dtype=tf.float16)}
        )

    def build_i_map(keys):
        """
        Make inverse map for keys: i -> item
        :param keys: listing items in order.
        :return:
        """
        return dict(zip(range(len(keys), keys)))

    def restore_info(uij, predicted, dic):
        iu, ii, _, ihist_i, _ = uij

        u = dic['deal_key'][iu]
        for i in range(len(iu[0])):
            u = dic['deal_key'][iu[i]]

    def _predict(sess):
        with file_io.FileIO(",".join(hparams.train_files), 'rb') as f:
            wepick_data = pickle.load(f)

        total_model = 0
        total_time = 0
        start = time.time()

        with file_io.FileIO(",".join(hparams.pred_out_path), 'w') as pred_f:
            for _, uij in DataInputTest(test_set, test_batch_size):
                outputs = []
                inf_start = time.time()
                users, histories, lengths = uij[0], uij[3], uij[4]
                predicted = model.test(sess, users, histories, lengths)
                inf_end = time.time()
                total_model += inf_end - inf_start

                for k in range(len(users)):
                    u = wepick_data['user_key'][users[k]]
                    hist = list(map(lambda x: wepick_data['deal_key'][x], histories[k][0:lengths[k]]))
                    sort_i = np.argsort(predicted[k,:])
                    sort_i = np.fliplr([sort_i])[0]
                    order = list(map(lambda x: (wepick_data['deal_key'][x], predicted[k, x]), sort_i))
                    outputs.append((u, hist, order))

                for u, hist, order in outputs:
                    h = '-'.join(map(lambda x: str(x), hist))
                    s = ':'.join(map(lambda x: "{}/{:.2f}".format(x[0], x[1]), order[:30]))
                    pred_f.write("{},{},{}\n".format(u, h, s))

        total_time += (time.time() - start)
        sys.stderr.wirte("Elapsed total {}: model {}\n".format(total_time, total_model))

    def restore(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


    model = Model(user_count, item_count, cate_count, cate_list, hparams.variable_strategy)


    # gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

      if hparams.run_mode == "test":

        restore(sess, hparams.job_dir)
        _predict(sess)
        return 0

      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      writer = tf.summary.FileWriter('log', sess.graph)

      print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model, best_auc))
      sys.stdout.flush()
      lr = 1.0
      start_time = time.time()
      # for _ in range(50):
      for _ in range(1):

        random.shuffle(train_set)

        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):

          loss, logits_all = model.train(sess, uij, lr, hparams.variable_strategy)

          loss_sum += loss

          if model.global_step.eval() % 1000 == 0:
            test_gauc, Auc = _eval(sess, model, best_auc)
            print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                  (model.global_epoch_step.eval(), model.global_step.eval(),
                   loss_sum / 1000, test_gauc, Auc))
            sys.stdout.flush()
            loss_sum = 0.0

          if model.global_step.eval() % 336000 == 0:
            lr = 0.1
        break

        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time()-start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()


      print('best test_gauc:', best_auc)
      _export(sess, uij, lr, logits_all, hparams.job_dir)
      sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--run-mode',
        help='train or test',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=32
    )
    parser.add_argument(
        '--test-batch-size',
        help='Batch size for test steps',
        type=int,
        default=512
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--export-dir',
        help='Where to locate savedModel result',
        required=True
    )
    parser.add_argument(
        '--variable-strategy',
        help='Where to locate variable operations',
        type=str,
        default='CPU'
    )
    parser.add_argument(
        '--pred-out-path',
        help='Where to locate predicted result',
        nargs="+",
        required=False
    )

    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO'
    )

    args = parser.parse_args()

    tf.logging.set_verbosity(args.verbosity)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
