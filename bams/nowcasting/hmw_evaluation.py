import numpy as np
import logging
import os
from collections import namedtuple
from nowcasting.config import cfg
#from hko_data.hko_iterator import get_exclude_mask
#from nowcasting.helpers.msssim import _SSIMForMultiScale
from nowcasting.helpers.msssim import _SSIMForMultiScale
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def pixel_to_dbz(img):  ##### [0, 1] >>  [0, 80] dbz 
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 80.0


def dbz_to_pixel(img):  ### [0, 80] dbz >> [0, 1] 
    """

    Parameters
    ----------
    tbb_img : np.ndarray

    Returns
    -------

    """
    return np.clip(img / 80.0, a_min=0.0, a_max=1.0)



def get_hit_miss_counts(prediction, truth, mask=None, thresholds=None, sum_batch=False):    #### be vital for evaluation !!!!
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.HKO.EVALUATION.THRESHOLDS  ## config.py
    assert 5 == prediction.ndim   ## dimension = 5 
    assert 5 == truth.ndim        ## dimension = 5 
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1   ## third dimension = 1
    thresholds = dbz_to_pixel(np.array(thresholds, dtype=np.float32)
                              .reshape((1, 1, len(thresholds), 1, 1)))
    #print("correct(evaluation:86)")
    #prediction[prediction<0.1] = 0
    bpred = (prediction >= thresholds)
    #print("bpred", bpred, bpred.shape)
    btruth = (truth >= thresholds)
    bpred_n = np.logical_not(bpred)
    btruth_n = np.logical_not(btruth)
    if sum_batch:
        summation_axis = (1, 3, 4)
    else:
        summation_axis = (3, 4)
    if mask is None:
        hits = np.logical_and(bpred, btruth).sum(axis=summation_axis)
        misses = np.logical_and(bpred_n, btruth).sum(axis=summation_axis)
        false_alarms = np.logical_and(bpred, btruth_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(bpred_n, btruth_n).sum(axis=summation_axis)
    else:
        hits = np.logical_and(np.logical_and(bpred, btruth), mask)\
            .sum(axis=summation_axis)
        misses = np.logical_and(np.logical_and(bpred_n, btruth), mask)\
            .sum(axis=summation_axis)
        false_alarms = np.logical_and(np.logical_and(bpred, btruth_n), mask)\
            .sum(axis=summation_axis)
        correct_negatives = np.logical_and(np.logical_and(bpred_n, btruth_n), mask)\
            .sum(axis=summation_axis)
    return hits, misses, false_alarms, correct_negatives



def get_correlation(prediction, truth):
    """

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------

    """
    assert truth.shape == prediction.shape   ## same
    assert 5 == prediction.ndim       ### dimension
    assert prediction.shape[2] == 1   ### third dimension = 1
    eps = 1E-12
    ret = (prediction * truth).sum(axis=(3, 4)) / (
        np.sqrt(np.square(prediction).sum(axis=(3, 4))) * np.sqrt(np.square(truth).sum(axis=(3, 4))) + eps)
    ret = ret.sum(axis=(1, 2))
    return ret


def get_PSNR(prediction, truth):      ### PSNR=Peak Signal-to-Noise Ratio
    """Peak Signal Noise Ratio

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    mse = np.square(prediction - truth).mean(axis=(2, 3, 4))
    ret = 10.0 * np.log10(1.0 / mse)
    ret = ret.sum(axis=1)
    return ret


def get_SSIM(prediction, truth):     ### SSIM=Structure SIMilarity
    """Calculate the SSIM score following
    [TIP2004] Image Quality Assessment: From Error Visibility to Structural Similarity

    Same functionality as
    https://github.com/coupriec/VideoPredictionICLR2016/blob/master/image_error_measures.lua#L50-L75

    We use nowcasting.helpers.msssim, which is borrowed from Tensorflow to do the evaluation

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    seq_len = prediction.shape[0]
    batch_size = prediction.shape[1]
    prediction = prediction.reshape((prediction.shape[0] * prediction.shape[1],
                                     prediction.shape[3], prediction.shape[4], 1))
    truth = truth.reshape((truth.shape[0] * truth.shape[1],
                           truth.shape[3], truth.shape[4], 1))
    ssim, cs = _SSIMForMultiScale(img1=prediction, img2=truth, max_val=1.0)
    print(ssim.shape)
    ret = ssim.reshape((seq_len, batch_size)).sum(axis=1)
    return ret


def get_GDL(prediction, truth, mask, sum_batch=False): ## the masked gradient difference loss
    """Calculate the masked gradient difference loss

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use

    Returns
    -------
    gdl : np.ndarray
        Shape: (seq_len,) or (seq_len, batch_size)
    """
    prediction_diff_h = np.abs(np.diff(prediction, axis=3))
    prediction_diff_w = np.abs(np.diff(prediction, axis=4))
    gt_diff_h = np.abs(np.diff(truth, axis=3))
    gt_diff_w = np.abs(np.diff(truth, axis=4))
    mask_h = mask[:, :, :, :-1, :] * mask[:, :, :, 1:, :]
    mask_w = mask[:, :, :, :, :-1] * mask[:, :, :, :, 1:]
    gd_h = np.abs(prediction_diff_h - gt_diff_h)
    gd_w = np.abs(prediction_diff_w - gt_diff_w)
    gd_h[:] *= mask_h
    gd_w[:] *= mask_w
    summation_axis = (1, 2, 3, 4) if sum_batch else (2, 3, 4)
    gdl = np.sum(gd_h, axis=summation_axis) + np.sum(gd_w, axis=summation_axis)
    return gdl


def get_balancing_weights(data, mask, base_balancing_weights=None, thresholds=None):
    #####  balance weight #####
    if thresholds is None:
        thresholds = cfg.HKO.EVALUATION.THRESHOLDS
    if base_balancing_weights is None:
        base_balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
    thresholds = dbz_to_pixel(np.array(thresholds, dtype=np.float32)
                                   .reshape((1, 1, 1, 1, 1, len(thresholds))))
    weights = np.ones_like(data) * base_balancing_weights[0]
    threshold_mask = np.expand_dims(data, axis=5) >= thresholds
    base_weights = np.diff(np.array(base_balancing_weights, dtype=np.float32))\
        .reshape((1, 1, 1, 1, 1, len(base_balancing_weights) - 1))
    weights += (threshold_mask * base_weights).sum(axis=-1)
    weights *= mask
    return weights


try:
    from nowcasting.numba_accelerated import get_GDL_numba, get_hit_miss_counts_numba,\
        get_balancing_weights_numba
except:
    # get_GDL_numba = get_GDL
    # get_hit_miss_counts_numba = get_hit_miss_counts
    # get_balancing_weights_numba = get_balancing_weights
    # print("Numba has not been installed correctly!")
    raise ImportError("Numba has not been installed correctly!")

class HMWEvaluation(object):
    def __init__(self, seq_len, use_central, no_ssim=True, threholds=None,
                 central_region=None):
        if central_region is None:
            central_region = cfg.HKO.EVALUATION.CENTRAL_REGION
        self._thresholds = cfg.HKO.EVALUATION.THRESHOLDS if threholds is None else threholds
        self._seq_len = seq_len          ## out_seq_len=12
        self._no_ssim = no_ssim
        self._use_central = use_central
        self._central_region = central_region
        #self._exclude_mask = get_exclude_mask()
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
                                                 dtype=np.int)
        self._mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._mae = np.zeros((self._seq_len, ), dtype=np.float32)
        self._bias = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mae = np.zeros((self._seq_len,), dtype=np.float32)
        self._gdl = np.zeros((self._seq_len,), dtype=np.float32)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._datetime_dict = {}
        self._total_batch_num = 0

    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        self._bias[:] = 0 
        self._gdl[:] = 0
        self._ssim[:] = 0
        self._total_batch_num = 0


    def cal_fpr_tpr(self, thresholds, pred, gt):
        tpr = []
        fpr = []
        for i in thresholds:
            print("threshold(eval:294)", i)
            ypred = np.where(pred>i, 1, 0)  # 把抽取的预测值当做分类阈值
            y = np.where(gt>i,1,0)
            cm = confusion_matrix(y.ravel(), ypred.ravel())  # 计算混淆矩阵，方便算fpr tpr
            print(cm)
            tpr.append(cm[1, 1] / (cm[1, 0] + cm[1, 1]))  # 真 正例率
            fpr.append(cm[0, 1] / (cm[0, 1] + cm[0, 0]))  # 假 正例率
        #tpr.sort()
        #fpr.sort()
        return tpr, fpr

    def plot_roc(self, gt, pred,save_path):
        # 第一步、计算fpr tpr
        #thresholds = np.random.choice(pred.ravel(), 10, replace=True).tolist()  # 从样本预测值里面抽取一些
        #thresholds.sort()  # 把样本预测值从小到大排序
        print(np.max(pred), np.min(pred), np.max(gt), np.min(gt))
        #thresholds = dbz_to_pixel(np.array(self._thresholds, dtype=np.float32))
        thresholds = np.linspace(0.,0.9,8)

        tpr1, fpr1 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:5],gt=gt[:5])
        print(tpr1, fpr1)
        tpr2, fpr2 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:10],gt=gt[:10])
        print(tpr2, fpr2)
        tpr3, fpr3 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:15],gt=gt[:15])
        print(tpr3, fpr3)
        tpr4, fpr4 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:20],gt=gt[:20])
        print(tpr4, fpr4)
        tpr5, fpr5 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:25],gt=gt[:25])
        print(tpr5, fpr5)
        tpr6, fpr6 = self.cal_fpr_tpr(thresholds= thresholds,pred = pred[:30],gt=gt[:30])
        print(tpr6, fpr6)
        #roc_auc1 = auc(fpr1, tpr1)
        #roc_auc2 = auc(fpr2, tpr2)
        #roc_auc3 = auc(fpr3, tpr3)
        #roc_auc4 = auc(fpr4, tpr4)
        #roc_auc5 = auc(fpr5, tpr5)
        #roc_auc6 = auc(fpr6, tpr6)
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        # #plt.plot(fpr1, tpr1, label='Nowcasting=30 mins(area = {0:.4f})'.format(roc_auc1))
        # #plt.plot(fpr2, tpr2, label='Nowcasting=60 mins(area = {0:.4f})'.format(roc_auc2))
        # plt.plot(fpr3, tpr3, label='Nowcasting=90 mins(area = {0:.4f})'.format(roc_auc3))
        # plt.plot(fpr4, tpr4, label='Nowcasting=120 mins(area = {0:.4f})'.format(roc_auc4))
        # plt.plot(fpr5, tpr5, label='Nowcasting=150 mins(area = {0:.2f})'.format(roc_auc5))
        # plt.plot(fpr6, tpr6, label='Nowcasting=180 mins(area = {0:.4f})'.format(roc_auc6))
        plt.plot(fpr1,tpr1)
        plt.plot(fpr2,tpr2)
        plt.plot(fpr3,tpr3)
        plt.plot(fpr4,tpr4)
        plt.plot(fpr5,tpr5)
        plt.plot(fpr6,tpr6)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC curve of different threshold')
        plt.legend(loc="lower right")

        # plt.tight_layout()
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        plt.close()

    def roc_auc(self, gt, pred, mask, save_path):
        print("roc_mask info:", np.sum(mask==0))
        print("save_roc, thresholds = :",self._thresholds)
        thresholds = dbz_to_pixel(np.array(self._thresholds, dtype=np.float32)
                                  .reshape((1, 1, len(self._thresholds), 1, 1)))

        gt = (gt > thresholds)
        print("gt info:",gt.shape,np.sum(gt==0))
        print("pred info:",pred.shape,np.sum(pred==0))
        print(pred[:,:,0,:,:].ravel().shape)
        fpr1, tpr1, thershold1 = roc_curve(gt[:,:,0,:,:].ravel(), score[0].ravel(),pos_label=1, drop_intermediate=False)
        fpr2, tpr2, thershold2 = roc_curve(gt[:,:,1,:,:].ravel(), pred[:,:,1,:,:].ravel(),drop_intermediate=False)
        fpr3, tpr3, thershold3 = roc_curve(gt[:,:,2,:,:].ravel(), pred[:,:,2,:,:].ravel(),drop_intermediate=False)
        fpr4, tpr4, thershold4 = roc_curve(gt[:,:,3,:,:].ravel(), pred[:,:,3,:,:].ravel(),drop_intermediate=False)
        fpr5, tpr5, thershold5 = roc_curve(gt[:,:,4,:,:].ravel(), pred[:,:,4,:,:].ravel(),drop_intermediate=False)
        fpr6, tpr6, thershold6 = roc_curve(gt[:,:,5,:,:].ravel(), pred[:,:,5,:,:].ravel(),drop_intermediate=False)

        for i, value in enumerate(thershold1):
            print("%f %f %f" % (fpr1[i], tpr1[i], value))

        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.plot(fpr1, tpr1,  label='threshold=10 (area = {0:.2f})'.format(roc_auc1))
        plt.plot(fpr2, tpr2,  label='threshold=20 (area = {0:.2f})'.format(roc_auc2))
        plt.plot(fpr3, tpr3,  label='threshold=30 (area = {0:.2f})'.format(roc_auc3))
        plt.plot(fpr4, tpr4,  label='threshold=35 (area = {0:.2f})'.format(roc_auc4))
        plt.plot(fpr5, tpr5,  label='threshold=40 (area = {0:.2f})'.format(roc_auc5))
        plt.plot(fpr6, tpr6,  label='threshold=45 (area = {0:.2f})'.format(roc_auc6))

        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC curve of different threshold')
        plt.legend(loc="lower right")

        #plt.tight_layout()
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        plt.close()

    def update(self, gt, pred, mask, start_datetimes=None):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account
        start_datetimes : list
            The starting datetimes of all the testing instances

        Returns
        -------

        """
        if start_datetimes is not None:
            batch_size = len(start_datetimes)
            assert gt.shape[1] == batch_size
        else:
            batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape, "gt.shape{}, pred.shape{}".format(gt.shape, pred.shape)
        assert gt.shape == mask.shape

        if self._use_central:
            # Crop the central regions for evaluation
            pred = pred[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
            gt = gt[:, :, :,
                    self._central_region[1]:self._central_region[3],
                    self._central_region[0]:self._central_region[2]]
            mask = mask[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
        #print(pred)  #(12,1,1,300,300)   in [0,1]
        #print(gt)    #(12,1,1,300,300)   in [0,1]
        #print(mask)  #(12,1,1,300,300)   all [true]
        self._total_batch_num += batch_size
        #TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        mse = (mask * np.square(pred - gt)).sum(axis=(2, 3, 4))
        mae = (mask * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        #bias = ((pred - gt) / gt).sum(axis=(2,3,4))
        weights = get_balancing_weights_numba(data=gt, mask=mask,
                                              base_balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS,
                                              thresholds = None)
        #weights = get_balancing_weights(data=gt, mask=mask,
        #                                      base_balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS,
        #                                      thresholds = None)
        #print(gt[0])
        #print("balanced_weights:", weights)
        balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
        balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        gdl = get_GDL_numba(prediction=pred, truth=gt, mask=mask)
        self._mse += mse.sum(axis=1)
        self._mae += mae.sum(axis=1)
        #self._bias += bias.sum(axis=1)
        self._balanced_mse += balanced_mse.sum(axis=1)
        self._balanced_mae += balanced_mae.sum(axis=1)
        self._gdl += gdl.sum(axis=1)
        if not self._no_ssim:
            raise NotImplementedError
            # self._ssim += get_SSIM(prediction=pred, truth=gt)

        ### here may be wrong
        '''
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts_numba(prediction=pred, truth=gt, mask=mask,
                                     thresholds=self._thresholds)
        '''
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts(prediction=pred, truth=gt, mask=mask,
                                     thresholds=self._thresholds)
        #print(hits)
        
        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------

        """
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        pod = a/(a + c)
        far = b/(a + b)
        csi = a/(a + b + c)
        #bias = (a + b) / (a + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        #bias = self._bias
        balanced_mse = self._balanced_mse / self._total_batch_num
        balanced_mae = self._balanced_mae / self._total_batch_num
        gdl = self._gdl / self._total_batch_num
        #if not self._no_ssim:
        #    raise NotImplementedError
            # ssim = self._ssim / self._total_batch_num
        # return pod, far, csi, hss, gss, mse, mae, gdl
        return pod, far, csi, mse, mae, balanced_mse, balanced_mae, gdl,hss

    def print_stat_readable(self, prefix=""):
        logging.info("%sTotal Sequence Number: %d, Use Central: %d"
                     %(prefix, self._total_batch_num, self._use_central))
        pod, far, csi, mse, mae, balanced_mse, balanced_mae, gdl,hss= self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        logging.info("   Hits: " + ', '.join([">%g:%g/%g" % (threshold,
                                                             self._total_hits[:, i].mean(),
                                                             self._total_hits[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   POD: " + ', '.join([">%g:%g/%g" % (threshold, pod[:, i].mean(), pod[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        '''
        logging.info("   Bias: " + ', '.join([">%g:%g/%g" % (threshold, bias[:, i].mean(), bias[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   GSS: " + ', '.join([">%g:%g/%g" % (threshold, gss[:, i].mean(), gss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        '''
        logging.info("   HSS: " + ', '.join([">%g:%g/%g" % (threshold, hss[:, i].mean(), hss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))

        logging.info("   MSE: %g/%g" % (mse.mean(), mse[-1]))
        logging.info("   MAE: %g/%g" % (mae.mean(), mae[-1]))
        logging.info("   Balanced MSE: %g/%g" % (balanced_mse.mean(), balanced_mse[-1]))
        logging.info("   Balanced MAE: %g/%g" % (balanced_mae.mean(), balanced_mae[-1]))
        logging.info("   GDL: %g/%g" % (gdl.mean(), gdl[-1]))
        #logging.info("   HSS: %g/%g" % (hss.mean(), hss[-1]))
        if not self._no_ssim:
            raise NotImplementedError

    def save_pkl(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info("Saving Evaluation to %s" %path)
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_txt_readable(self, path):     ## be used!

        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pod, far, csi, mse, mae, balanced_mse, balanced_mae, gdl,hss = self.calculate_stat()
        logging.info("Saving readable txt of Evaluation to %s" % path)
        f = open(path, 'w')
        f.write("Total Sequence Num: {}, Out Seq Len: {} \n".format(self._total_batch_num, self._seq_len))
        
        mid_idx = (self._seq_len - 1) // 2 
        for (i, threshold) in enumerate(self._thresholds):

            f.write("Threshold = %g:\n" %threshold)
            f.write("   POD: %s\n" % str(list(pod[:, i])))
            f.write("   FAR: %s\n" % str(list(far[:, i])))
            f.write("   CSI: %s\n" % str(list(csi[:, i])))
            #f.write("   Bias: %s\n" % str(list(bias[:, i])))
            #f.write("   GSS: %s\n" % str(list(gss[:, i])))
            f.write("   HSS: %s\n" % str(list(hss[:, i])))

            if self._seq_len == 20:
                f.write("   POD stat: avg %.3f\t30min %.3f\t60min %.3f\t90min %.3f\t120min %.3f\n" \
                       %(pod[:, i].mean(), pod[4, i], pod[9, i], pod[14, i], pod[-1, i]) )
                f.write("   FAR stat: avg %.3f\t30min %.3f\t60min %.3f\t90min %.3f\t120min %.3f\n" \
                       %(far[:, i].mean(), far[4, i], far[9, i], far[14, i], far[-1, i]) )
                f.write("   CSI stat: avg %.3f\t30min %.3f\t60min %.3f\t90min %.3f\t120min %.3f\n" \
                       %(csi[:, i].mean(), csi[4, i], csi[9, i], csi[14, i], csi[-1, i]) )
            else:
                f.write("   POD stat: avg %g\tmid %g\tfinal %g\n" %(pod[:, i].mean(), pod[mid_idx, i], pod[-1, i]))
                f.write("   FAR stat: avg %g\tmid %g\tfinal %g\n" %(far[:, i].mean(), far[mid_idx, i], far[-1, i]))
                f.write("   CSI stat: avg %g\tmid %g\tfinal %g\n" %(csi[:, i].mean(), csi[mid_idx, i], csi[-1, i]))
            #f.write("   Bias stat: avg %g/mid %g/final %g\n" %(bias[:, i].mean(),bias[mid_idx, i],bias[-1, i]))
            #f.write("   GSS stat: avg %g/mid %g/final %g\n" %(gss[:, i].mean(),gss[mid_idx, i],gss[-1, i]))
                f.write("   HSS stat: avg %g/mid %g/final %g\n" %(hss[:, i].mean(),hss[mid_idx, i],hss[-1, i]))

        f.write("MSE: %s\n" % str(list(mse)))
        f.write("MAE: %s\n" % str(list(mae)))
        f.write("Balanced MSE: %s\n" % str(list(balanced_mse)))
        f.write("Balanced MAE: %s\n" % str(list(balanced_mae)))
        f.write("GDL: %s\n" % str(list(gdl)))
        #f.write("HSS: %s\n" % str(list(hss)))
        f.write("MSE stat: avg %g/final %g\n" % (mse.mean(), mse[-1]))
        f.write("MAE stat: avg %g/final %g\n" % (mae.mean(), mae[-1]))
        f.write("Balanced MSE stat: avg %g/final %g\n" % (balanced_mse.mean(), balanced_mse[-1]))
        f.write("Balanced MAE stat: avg %g/final %g\n" % (balanced_mae.mean(), balanced_mae[-1]))
        f.write("GDL stat: avg %g/final %g\n" % (gdl.mean(), gdl[-1]))
        #f.write("hss stat: avg %g/final %g\n" % (hss.mean(), hss[-1]))
        f.close()

        return mse.mean(), mae.mean(), csi.mean(axis=0)

    def save(self, prefix):
        #self.save_pkl(prefix + ".pkl")
        return self.save_txt_readable(prefix + ".txt")

if __name__ == '__main__':
    a = HMWEvaluation()
