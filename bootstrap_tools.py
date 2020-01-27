#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:51:37 2016

@author: pipolose
"""

from __future__ import division
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score  # , auc
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


def provide_rand_idx(out_len, runs_per_subj, replacement=True):
    '''
    SUBJECT BOOTSTRAP RESAMPLING MAIN FUNCTION
    Inputs:
        out_len: max index length,
        runs_per_subj: how many runs for each subject
        replacement: wether to allow sampling w replacement or not
    Output:
        Vector of runs, having resampled over subjects
    '''
    assert np.remainder(out_len, runs_per_subj) == 0
    n_subjects = int(out_len / runs_per_subj)
    if replacement:
        subj_idx_used = np.random.randint(n_subjects, size=n_subjects)
    else:
        subj_idx_used = np.random.permutation(n_subjects)
    ix_used = np.empty((out_len,), dtype=int)
    for k in range(runs_per_subj):
        ix_used[k::runs_per_subj] = subj_idx_used * runs_per_subj + k
    return ix_used


def provide_performance_distributions_from_df(results_df, n_resamples,
                                              runs_per_subj=0,
                                              shuffle=False,
                                              replacement=True):
    '''
    INPUT: results_df: A dataframe with classifier predictions,
           runs_per_subj: number of resamples to use (optional if results_df
                          has a col named 'subjid')
           shuffle: If a null distribution is being computed
           replacement: If sampling with replacement is allowed. If shuffle is
           False, replacement is automatically set to True.
    OUTPUT: dictionary with distributions of correct classify, TP, TN, FP, FN
    '''

    if not shuffle and not replacement:
        replacement = True
        warnings.warn('Replacement was automatically set to True because'
                      'shuffle is False')

    if runs_per_subj == 0:
        non_subj_cols = [col for col in results_df.columns if col != 'subjid']
        run_counts = results_df.groupby('subjid')[non_subj_cols[-1]]\
            .count().unique()
        runs_per_subj = run_counts[0]
        assert run_counts.shape == (1,)
    assert np.remainder(results_df.shape[0], runs_per_subj) == 0
    n_samples = results_df.shape[0]
#    perf = np.empty(n_resamples)
#    TP_rate = np.empty(n_resamples)
#    TN_rate = np.empty(n_resamples)
#    FP_rate = np.empty(n_resamples)
#    FN_rate = np.empty(n_resamples)
    TP = np.empty(n_resamples)
    TN = np.empty(n_resamples)
    FP = np.empty(n_resamples)
    FN = np.empty(n_resamples)
    for t in range(n_resamples):
        ix_used = provide_rand_idx(n_samples,
                                   runs_per_subj,
                                   replacement=replacement)
        if shuffle:
            TP[t] = ((results_df.iloc[ix_used]['labels'] == 1).values &
                  (results_df['prediction'] == 1).values).sum()
            TN[t] = ((results_df.iloc[ix_used]['labels'] == -1).values &
                  (results_df['prediction'] == -1).values).sum()
            FP[t] = ((results_df.iloc[ix_used]['labels'] == -1).values &
                  (results_df['prediction'] == 1).values).sum()
            FN[t] = ((results_df.iloc[ix_used]['labels'] == 1) &
                  (results_df['prediction'] == -1).values).sum()

        else:
            TP[t] = ((results_df.iloc[ix_used]['labels'] == 1) &
                  (results_df.iloc[ix_used]['prediction'] == 1)).sum()
            TN[t] = ((results_df.iloc[ix_used]['labels'] == -1) &
                  (results_df.iloc[ix_used]['prediction'] == -1)).sum()
            FP[t] = ((results_df.iloc[ix_used]['labels'] == -1) &
                  (results_df.iloc[ix_used]['prediction'] == 1)).sum()
            FN[t] = ((results_df.iloc[ix_used]['labels'] == 1) &
                  (results_df.iloc[ix_used]['prediction'] == -1)).sum()
    acc = (TN + TP) / (TN + TP + FN + FP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * TP / (2 * TP + FN + FP)
#        perf[t] = (TP + TN) / n_samples
#        TP_rate[t] = TP / (TP + FN)
#        FP_rate[t] = FP / (TN + FP)
#        TN_rate[t] = TN / (TN + FP)
#        FN_rate[t] = FN / (TN + FN)

#    from IPython.terminal.debugger import TerminalPdb; TerminalPdb().set_trace()
    out_dict = {'Accuracy': acc,
                'Sensitivity': TPR,
                'Specificity': TNR,
                'NPV': NPV,
                'PPV': PPV,
                'F1 Score': F1}
    return out_dict


def plot_bootstrap_distributions(perf_dist_dict, chance_dict={},
                                 perf_alpha=5, chance_alpha=None,
                                 bar_width=1, x_step=1.75, fig_title=None,
                                 plot_aspect=6):

    '''
    MAKES BAR PLOT OF PERFORMANCE AND CONFUSION MATRIX FROM DICTIONARY WITH
    EMPIRICAL DISTIRBUTIONS
    Inputs:
       perf_dist_dict: dict of empirical distributions, as made by
                       provide_performance_distributions_from_df
       chance_dict (optional): idem, but for random chance distributions
       perf_alpha: alpha value (in percent) for performane confidence interval
       chance_alpha: idem, but for chance level. If None, it is set like
                     perf_alpha
       bar_width (=1): width of bars
       x_step (=1.75): inter-bar separation (from left edge to left edge)
       fig_title (=None): Figure title, optional
       plot_aspect (=6): Plot aspect ratio
    Outputs:
        fig: Figure object
        ax: Axis object
    '''
    if chance_alpha is None:
        chance_alpha = perf_alpha

    perf_mean = {k: np.mean(v) for k, v in perf_dist_dict.items()}

    if perf_alpha:
        perf_conf = {k: np.array([np.percentile(v, perf_alpha/2),
                                  np.percentile(v, (100-perf_alpha/2))])
                     for k, v in perf_dist_dict.items()}

    if chance_dict:
        rand_mean = {k: np.mean(v) for k, v in chance_dict.items()}
        if chance_alpha:
            rand_int = {k: np.array([np.percentile(v, chance_alpha/2),
                                     np.percentile(v, (100-chance_alpha/2))])
                        for k, v in chance_dict.items()}

    col_names = ['Accuracy', 'Sensitivity', 'Specificity',
                 'PPV', 'NPV']
    #['Fraction_correct', 'TP', 'TN', 'FP', 'FN']

    # Check the col names are all in the dict keys:
    assert np.all([cn in perf_dist_dict.keys() for cn in col_names])

    # Make plot
    sns.set_style("white", {'ytick.major.size': 3})
    #sns.set_style('white')
    fig, ax = plt.subplots()
    sns.despine(offset=8, ax=ax)
    mpl.rcParams.update({'font.size': 13.0})

    # Height of bars:
    y = np.array([perf_mean[cn] for cn in col_names])
    # Left x-coord or bars:
    left = np.arange(y.shape[0]) * x_step #- bar_width / 2.

    bh = ax.bar(left, y, bar_width, color=sns.color_palette("Set2",
                                                            len(col_names)))


    ax.set_ylim([0, 1])
    ax.set_ylabel('Performance', {'fontsize': 16.0})
    #from IPython.terminal.debugger import TerminalPdb; TerminalPdb().set_trace()
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'],
                       {'fontsize': 13})
    if fig_title:
        if fig_title == 'accuracy':
            accu = perf_mean['Accuracy'].mean()
            fig_title = "Accuracy: {:.1f}%".format(100 * accu)
        elif fig_title == 'full':
            accu = perf_mean['Accuracy'].mean()
            p_val = (chance_dict['Accuracy'] >
                     perf_mean['Accuracy']).sum() /\
                     float(chance_dict['Accuracy'].shape[0])
            fig_title = "Accuracy: {:.1f}%, p-val={}".format(100 * accu,
                                                             p_val)
            print(p_val)
        ax.set_title(fig_title, {'fontsize': 18})
    ax.set_xticks(left)# + bar_width/2)
    ax.set_xticklabels([cn.replace('_', ' ') for cn in col_names],
                       {'fontsize': 13})

    #  Line handles and text for legend
    lg_h_in = []
    lg_h_text = []

    if perf_alpha:
        # Draw error lines
        err_x = left #+ bar_width/2
        err_ymin = np.array([perf_conf[cn][0] for cn in col_names])
        err_ymax = np.array([perf_conf[cn][1] for cn in col_names])
        err_lh = plt.vlines(err_x, err_ymin, err_ymax)
        err_lh.set_linewidths(2)
        lg_h_in.append(err_lh)
        lg_h_text.append('{}% CI'.format(100-perf_alpha))

    if chance_dict:
        # Draw chance performance
        chance_y = np.array([rand_mean[cn] for cn in col_names])
        chance_xmin = left - .1 * bar_width - bar_width / 2.
        chance_xmax = left + 1.1 * bar_width - bar_width / 2.
        chance_mlh = plt.hlines(chance_y, chance_xmin, chance_xmax)
        chance_mlh.set_linewidths(.5)
        lg_h_in.append(chance_mlh)
        lg_h_text.append('Chance')
        if chance_alpha:
            for line_idx, line_name in enumerate(['min', 'max']):
                chance_y = np.array([rand_int[cn][line_idx]
                                     for cn in col_names])
                chance_xmin = left - .1 * bar_width - bar_width / 2.
                chance_xmax = left + 1.1 * bar_width- bar_width / 2.
                chance_lh = plt.hlines(chance_y, chance_xmin, chance_xmax,
                                       linestyles='dotted')
                chance_lh.set_linewidths(.5)
            lg_h_in.append(chance_lh)
            lg_h_text.append('Chance\n{}% CI'.format(100-chance_alpha))
    ax.set_aspect(plot_aspect)
    if lg_h_in:
        ax.legend(lg_h_in, lg_h_text)


    return fig, ax


def transition_matrix_from_2D_array(init_final):
    '''
    Expects as input a 2-D array with subjets on each row,
    and one sample on each column labeled as
    'TN', 'FP', 'FN', or 'TP'
    Returns a transition probability matrix
    '''

    key = np.array(['TN', 'FP', 'FN', 'TP'])
    T = np.zeros((4, 4))
    for i, ki in enumerate(key):
        for j, kj in enumerate(key):
            in_subjs = (init_final[:, 0] == ki) & (init_final[:, 1] == kj)
            T[i, j] = in_subjs.sum()
    N = np.reshape(T.sum(axis=1), (4, 1))  # Normalized transition matrix
    T /= np.tile(N, (1, 4))
    T[np.isnan(T)] = 0
    return T


def make_transition_dist(init_final, n_iter, shuffle=False, replacement=True):
    '''
    Takes a vector just like in transition_matrix_from_2D_array,
    and provides a distribution of transition matrices either
    by resampling or randomization
    '''
    T_dist = np.zeros((4, 4, n_iter))
    subsets = [np.array(['TN', 'FP']), np.array(['FN', 'TP'])]
    if_ss = np.zeros((2,)).tolist()
    idx_ss = np.zeros((2,)).tolist()
    lidx_ss = np.zeros((2,)).tolist()
    for si, ss in enumerate(subsets):
        lidx_ss[si] = np.in1d(init_final[:, 0], ss)
        idx_ss[si] = np.argwhere(lidx_ss[si]).flatten()
        if_ss[si] = init_final[idx_ss[si], :]
    for it in range(n_iter):
        A = np.zeros_like(init_final)
        for si, ss in enumerate(subsets):
            new_idx = provide_rand_idx(if_ss[si].shape[0],
                                       runs_per_subj=1,
                                       replacement=replacement).astype(int)
            new_new_idx = idx_ss[si][new_idx]
            A[idx_ss[si], :] = init_final[new_new_idx, :].copy()
            if shuffle:
                A[idx_ss[si], 0] = if_ss[si][:, 0]
        T_dist[:, :, it] = transition_matrix_from_2D_array(A)
    return T_dist


def provide_auc_distributions_from_df(prediction_df, n_resamples,
                                      runs_per_subj=0,
                                      shuffle=False,
                                      replacement=True):
    '''
    Provides a distribution of auc values, can be random or boostrap resamples
    '''
    prediction_df['continuous_prediction'] = pd.to_numeric(prediction_df['continuous_prediction'],
                                                           errors='coerce')

    if not shuffle and not replacement:
        replacement = True
        warnings.warn('Replacement was automatically set to True because'
                      'shuffle is False')

    if runs_per_subj == 0:
        non_subj_cols = [col for col in prediction_df.columns
                         if col != 'subjid']
        run_counts = prediction_df.groupby('subjid')[non_subj_cols[-1]]\
            .count().unique()
        runs_per_subj = run_counts[0]
        assert run_counts.shape == (1,)
    assert np.remainder(prediction_df.shape[0], runs_per_subj) == 0
    n_samples = prediction_df.shape[0]

    dist_aucs = np.nan * np.zeros((n_resamples,))
    for resamp_idx in range(n_resamples):
        dist_idx = provide_rand_idx(n_samples,
                                    runs_per_subj,
                                    replacement=replacement)
        dist_labels = prediction_df['labels'].iloc[dist_idx].values
        aucs = []
        for fold_idx in prediction_df['test_fold'].unique():
            if shuffle:
                sample_idx = (prediction_df['test_fold'] == fold_idx).values
                used_pred = prediction_df['continuous_prediction'][sample_idx]
            else:
                sample_idx = (prediction_df.iloc[dist_idx]['test_fold'] == fold_idx).values
                used_pred = prediction_df.iloc[dist_idx]['continuous_prediction'][sample_idx]
            if np.all(used_pred.isna()):
                fold_auc = .5
            else:
                if np.unique(dist_labels[sample_idx]).shape[0] >1:
                    fold_auc = roc_auc_score(dist_labels[sample_idx], used_pred)
                else:
                    fold_auc = np.nan
            aucs.append(fold_auc)
        dist_aucs[resamp_idx] = np.nanmean(aucs)
    return dist_aucs

def provide_cont_correlation_distributions_from_df(prediction_df, n_resamples,
                                                   runs_per_subj=0,
                                                   shuffle=False,
                                                   replacement=True,
                                                   provide_fisher_transformed=True,
                                                   corr_type='pearson'):
    '''
    Provides a distribution of correlation values for continuous prediction
    and continuous
    , can be random or boostrap resamples

    INPUTS
    provide_fisher_transformed: Bool, default True
        If true the distribution is fisher transformed
    '''
    prediction_df['continuous_prediction'] = pd.to_numeric(prediction_df['continuous_prediction'],
                                                           errors='coerce')

    if not shuffle and not replacement:
        replacement = True
        warnings.warn('Replacement was automatically set to True because'
                      'shuffle is False')
    '''
    Estimate runs_per_subj if not given
    '''
    if runs_per_subj == 0:
        non_subj_cols = [col for col in prediction_df.columns
                         if col != 'subjid']
        run_counts = prediction_df.groupby('subjid')[non_subj_cols[-1]]\
            .count().unique()
        runs_per_subj = run_counts[0]
        assert run_counts.shape == (1,)
    assert np.remainder(prediction_df.shape[0], runs_per_subj) == 0

    '''
    List of indices for each fold
    '''
    row_idx_per_fold = []
    n_samples_per_fold = []
    for fold_idx in prediction_df['test_fold'].unique():
        these_rows = np.argwhere(prediction_df['test_fold'] == fold_idx).flatten()
        row_idx_per_fold.append(these_rows)
        n_samples_per_fold.append(these_rows.shape[0])

    '''
    Compute distribution of correlations
    '''
    dist_corrs = np.nan * np.zeros((n_resamples,))
    # resampled_rows = np.empty((prediction_df.shape[0],), dtype=int)
    dist_idx_per_fold = np.empty((prediction_df.shape[0],), dtype=int)

    # preallocated df to be refilled on each resamle
    resamp_df = prediction_df[['continuous_prediction', 'continuous_label',
                               'test_fold']].copy()
    do_the_analysis = ~np.all(np.isnan(prediction_df['continuous_prediction']))
    if do_the_analysis:
        idx = pd.IndexSlice
        for resamp_idx in range(n_resamples):
            for fold_idx in prediction_df['test_fold'].unique():
                dist_idx_per_fold[row_idx_per_fold[fold_idx]] =\
                    row_idx_per_fold[fold_idx][provide_rand_idx(n_samples_per_fold[fold_idx],
                                                                runs_per_subj,
                                                                replacement=replacement)]
            resamp_df['continuous_prediction'] =\
                prediction_df.iloc[dist_idx_per_fold]['continuous_prediction'].values
            if not shuffle:
                resamp_df['continuous_label'] = prediction_df.iloc[dist_idx_per_fold]['continuous_label'].values
            fish_corrs = np.arctanh(resamp_df.groupby('test_fold')
                                    ['continuous_prediction',
                                     'continuous_label'].corr(method=corr_type).
                                    loc[idx[:, 'continuous_label'],
                                        'continuous_prediction'].values)
            dist_corrs[resamp_idx] = fish_corrs.mean()
    if not provide_fisher_transformed:
        dist_corrs = np.tanh(dist_corrs)
    return dist_corrs


def plot_ROC_curve_from_df(prediction_df, runs_per_subj=0, n_resamples=200, alpha=5,
                           rand_dist=np.nan):
    '''
    Takes a dataframe with continuous predictions, labels and test fold indices
    and provides a plot of the ROC, with confidence interval
    '''
    n_xpoints = 100
    dist_tprs = np.nan * np.zeros((n_resamples, n_xpoints))
    med_fpr = np.linspace(0, 1, n_xpoints)
    dist_aucs = np.nan * np.zeros((n_resamples, ))
    n_samples = prediction_df.shape[0]

    if runs_per_subj == 0:
        non_subj_cols = [col for col in prediction_df.columns
                         if col != 'subjid']
        run_counts = prediction_df.groupby('subjid')[non_subj_cols[-1]]\
            .count().unique()
        runs_per_subj = run_counts[0]
        assert run_counts.shape == (1,)
    assert np.remainder(prediction_df.shape[0], runs_per_subj) == 0

    for res_idx in range(n_resamples):
        dist_idx = provide_rand_idx(n_samples,
                                    runs_per_subj,
                                    replacement=True)
        dist_df = prediction_df.iloc[dist_idx]
        dist_labels = dist_df['labels'].values
        tprs = []
        aucs = []
        for fold_idx in dist_df['test_fold'].unique():
            sample_idx = (dist_df['test_fold'] == fold_idx).values
            used_pred = dist_df['continuous_prediction'][sample_idx]
            if np.all(used_pred.isna()):
                fold_auc = .5
                fpr = med_fpr.copy()
                tpr = .5 * np.ones(fpr.shape)
            else:
                if np.unique(dist_labels[sample_idx]).shape[0] >1:
                    fold_auc = roc_auc_score(dist_labels[sample_idx], used_pred)
                    fpr, tpr, thresholds = roc_curve(dist_labels[sample_idx],
                                                     used_pred)
                else:
                    fold_auc = np.nan
                    fpr = med_fpr.copy()
                    tpr = np.nan * np.ones(fpr.shape)
            aucs.append(fold_auc)  # auc(fpr, tpr))
            tprs.append(np.interp(med_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        dist_tprs[res_idx, :] = np.nanmean(tprs, axis=0)
        dist_aucs[res_idx] = np.nanmean(aucs)

    med_auc = np.median(dist_aucs)
    med_tprs = np.median(dist_tprs, axis=0)
    max_tprs = np.percentile(dist_tprs, 100-alpha/2., axis=0)
    min_tprs = np.percentile(dist_tprs, alpha/2., axis=0)

    fh, ah = plt.subplots()

    ah.plot(med_fpr, med_tprs, color='black', label='Median')
    ah.fill_between(med_fpr, min_tprs, max_tprs, color='grey', alpha=.2,
                    label=r'{:2.0f}% CI'.format(100-alpha))

    ah.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
            label='chance curve', alpha=.8)

    ah.set_xlim([-0.05, 1.05])
    ah.set_ylim([-0.05, 1.05])
    ah.set_xlabel('False Positive Rate')
    ah.set_ylabel('True Positive Rate')
    if rand_dist is np.nan:
        title = 'ROC curve, median AUC: {:0.2f}'.format(med_auc)
    else:
        p_val = (rand_dist > med_auc).sum() / float(rand_dist.shape[0])
        print(p_val)
        title = 'ROC curve, median AUC: {:0.2f}; p-val: {} '.format(med_auc,
                                                                    p_val)
    ah.set_title(title)
    plt.legend(loc="lower right")
    return fh, ah

def plot_corr_per_fold_from_df(prediction_df, runs_per_subj=0, n_resamples=200,
                               alpha=5, rand_dist=np.nan, expected_sign=-1,
                               compress_axes=0.05, corr_type='pearson'):
    '''
    Takes a dataframe with continuous predictions, continuous labels
    and test fold indices and provides a plot PER FOLD of the
    relationship between prediction and labels
    '''

    prediction_df['continuous_label'] *= expected_sign

    corr_dist = provide_cont_correlation_distributions_from_df(prediction_df,
                                                               n_resamples=n_resamples,
                                                               runs_per_subj=runs_per_subj,
                                                               shuffle=False,
                                                               replacement=True,
                                                               provide_fisher_transformed=True,
                                                               corr_type=corr_type)

    mean_corr = np.tanh(np.nanmean(corr_dist))
    max_tprs = np.tanh(np.percentile(corr_dist, 100-alpha/2., axis=0))
    min_tprs = np.tanh(np.percentile(corr_dist, alpha/2., axis=0))


    with sns.axes_style("darkgrid"):
        facet = sns.lmplot('continuous_prediction', 'continuous_label',
                           prediction_df, col='test_fold', col_wrap=2)
    for ix, ax in enumerate(facet.axes):
        ax.set_title("Test fold {}".format(ix + 1))
        xl = ax.get_xlabel()
        if xl:
            ax.set_xlabel("Hyperplane distance (a.u.)")
        yl = ax.get_ylabel()
        if yl:
            ax.set_ylabel("Continous label")
        if compress_axes > 0:
            y_min = prediction_df['continuous_label'].quantile(q=compress_axes/2)
            y_max = prediction_df['continuous_label'].quantile(q=1-compress_axes/2)
            ax.set_ylim([y_min, y_max])
    if rand_dist is np.nan:
        title = 'Corr: {:0.2f} CI: [{:0.2f}, {:0.2f}]'.\
            format(mean_corr, min_tprs, max_tprs)
    else:
        p_val = (rand_dist > mean_corr).sum() / float(rand_dist.shape[0])
        title = 'Corr: {:0.2f} CI: [{:0.2f}, {:0.2f}] p-val: {}'.\
            format(mean_corr, min_tprs, max_tprs, p_val)
    fh = facet.fig
    fh.suptitle(title)
    return fh, facet.axes
