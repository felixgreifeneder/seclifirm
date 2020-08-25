import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
import sklearn.metrics as skmetrics
from sklearn import linear_model
from scipy.stats import pearsonr

def monthly_model(area='west'):
    if area == 'west':
        csvname = 'occidental_alps.csv'
    elif area == 'east':
        csvname = 'oriental_alps.csv'
    elif area == 'apennines':
        csvname = 'apennines.csv'
    east_alps = pd.read_csv("//projectdata.eurac.edu/projects/seclifirm/ENEL case study/" + csvname,
                            header=0,
                            index_col=0,
                            parse_dates=True,
                            infer_datetime_format=True)

    # No Snow:
    # Train a model for the estimation of runoff for each month

    # OLS
    surf_model_list = list()
    sub_model_list = list()
    for i in range(12):
        #ols
        # surf_model_list.append(linear_model.LinearRegression())
        # sub_model_list.append(linear_model.LinearRegression())
        #ridge
        surf_model_list.append(linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)))
        sub_model_list.append(linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)))
        #lasso
        #surf_model_list.append(linear_model.LassoCV(cv=20))
        #sub_model_list.append(linear_model.LassoCV(cv=20))
        # elastic net
        #surf_model_list.append(linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7))
        #sub_model_list.append(linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7))
        # lasso lars
        #surf_model_list.append(linear_model.LassoLars(alpha=.1))
        #sub_model_list.append(linear_model.LassoLars(alpha=.1))
        # feature selection
        # surf_model_tmp = SelectFromModel(estimator=linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)), threshold=0.25)
        # sub_model_tmp = SelectFromModel(estimator=linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)), threshold=0.25)
        # surf_model_list.append(surf_model_tmp)
        # sub_model_list.append(sub_model_tmp)

    # seting up the prediction time series
    surf_runoff_pred = pd.Series(index=pd.date_range(dt.date(day=1, month=1, year=2013), dt.date(day=1, month=12, year=2017), freq='MS'))
    surf_runoff_true = pd.Series(
        index=pd.date_range(dt.date(day=1, month=1, year=2013), dt.date(day=1, month=12, year=2017), freq='MS'))
    sub_runoff_pred = pd.Series(
        index=pd.date_range(dt.date(day=1, month=1, year=2013), dt.date(day=1, month=12, year=2017), freq='MS'))
    sub_runoff_true = pd.Series(
        index=pd.date_range(dt.date(day=1, month=1, year=2013), dt.date(day=1, month=12, year=2017), freq='MS'))

    # setting out the report output file
    reportfile = open("//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/" + area + "/report.txt", 'w+')
    monthlabels = ['January', 'Februray', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
    reportfile.write('Summary of monthly linear fit between temperature, precipitation, and runoff\n\n')

    for imonth in range(12):
        reportfile.write(monthlabels[imonth] + ': \n')
        dateindx = np.array([dt.date(year=iyear, month=imonth+1, day=1) for iyear in range(1980,2018)])
        surf_runoff = np.array(east_alps['Surface Runoff [m]'][dateindx])
        sub_runoff = np.array(east_alps['Subsurface Runoff [m]'][dateindx])

        # the 12 months time-series of each feature is considered for the prediction. In the first case the variables
        # considered are total precipitation, temperature
        feature_matrix = np.full((len(surf_runoff), 24), 0.0)

        # fill the feature matrix
        row_count = 0
        for i in dateindx:
            col_count = 0
            for j in range(12):
                feature_matrix[row_count, col_count] = east_alps['Total Precipitation [m]'][subtract_n_month(i,j)]
                col_count += 1
            for j in range(12):
                feature_matrix[row_count, col_count] = east_alps['Temperature [K]'][subtract_n_month(i,j)]
                col_count += 1
            row_count += 1

        # define training and test set
        # we select 1980-2015 as training dataset and 2016, 2017 as a validation period
        surf_runoff_train = surf_runoff[0:-5]
        surf_runoff_test = surf_runoff[-5::]
        sub_runoff_train = sub_runoff[0:-5]
        sub_runoff_test = sub_runoff[-5::]
        feature_matrix_train = feature_matrix[0:-5, :]
        feature_matrix_test = feature_matrix[-5::, :]
        test_index = dateindx[-5::]

        # plot the mutal information
        feature_names = ['p-0', 'p-1', 'p-2', 'p-3', 'p-4', 'p-5', 'p-6', 'p-7',
                         'p-8', 'p-9', 'p-10', 'p-11',
                         't-0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-6', 't-7', 't-8',
                         't-9',
                         't-10', 't-11']

        plt.figure(figsize=(4, 5))
        m_info_surf = mutual_info_regression(feature_matrix_train, surf_runoff_train, random_state=42)
        plt.barh(feature_names, m_info_surf)
        plt.title('Mutual Information')
        plt.tight_layout()
        plt.savefig('//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/' + area + '/no_snow_misurf_' + str(imonth) + '.png', dpi=600)
        plt.close()
        plt.figure(figsize=(4, 5))
        m_info_sub = mutual_info_regression(feature_matrix_train, sub_runoff_train, random_state=42)
        plt.barh(feature_names, m_info_sub)
        plt.title('Mutual Information')
        plt.tight_layout()
        plt.savefig(
            '//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/' + area + '/no_snow_misub_' + str(
                imonth) + '.png', dpi=600)
        plt.close()
        # plot f_scores and p value
        f_scores_surf = f_regression(feature_matrix_train, surf_runoff_train)
        # calculate r2
        r2_scores = [np.square(pearsonr(feature_matrix_train[:,c], surf_runoff_train)[0]) for c in range(feature_matrix_train.shape[1])]
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5, 2), dpi=600)
        ax1.barh(feature_names, f_scores_surf[0])
        ax1.set_title('F-Value', fontdict={'fontsize': 8})
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=5)
        ax2.barh(feature_names, f_scores_surf[1])
        ax2.set_title('p-Value', fontdict={'fontsize': 8})
        ax2.set_xlim(0, 0.1)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=5)
        ax3.barh(feature_names, r2_scores)
        ax3.set_title('R2', fontdict={'fontsize': 8})
        ax3.tick_params(axis='x', labelsize=8)
        ax3.tick_params(axis='y', labelsize=5)
        ax3.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig('//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/' + area + '/no_snow_fsurf_' + str(imonth) + '.png', dpi=600)
        plt.close()
        f_scores_sub = f_regression(feature_matrix_train, sub_runoff_train)
        r2_scores = [np.square(pearsonr(feature_matrix_train[:, c], sub_runoff_train)[0]) for c in
                     range(feature_matrix_train.shape[1])]
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5, 2), dpi=600)
        ax1.barh(feature_names, f_scores_sub[0])
        ax1.set_title('F-Value', fontdict={'fontsize': 8})
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=5)
        ax2.barh(feature_names, f_scores_sub[1])
        ax2.set_title('p-Value', fontdict={'fontsize': 8})
        ax2.set_xlim(0, 0.1)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=5)
        ax3.barh(feature_names, r2_scores)
        ax3.set_title('R2', fontdict={'fontsize': 8})
        ax3.tick_params(axis='x', labelsize=8)
        ax3.tick_params(axis='y', labelsize=5)
        ax3.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(
            '//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/' + area + '/no_snow_fsub_' + str(
                imonth) + '.png', dpi=600)
        plt.close()

        # fit the model
        surf_model_list[imonth].fit(feature_matrix_train[:,f_scores_surf[1] <= 0.05], surf_runoff_train)
        sub_model_list[imonth].fit(feature_matrix_train[:,f_scores_sub[1] <= 0.05], sub_runoff_train)

        surf_tmp_pred = surf_model_list[imonth].predict(feature_matrix_test[:,f_scores_surf[1] <= 0.05])
        sub_tmp_pred = sub_model_list[imonth].predict(feature_matrix_test[:,f_scores_sub[1] <= 0.05])
        pred_count = 0
        for i in test_index:
            surf_runoff_pred[i] = surf_tmp_pred[pred_count]
            sub_runoff_pred[i] = sub_tmp_pred[pred_count]
            surf_runoff_true[i] = surf_runoff_test[pred_count]
            sub_runoff_true[i] = sub_runoff_test[pred_count]
            pred_count = pred_count + 1

        # compute the monthly performance
        reportfile.write('Surface Runoff RMSE: ' + str(np.sqrt(skmetrics.mean_squared_error(surf_runoff_test, surf_tmp_pred))))
        reportfile.write('; R2: ' + str(skmetrics.r2_score(surf_runoff_test, surf_tmp_pred)))
        reportfile.write(
            '\nSubsurface Runoff RMSE: ' + str(np.sqrt(skmetrics.mean_squared_error(sub_runoff_test, sub_tmp_pred))))
        reportfile.write('; R2: ' + str(skmetrics.r2_score(sub_runoff_test, sub_tmp_pred)))
        reportfile.write('\n')
        # output of the linear functions
        reportfile.write('Determined linear relationships: \n')
        sel_features_surf = np.where(f_scores_surf[1] <= 0.05)
        eq_string = 'Surface Runoff = ' + str(surf_model_list[imonth].intercept_)
        coef_counter = 0
        for ii in sel_features_surf[0]:
            eq_string = eq_string + ' + ' + str(surf_model_list[imonth].coef_[coef_counter]) + feature_names[ii]
        reportfile.write(eq_string + '\n')
        sel_features_sub = np.where(f_scores_sub[1] <= 0.05)
        eq_string = 'Subsurface Runoff = ' + str(sub_model_list[imonth].intercept_)
        coef_counter = 0
        for ii in sel_features_sub[0]:
            eq_string = eq_string + ' + ' + str(sub_model_list[imonth].coef_[coef_counter]) + feature_names[ii]
        reportfile.write(eq_string + '\n\n')


    # accuracy assessment
    surf_rmse = np.sqrt(skmetrics.mean_squared_error(surf_runoff_true, surf_runoff_pred))
    sub_rmse = np.sqrt(skmetrics.mean_squared_error(sub_runoff_true, sub_runoff_pred))
    surf_r2 = skmetrics.r2_score(surf_runoff_true, surf_runoff_pred)
    sub_r2 = skmetrics.r2_score(sub_runoff_true, sub_runoff_pred)
    reportfile.write('Overall accuracy: \n')
    reportfile.write('Surface Runoff RMSE: ' + str(surf_rmse) + '; R2: ' + str(surf_r2) + '\n')
    reportfile.write('Subsurface Runoff RMSE: ' + str(sub_rmse) + '; R2: ' + str(sub_r2) + '\n')
    reportfile.close()

    ts_comps = {'Surface Runoff*': surf_runoff_pred, 'Surface Runoff': surf_runoff_true,
                'Subsurface Runoff*': sub_runoff_pred, 'Subsurface Runoff': sub_runoff_true}
    ts_comps_df = pd.DataFrame(ts_comps)
    styles = ['b--', 'b-', 'r--', 'r-']
    tsax = ts_comps_df.plot(figsize=(10, 5),
                            title='Estimation of Surface and Subsurface runoff using Linear-Regression\n ' + \
                            'considering precipitation and temperature',
                            style=styles,
                            legend=False)
    plt.ylabel('Runoff [m]')
    plt.text(0.05, 0.8,
             'Surface Runoff:\nRMSE: ' + '{0:.4f}'.format(surf_rmse) + '\nR2: ' + '{0:.2f}'.format(
                 surf_r2) + '\n\n' + \
             'Subsurface Runoff:\nRMSE: ' + '{0:.4f}'.format(sub_rmse) + '\nR2: ' + '{0:.2f}'.format(sub_r2),
             horizontalalignment='left',
             verticalalignment='center',
             transform=tsax.transAxes, fontsize=10)
    plt.ylim(0, 0.12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('//projectdata.eurac.edu/projects/seclifirm/ENEL case study/monthly_models/' + area + '/no_snow_linreg.png', dpi=600)