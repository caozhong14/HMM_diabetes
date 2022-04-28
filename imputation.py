#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
import os
import argparse
PPP = True
endyear = 2051
projectStartYear = 2020
DoPCA = False
if DoPCA:
    summaryfile = 'tmpresults/models_summary_prevalence_dalys_pca.txt'
else:
    summaryfile = 'tmpresults/models_summary_prevalence_dalys_nopca.txt'
# ## Part 0. Get population / GDP / per GDP data
coefficient_file = 'tmpresults/models_coefficient.txt'
# In[ ]:


## MAIN FUNCTION - input
## OUTPUT FILE
save_filename_gdp_total = 'data/GDP_TOTAL.csv'
save_filename_pop_total = 'data/POP_TOTAL.csv'
save_filename_gdp_psy = 'data/GDP_PSY.csv'
save_filename_pop_psy = 'data/POP_PSY.csv'




if os.path.exists(save_filename_gdp_total) and \
    os.path.exists(save_filename_pop_total) and \
    os.path.exists(save_filename_gdp_psy) and \
    os.path.exists(save_filename_pop_psy) :
    pass
else:
    ## Prepare the gdp and pop data
    ## INPUT FILE
    # pop_data = pd.read_csv('../data/population_final.csv')
    if PPP:
        gdp_data = pd.read_csv('data/GDP_ppp.csv')
        gdp_cia = pd.read_csv('data/GDP_ppp_cia.csv')
    else:
        gdp_data = pd.read_csv('data/GDP.csv')

    pop_data_total = pd.read_csv('data/population_total.csv')

    # percap = pd.read_csv('../data/GDP_per_cia.csv')

    grow_rate = 0.03


    # In[ ]:


    GDP = gdp_data.set_index('Country Code')
    GDP_total = GDP[[str(i) for i in range(projectStartYear,endyear,1)]].sum(axis=1)
    GDP_total = GDP_total[GDP_total!=0]

    if PPP:
        gdp = gdp_cia.set_index('Country Code')
        gdp = gdp.sort_index()
        years = (endyear - projectStartYear)
        GDP_CIA = ((1+grow_rate) ** years - 1) / grow_rate * gdp['value']
        GDP_filled = pd.concat([GDP_total, GDP_CIA]).to_frame('totalGDP')
    else:
        GDP_filled = GDP_total.to_frame('totalGDP')

    GDP_filled = GDP_filled.reset_index()
    GDP_filled = GDP_filled.drop_duplicates(subset=['Country Code'], keep='first')
    GDP_filled.to_csv(save_filename_gdp_total, index=False)


    POP = pop_data_total.set_index('Country Code')
    POP_total = POP[[str(i) for i in range(projectStartYear,endyear,1)]].sum(axis=1)
    POP_filled = POP_total.to_frame('totalPOP')
    POP_filled = POP_filled.reset_index()
    POP_filled.to_csv(save_filename_pop_total, index=False)


    # In[ ]:


    pop_psy = POP.groupby(['Country Code']).sum()[[str(projectStartYear)]]
    pop_psy = pop_psy.rename(columns={str(projectStartYear):'pop_psy'})
    pop_psy = pop_psy.reset_index()

    gdp_psy1 = GDP[str(projectStartYear)].to_frame('gdp_psy')
    gdp_psy1 = gdp_psy1.dropna()
    gdp_psy2 = gdp['value'].to_frame('gdp_psy')
    gdp_psy = pd.concat([gdp_psy1, gdp_psy2], axis=0).reset_index()
    gdp_psy = gdp_psy.drop_duplicates(subset=['Country Code'], keep='first')

    pop_psy.to_csv(save_filename_pop_psy, index=False)
    gdp_psy.to_csv(save_filename_gdp_psy, index=False)


# ## Part 1. raw estimate for each disease here


# In[ ]:


def get_df(df_result, ConsiderTC, ConsiderMB, informal, discount, scenario):
    df = df_result[(df_result['discount']==discount)&
                   (df_result['ConsiderTC']==ConsiderTC)&
                   (df_result['ConsiderMB']==ConsiderMB)&
                   (df_result['informal']==informal)&
                   (df_result['scenario']==scenario)]
    df = df[df['tax'] > 0]
    return df

def get_IHME_data(df_IHME, disease, scenario):
    countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv')
    df_IHME_disease = df_IHME[df_IHME['cause'] == disease]
    data = pd.pivot_table(df_IHME_disease, index='Country Code', columns='measure', values=scenario)
    if DoPCA:
        x = data[['DALYs (Disability-Adjusted Life Years)', 'Deaths', 'Prevalence', 'Incidence','YLDs (Years Lived with Disability)', 'YLLs (Years of Life Lost)']]
        pca_model = PCA(x, standardize=False, demean=True)
        x_input = pca_model.factors.iloc[:, :2] 
    else:
        x = data[['DALYs (Disability-Adjusted Life Years)']]
        # x = data[['YLLs (Years of Life Lost)', 'Prevalence']]
        x.columns = ['comp_0']
        x_input = x
    x_input.index = countries_info[countries_info['country'].notna()]['Country Code']
    return x_input

def get_Indicator_data():
    countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv')
    Income = countries_info[['Country Code', 'Income group']]
    # print(Income['Income group'].unique())
    col1 = Income['Country Code']
    col2 = (Income['Income group'] == 'High income').astype(int)
    col3 = (Income['Income group'] == 'Upper middle income').astype(int)
    col4 = (Income['Income group'] == 'Lower middle income').astype(int)
    col5 = (Income['Income group'] == 'Low income').astype(int)
    col6 = col2
    df_income = pd.concat([col1, col6], axis=1)
    # df_income.columns = ['Country Code', 'High income', 'Upper middle income', 'Lower middle income', 'Low income']
    df_income.columns = ['Country Code', 'Upper income']
    return df_income

def get_aggregate_data(df_agg, disease, IHME_data):
    df = df_agg[df_agg['disease'] == disease]
    data = df.merge(IHME_data, on='Country Code',how='inner')
    # for IHME_data in IHMElist:
    #     data = data.merge(IHME_data, on='Country Code',how='inner')
    # country_touse = pd.read_csv('../country_touse.csv')
    # data = data.merge(country_touse, on=['Country Code'],how='inner')
    data = data.dropna()
    return data

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def get_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] <= fence_low) | (df_in[col_name] >= fence_high)]
    return df_out

# def fit_ols_model(data):
#     X = sm.add_constant(data[['Prevalence', 'DALYs (Disability-Adjusted Life Years)']])#, 'Incidence','DALYs (Disability-Adjusted Life Years)']])
#     ols_model = sm.OLS(data['tax'], X)
#     # ols_model = sm.RLM(data['tax'], X, M=sm.robust.norms.HuberT())
#     ols_results = ols_model.fit()
#     return ols_results

def fit_ols_model_pca(data):
    # x_input = data[['comp_0', 'comp_1']]
    if DoPCA:
        x_input = data[['comp_0', 'comp_1', 'Upper income']]
    else:
        x_input = data[['comp_0']]
    # x0, x1 = data[['comp_0']].values, data[['comp_1']].values
    # x00 = x0 * x0
    # x11 = x1 * x1
    # x01 = x0 * x1
    # x_input = np.concatenate([x0, x1, x00, x01, x11], axis=1)
    X = sm.add_constant(x_input)#, 'Incidence','DALYs (Disability-Adjusted Life Years)']])
    ols_model = sm.OLS(data['tax'], X)
    # ols_model = sm.RLM(data['tax'], X, M=sm.robust.norms.HuberT())
    ols_results = ols_model.fit()
    return ols_results

# for estimation
def get_estimation_prepare(df_agg, disease, IHME_data):
    df = df_agg[df_agg['disease'] == disease]
    # for IHME_data in IHMElist:
    #     est = est.merge(IHME_data, on='Country Code',how='inner')    
    est = IHME_data.merge(df[['Country Code','tax']],on='Country Code',how='outer')
    est = est[(est['tax'].isnull())]
    # print (len(est))
    return est

def get_estimation_result(STATISTICS_DATA, est, ols_results):
    estdata = est.merge(STATISTICS_DATA, on='Country Code').set_index('Country Code')
    # save_folder = os.path.join(result_folder, disease)
    if DoPCA:
        est['tax'] = ols_results.params[0]+ols_results.params[1]*est['comp_0']+ols_results.params[2]*est['comp_1'] +ols_results.params[3]*est['Upper income'] ##+ols_results.params[3]*est['Incidence']+ols_results.params[3]*est['DALYs (Disability-Adjusted Life Years)']
    else:
        est['tax'] = ols_results.params[0]+ols_results.params[1]*est['comp_0']
    est = est.set_index('Country Code')
    est['GDPloss'] = est['tax']*estdata['totalGDP']
    est['pc_loss'] = est['GDPloss']/(estdata['totalPOP']/(endyear-projectStartYear))
    # convert to billions
    est['GDPloss'] = est['GDPloss']/1e9
    # print(est)
    est = est.reset_index()
    # print(est.columns)
    est = est[['Country Code', 'pc_loss','GDPloss','tax']]
    
    # if not os.path.exists(save_folder):
        # os.makedirs(save_folder)
    # print ('saving estimation results in ', save_folder)
    # est.to_csv(os.path.join(save_folder, 'est_discount_%s.csv'%(discount)))
    return est


# In[ ]:


def Process(df_result, df_IHME, diseases, STATISTICS_DATA, ConsiderTC, ConsiderMB, informal, discount, scenario):
    pieces = []
    df_agg = get_df(df_result, ConsiderTC, ConsiderMB, informal, discount, scenario)
    l1 =  len(df_agg)
    if l1 == 0:
        return df_agg
    Indicator = get_Indicator_data()

    with open(summaryfile, 'a+') as f:
        with open(coefficient_file, 'a+') as f2:   
            for i, disease in enumerate(diseases):
                if (disease == 'Other malignant neoplasms' or disease == 'Neoplasms' or disease == 'Total cancers'):
                    continue
                disease = diseases[i]
                IHMEdata = get_IHME_data(df_IHME, disease, scenario)
                IHMEdata = IHMEdata.merge(Indicator, on='Country Code')
                data = get_aggregate_data(df_agg, disease, IHMEdata)
                data = remove_outlier(data, 'tax')
                ols_results = fit_ols_model_pca(data)
                print("****************************", file=f)
                print("****************************", file=f)
                print("****************************", file=f)
                print(i, disease, file=f)
                print(ols_results.summary(), file=f)
                est_prepare = get_estimation_prepare(df_agg, disease, IHMEdata)
                est = get_estimation_result(STATISTICS_DATA, est_prepare, ols_results)
                est['disease'] = disease
                est['scenario'] = scenario
                est['ConsiderTC'] = ConsiderTC
                est['ConsiderMB'] = ConsiderMB
                est['informal'] = informal
                est['discount'] = discount     
                pieces.append(est)  
                print (disease, file=f2)      
                print(ols_results.params, file=f2)
                print(ols_results.pvalues, file=f2)
                print("R-squared    ", ols_results.rsquared, file=f2)

    df = pd.concat(pieces)
    l2 = len(df)
    print(l1, l2)
    assert(l1 + l2 == 204 * 1)
    
    return df
        
if __name__ == "__main__":

    if not os.path.exists('tmpresults/'):
        os.makedirs('tmpresults')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', type=str, default='tmpresults/aggregate_results.csv') # or 0
    parser.add_argument('-o', '--output', type=str, default='tmpresults/aggregate_results_imputed.csv') # or 0
    parser.add_argument('-g', '--gpdfile', type=str, default='data/GDP_TOTAL.csv') # or 'lower', 'upper'
    parser.add_argument('-p', '--popfile', type=str, default='data/POP_TOTAL.csv') # or 0.02, 0.03
    args = parser.parse_args()
    ## ## MAIN FUNCTION - input
    

    ## INPUT FILE
    df_result = pd.read_csv(args.input)
    diseases = sorted(df_result['disease'].unique())
    GDP_filled = pd.read_csv(args.gpdfile).set_index('Country Code')
    POP_filled = pd.read_csv(args.popfile).set_index('Country Code')
    STATISTICS_DATA = GDP_filled.merge(POP_filled, on='Country Code')
    countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv')
    code_map = dict(zip(countries_info.country, countries_info['Country Code']))
    df_IHME = pd.read_csv("bigdata/data_diabetes/IHME.csv")
    df_IHME['Country Code'] = df_IHME['location'].apply(lambda x:code_map[x])
    df_IHME = df_IHME[df_IHME['year']==2019]
    print(df_result.columns)


    # In[ ]:


    import warnings
    warnings.filterwarnings("ignore")
    print('imputing.........')
    est_pieces = []
    for ConsiderTC in [1, 0]:
        for ConsiderMB in [1]:
            for scenario in ['val', 'lower', 'upper']:
                for informal in [0, 0.05, 0.11, 0.23]:
                    for discount in [0, 0.02, 0.03]:
                        print(ConsiderTC, ConsiderMB, informal, discount, scenario)
                        est_df = Process(df_result, df_IHME, diseases, STATISTICS_DATA, ConsiderTC, ConsiderMB, informal, discount, scenario)
                        est_pieces.append(est_df)
    df = pd.concat(est_pieces)
    df.to_csv('tmpresults/est.csv', index=False)


    # In[ ]:
    df_imputed = pd.concat([df_result, df])
    df_imputed = df_imputed.drop_duplicates(subset=['ConsiderTC','ConsiderMB','informal','discount','scenario','disease','Country Code'], keep='last')
    df_imputed.sort_values(['ConsiderTC', 'ConsiderMB','informal','discount','scenario','disease','Country Code'], inplace=True)
    df_imputed.to_csv(args.output, index=False)

    # In[ ]:
    print('primary data:', len(df_result))
    print('imputed data:', len(df_imputed))

    # In[ ]:




