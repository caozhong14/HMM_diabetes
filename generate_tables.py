import pandas as pd
import numpy as np
import math
import sys
import os
import argparse


class Tables():
    def __init__(self, discount=0, filename='results/aggregate_results_imputed.csv'):
        self.df_input = pd.read_csv(filename)
        self.countries = self.df_input['Country Code'].unique()
        self.default_discount = discount
        self.set_state()
        self.set_params()
        self.df_state = self.get_data()
        

    def set_params(self):
        countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv')
        self.countries_info = countries_info[['Country Code', 'Region', 'Income group', 'WBCountry', 'country']]
        # self.codemap = countries_info.dropna()
        self.endyear = 2051
        self.projectStartYear = 2020
        gdp_total_df = pd.read_csv('data/GDP_TOTAL.csv').set_index('Country Code')
        pop_total_df = pd.read_csv('data/POP_TOTAL.csv').set_index('Country Code')
        gdp_psy_df = pd.read_csv('data/GDP_PSY.csv').set_index('Country Code')
        pop_psy_df = pd.read_csv('data/POP_PSY.csv').set_index('Country Code')
        self.INFODATA = gdp_total_df.merge(pop_total_df, on='Country Code').merge(gdp_psy_df, on='Country Code').merge(pop_psy_df, on='Country Code')
        self.INFODATA = self.INFODATA.merge(countries_info, on='Country Code')
        self.INFODATA = self.INFODATA[self.INFODATA['Country Code'].isin(self.countries)]
        print(len(self.INFODATA))
               
    def set_state(self, state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'}):
        state['discount'] = self.default_discount
        self.state = state

    def get_data(self):
        df = self.df_input[(self.df_input['discount']==self.state['discount'])&
                       (self.df_input['ConsiderTC']==self.state['ConsiderTC'])&
                       (self.df_input['ConsiderMB']==self.state['ConsiderMB'])&
                       (self.df_input['scenario']==self.state['scenario'])]
        self.imputed = df.merge(self.INFODATA, on='Country Code')
        return self.imputed

    def get_group_data(self, identify=['Country Code', 'disease']): #each Country, each disease
        imputed = self.imputed
        group = pd.DataFrame()
        assert 'disease' in identify
        group['GDPloss'] = imputed.groupby(identify).sum()['GDPloss']
        group['GDPlossRatio'] = imputed.groupby(identify).sum()['GDPloss']/imputed['GDPloss'].sum()
        group['tax']=  imputed.groupby(identify).sum()['GDPloss'] * 1000000000 / imputed.groupby(identify)['totalGDP'].sum()#
        group['pc_loss']=  imputed.groupby(identify).sum()['GDPloss'] * 1000000000 /(imputed.groupby(identify)['totalPOP'].sum() / (self.endyear - self.projectStartYear) ) #

        group = group.reset_index()
        group.sort_values([identify[0],'GDPlossRatio'], ascending = [True,False], inplace=True)

        # group['tax'] = group['tax']*1000 # rate - > 1‰
        group['tax'] = group['tax']*100 # rate - > 1%
        return group

    def generate_table1(self):
        identify =['Country Code', 'disease']
        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'lower'})
        self.get_data()
        group1 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'upper'})
        self.get_data()
        group2 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'})
        self.get_data()
        group3 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        lower = group1.groupby('Country Code').sum()
        upper = group2.groupby('Country Code').sum()
        base = group3.groupby('Country Code').sum()
        
        df = base.merge(upper,on='Country Code',suffixes=('', '_upper'))
        df= df.merge(lower,on='Country Code',suffixes=('', '_lower'))
        df = df.merge(self.countries_info, on='Country Code')

        data = df.copy()
        data['totalloss'] = data.apply(lambda row: str(round(1000*row['GDPloss']))+'('+ str(round(1000*row['GDPloss_lower']))+'-'+str(round(1000*row['GDPloss_upper']))+')', axis=1)
        # data['tax ‰'] = data.apply(lambda row: str(round(row['tax'],3))+'('+ str(round(row['tax_lower'],3))+'-'+str(round(row['tax_upper'],3))+')', axis=1)
        data['tax %'] = data.apply(lambda row: str(round(row['tax'],2))+'('+ str(round(row['tax_lower'],2))+'-'+str(round(row['tax_upper'],2))+')', axis=1)
        data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss']))+'('+ str(round(row['pc_loss_lower']))+'-'+str(round(row['pc_loss_upper']))+')', axis=1)
        data = data.sort_values(['Region','country'])[['Region','country', 'WBCountry', 'totalloss','pc_loss','tax %']]
        # df.reset_index().to_csv('tables/tmp_Table1_discount%s.csv'%self.state['discount'], index=False, float_format='%.3f')
        data.to_csv('tables/Table1_discount%s.csv'%self.state['discount'], index=False)

        data1 = group3.groupby('Country Code').head(1).reset_index()
        data2 = group3.groupby('disease').sum().reset_index()
        ranklist = data2.sort_values(['GDPloss'], ascending=False)['disease'].unique().tolist()
        # print(ranklist)
        # ranklist = ['Tracheal, bronchus, and lung cancer', 'Colon and rectum cancer', 'Breast cancer', 'Stomach cancer', 'Liver cancer', 'Pancreatic cancer', 'Brain and central nervous system cancer', 'Leukemia', 'Other neoplasms', 'Non-Hodgkin lymphoma', 'Esophageal cancer', 'Cervical cancer', 'Prostate cancer', 'Ovarian cancer', 'Kidney cancer', 'Lip and oral cavity cancer', 'Bladder cancer', 'Nasopharynx cancer', 'Other pharynx cancer', 'Uterine cancer', 'Malignant skin melanoma', 'Gallbladder and biliary tract cancer', 'Multiple myeloma', 'Larynx cancer', 'Non-melanoma skin cancer', 'Thyroid cancer', 'Hodgkin lymphoma', 'Testicular cancer', 'Mesothelioma', 'Other malignant neoplasms']
        def getrank(disease):
            return ranklist.index(disease) + 1
        data1['rank'] = data1['disease'].apply(getrank)
        rankranklist = sorted(data1['rank'].unique())
        def getrankrank(rank):
            return rankranklist.index(rank) + 1
        data1['rankrank'] = data1['rank'].apply(getrankrank)
        data1.sort_values(['rankrank', 'tax'], ascending = [True, False], inplace=True)
        data1.to_csv('tables/Table1g_discount%s.csv'%self.state['discount'], index=False)
        # data2.to_csv('tables/tmp_diseases_discount%s.csv'%self.state['discount'], index=False)

             
    def generate_table2(self):
        identify =['disease']
        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'lower'})
        self.get_data()
        lower = self.get_group_data(identify)[['disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'upper'})
        self.get_data()
        upper = self.get_group_data(identify)[['disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'})
        self.get_data()
        base = self.get_group_data(identify)[['disease', 'GDPloss', 'tax', 'pc_loss']]
        
        df = base.merge(upper,on='disease',suffixes=('', '_upper'))
        df= df.merge(lower,on='disease',suffixes=('', '_lower'))

        data = df.copy()
        data['totalloss'] = data.apply(lambda row: str(round(row['GDPloss']))+'('+ str(round(row['GDPloss_lower']))+'-'+str(round(row['GDPloss_upper']))+')', axis=1)
        # data['tax ‰'] = data.apply(lambda row: str(round(row['tax'],3))+'('+ str(round(row['tax_lower'],3))+'-'+str(round(row['tax_upper'],3))+')', axis=1)
        data['tax %'] = data.apply(lambda row: str(round(row['tax'],3))+'('+ str(round(row['tax_lower'],3))+'-'+str(round(row['tax_upper'],3))+')', axis=1)
        data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss']))+'('+ str(round(row['pc_loss_lower']))+'-'+str(round(row['pc_loss_upper']))+')', axis=1)
        data = data.sort_values(['GDPloss'], ascending=False)[['disease','totalloss','pc_loss','tax %']]
        data.to_csv('tables/Table2_discount%s.csv'%self.state['discount'], index=False)

    def generate_table3(self):
        identify =['Region', 'disease']
        group1 = self.get_group_data(identify)
        data1 = group1.groupby('Region').head(5).reset_index()
        data_sum1 = group1.groupby('Region').sum().reset_index()
        data_sum1['disease'] = 'global'
        data1 = data1.append(data_sum1)
        data1['location'] = data1['Region']
        data1.sort_values(['Region','GDPlossRatio'], ascending = [True,False], inplace=True)


        identify =['Income group', 'disease']
        group2 = self.get_group_data(identify)
        data2 = group2.groupby('Income group').head(5).reset_index()
        data_sum2 = group2.groupby('Income group').sum().reset_index()
        data_sum2['disease'] = 'global'
        data2 = data2.append(data_sum2)
        data2['location'] = data2['Income group']
        data2.sort_values(['Income group','GDPlossRatio'], ascending = [True,False], inplace=True)

        data = pd.concat([data1, data2])   
        data['burden'] = data.apply(lambda row: str(round(row['GDPloss']))+' ('+   "{0:.1%}".format(row['GDPlossRatio'])   +')', axis=1)
        data['tax %'] = data.apply(lambda row: str(round(row['tax'],2)), axis=1)
        data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss'])), axis=1) 
        data[['location','disease','burden','tax %', 'pc_loss']].to_csv('tables/Table3_discount%s.csv'%self.state['discount'], index=False) 

        global_result = data[data['disease'] == 'global']
        return global_result[['location', 'GDPloss', 'GDPlossRatio']]


    # def generate_table4(self):
    #     identify =['Region', 'disease']
    #     group1 = self.get_group_data(identify)
    #     data1 = group1.groupby('Region').sum().reset_index()
    #     data1['location'] = data1['Region']

    #     identify =['Income group', 'disease']
    #     group2 = self.get_group_data(identify)
    #     data2 = group2.groupby('Income group').sum().reset_index()
    #     data2['location'] = data2['Income group']

    #     data = pd.concat([data1, data2])   
    #     data['burden'] = data.apply(lambda row: str(round(row['GDPloss']))+' ('+   "{0:.1%}".format(row['GDPlossRatio'])   +')', axis=1)
    #     data['tax ‰'] = data.apply(lambda row: str(round(row['tax'],3)), axis=1)
    #     data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss'])), axis=1) 
    #     data[['location', 'burden','tax ‰', 'pc_loss']].to_csv('tables/Table4.csv',index=False)

    #     return data[['location', 'GDPloss', 'GDPlossRatio']]

    def generate_others(self, df_burden):
        data1 = self.INFODATA.groupby('Region').sum().reset_index()
        data1['location'] = data1['Region']
        data1['gdp_psy_Ratio'] = data1['gdp_psy']/data1['gdp_psy'].sum()   
        data1['pop_psy_Ratio'] = data1['pop_psy']/data1['pop_psy'].sum()  
        data1['gdp_psy'] = data1['gdp_psy'] / 1000000000
        data1['pop_psy'] = data1['pop_psy'] / 1000000
        data1['totalGDP_Ratio'] = data1['totalGDP']/data1['totalGDP'].sum()   
        data1['totalPOP_Ratio'] = data1['totalPOP']/data1['totalPOP'].sum()  
        data1['averageGDP'] = data1['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data1['averagePOP'] = data1['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)

        data2 = self.INFODATA.groupby('Income group').sum().reset_index()
        data2['location'] = data2['Income group']
        data2['gdp_psy_Ratio'] = data2['gdp_psy']/data2['gdp_psy'].sum()   
        data2['pop_psy_Ratio'] = data2['pop_psy']/data2['pop_psy'].sum()   
        data2['gdp_psy'] = data2['gdp_psy'] / 1000000000
        data2['pop_psy'] = data2['pop_psy'] / 1000000
        data2['totalGDP_Ratio'] = data2['totalGDP']/data2['totalGDP'].sum()   
        data2['totalPOP_Ratio'] = data2['totalPOP']/data2['totalPOP'].sum()  
        data2['averageGDP'] = data2['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data2['averagePOP'] = data2['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)

        data = pd.concat([data1, data2])

        # df = pd.read_csv('IHME_locations.csv')
        # dff = pd.pivot_table(df, index=['location', 'cause'], columns=['measure'], values='val').reset_index()
        # dff = dff.groupby('location').sum().reset_index()
        # data = data.merge(dff, on='location').merge(df_burden, on='location')
        # data['dalys'] = data['DALYs (Disability-Adjusted Life Years)'] * data['pop_psy'] / 100000
        # data['dalys_radio'] = data['dalys'] /  data['dalys'].loc[:6].sum()
        df = pd.read_csv('bigdata/Total cancers/IHME_TotalCancers_Dalys.csv') # download from IHME website
        df1 = df[(df['cause'] == 'Total cancers') & (df['metric'] == 'Number')]
        dff = pd.pivot_table(df1, index=['location', 'cause'], columns=['measure'], values='val')
        dff['dalys'] = dff['DALYs (Disability-Adjusted Life Years)'] / 1000000
        dff['dalys_radio'] = dff['DALYs (Disability-Adjusted Life Years)'] /  dff.loc['Global', 'Total cancers']['DALYs (Disability-Adjusted Life Years)']
        dff = dff.reset_index()
        data = data.merge(dff, on='location').merge(df_burden, on='location')
        df2 = df[(df['cause'] == 'Total cancers') & (df['metric'] == 'Rate')]
        dff2 = pd.pivot_table(df2, index=['location', 'cause'], columns=['measure'], values='val')
        dff2['dalys_rate'] = dff2['DALYs (Disability-Adjusted Life Years)'] 
        dff2 = dff2.reset_index()
        data = data.merge(dff2, on='location')      
    
        data = data[['location', 'GDPloss', 'GDPlossRatio', 'dalys', 'dalys_radio', 'dalys_rate', 
                    'gdp_psy', 'gdp_psy_Ratio', 'pop_psy', 'pop_psy_Ratio', 
                    'averageGDP', 'totalGDP_Ratio', 'averagePOP', 'totalPOP_Ratio']]
        
        # dff['dalys_radio'] = dff['DALYs (Disability-Adjusted Life Years)'] / dff['DALYs (Disability-Adjusted Life Years)'].sum() * 
        data.loc['Global'] = data.loc[:6].sum(axis=0)
        data.loc['Global', 'location'] = 'Global'
        # print(data)
        data['burden'] = data.apply(lambda row: str(round(row['GDPloss']))+' ('+   "{0:.1%}".format(row['GDPlossRatio'])   +')', axis=1)
        data['dalys 2019'] = data.apply(lambda row: str(round(row['dalys']))+' ('+   "{0:.1%}".format(row['dalys_radio'])   +')', axis=1)
        data['GDP 2020'] = data.apply(lambda row: str(round(row['gdp_psy']))+' ('+   "{0:.1%}".format(row['gdp_psy_Ratio'])   +')', axis=1)
        data['POP 2020'] = data.apply(lambda row: str(round(row['pop_psy']))+' ('+   "{0:.1%}".format(row['pop_psy_Ratio'])   +')', axis=1)
        data['averageGDP'] = data.apply(lambda row: str(round(row['averageGDP']))+' ('+   "{0:.1%}".format(row['totalGDP_Ratio'])   +')', axis=1)
        data['averagePOP'] = data.apply(lambda row: str(round(row['averagePOP']))+' ('+   "{0:.1%}".format(row['totalPOP_Ratio'])   +')', axis=1)
        data[['location', 'burden','dalys 2019', 'GDP 2020', 'POP 2020', 'averageGDP', 'averagePOP']].to_csv('tables/Table4_discount%s.csv'%self.state['discount'], index=False)
        # dff.to_csv('tables/tmp_discount%s.csv'%self.state['discount'], index=False, float_format='%.0f')
        data.to_csv('tables/tmp_moreprecious_discount%s.csv'%self.state['discount'], index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', type=str, default='results/aggregate_results_imputed.csv') 
    parser.add_argument('-d', '--discount', type=float, default=0) # or 0.02, 0.03
    args = parser.parse_args()
    # In[19]:
    mytable = Tables(discount=args.discount, filename=args.filename)
    mytable.generate_table1()
    mytable.generate_table2()
    df_burden = mytable.generate_table3()
    # mytable.generate_table3()
    # df_burden = mytable.generate_table4()
    mytable.generate_others(df_burden)


