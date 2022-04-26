## HMM model for calculating the economic burden of diabetes mellitus

### Differences from the most researches
We focused on the macroeconomic burden of diabetes mellitus. Previous studies calculated the loss as direct medical expenditures and indirect average income of labor. From the perspective of macroeconomic of sociaty, the medical payments were part of incomes of medical institutions instead of the loss of macroe conomics. The labor loss due to diabetes mellitus can be supplied by other workers. Thus the average income of these labor for diabetes mellitus can not be calculated as total loss of indirect macroeconomic costs of diabetes mellitus.

As for direct medical costs, the payments for medical care were transfers from investment to consumptions rather than loss of macroeconomics. We regarded the direct individual- or insurance- treatment cost as reduction of investment. As for indirect costs, we focused the change of labour supplement, including three different aspects, labor loss due to mortality, labor hour loss due to morbidity, and informal labor loss due to informal caregives. Thus, we estimated the counterfactual labor supplement to calculated the counterfactual productions for estimated our labor loss, not simply multiplied average income and labors.

```
According to American Diabetes Association, Sundar S. Shrestha[1] reported that economic costs attributable to diabetes was $5.9 billion (2017 U.S. dollars). It includes direct medical costs based on noninstutionalized population and institutionalized population. It also includes indirect costs such as morbidity-related productivity losses (absenteeism costs, presenteeism costs, household productivity losses and inability costs) and mortalityity-related productivity losses (lifetime labor earnings and household productivity costs). People with diagnosed diabetes incur average medical expenditures of $16,752 per year, of which about $9,601 is attributed to diabetes.

The total estimated 2017 cost of diagnosed diabetes of $327 billion includes $237 billion in direct medical costs and $90 billion in reduced productivity.

The largest components of medical expenditures are:
	Hospital inpatient care (30% of the total medical cost)
	Prescription medications to treat complications of diabetes (30%)
	Anti-diabetic agents and diabetes supplies (15%)
	Physician office visits (13%)

Indirect costs include:
	Increased absenteeism ($3.3 billion)
	Reduced productivity while at work ($26.9 billion) for the employed population
	Reduced productivity for those not in the labor force ($2.3 billion)
	Inability to work as a result of disease-related disability ($37.5 billion)
	Lost productive capacity due to early mortality ($19.9 billion)

```




### Data souce introduction
Here we calculate all burdens for 204 countries for diabetes mellitus.
The code for preparing data is released.

#### Informal labor multiplier
We calculate informal labor multiplier as 0.11 (0.05-0.23), that is, 0.11 informal labor per patient.
We adopted weekly hour 4.0(1.9-8.3) provided by informal labor reported by Langa et al.[2], and 35.9 weekly working hour per employed person reported by ILOSTAT [https://ilostat.ilo.org/topics/working-time/]. Then we have informal multiplier as 0.11 (0.05-0.23)


```
Table. Weekly Hours and Yearly Cost of Informal Care, by Diabetes Mellitus (DM) Category
 	
| 	Hours Per Week (95% CI) 	| | | | 	Cost Per Year (US $)b (95% CI) 	| 
| --- | --- | --- | --- | --- | 
| DM Category |	Unadjusted | Adjusteda | Unadjusted | Adjusted |
| No DM | 6.1 (5.7–6.5) | 6.6 (6.2–6.9) | $2,600 (2,400–2,800) | $2,800 (2,600–2,900) | 
| DM, taking no medication | 10.5 (10.0–11.0) | 9.0 (8.5–9.5) | $4,500 (4,200–4,700) | $3,800 (3,600–4,000) | 
| DM, taking oral medication only | 10.1 (9.6–10.6) | 8.5 (8.1–9.0) | $4,300 (4,100–4,500) | $3,600 (3,400–3,800) | 
| DM, taking insulin | 14.4 (13.8–15.0) | 10.6 (10.1–11.2) | $6,100 (5,900–6,400) | $4,500 (4,300–4,800) |

Note: CI = confidence interval.

Langa et al. 2002[2] reported 
2.4–4.4 weekly hours (annual $1000-$1900 per person) attributable to diabetes mellitus without medication per person, 
1.9–4.0 weekly hours (annual $800-$1700 per person) to diabetes mellitus with oral medication, 
and 4.0–8.3 weekly hours (annual $1700-$3500 per person) to diabetes mellitus with insulin. 
Chatterjee et al. 2011[3] reported that it costs 14.9 hours of informal care per person per week in Tailand. 

Informal labor multiplier is weekly hours 1.9-8.3 divided by 35.9 average weekly hours, ranging from 1.9/35.9 - 8.3/35.9 = 0.05-0.23.
```
For other data, please refre to ./dataPreparation/Readme.md.

### Step 0. Prepare IHME data
First, we can download data from the IHME website.
http://ghdx.healthdata.org/gbd-results-tool, click IHME Data, GBD Results Tool. We have prepared the javascript codes for download IHME data for diabetes. You can also download with codes in bigdata/javascript.level3 if you are familiar with javascript in chrome. We download all 204 countries diabetes mellitus data from 2010-2019.

Then, we should run the codes in dataPreparation/Mortality_morbidity/IHME.ipynb to deal with the IHME data.

For simplicity, you can skip all above steps and get the prepared data for HMM by just run the following codes in bash.

```bash
bash IHME_prepared_download.sh
```

### Step 1. Run HMM model
Here we run HMM model to get predict the economic burden.
The default config is scenario 'val', consider treatment cost, consider morbidity, and set the discount as 0.02, consider the informal labor multiplier as 0.0. You can run the model with your settings.
I run for 204 countries on a MacBook Air (M1, 2020), it takes about 1 min for each parameter. 

```python
python HMM_main.py -t 1 -m 1 -i 0.5 -d 0 -s 'val' 2>&1 | tee logs/log_t1m1i5d0v.txt
```

There are at most "-t -m -i -s -d 2X1X4X3X3 = 72" kinds of parameters. We list basic 24 commands in run.sh, please confirm that your Platform has enough CPU to run these processing together before running all commands. Then you can combine all results produced by these commands by running the following command.

```python
python combine.py 
```

An alternative choice is to directly use the download results 'results/aggregate_results.csv' for saving time.

### Step 2. Impute data
We can get 141 countries' data from step 2 due to lacking physical data, saving data, or others. Then we impute the economic burden of all 204 - 141 = 63 countries based on these 141 countries. We need to prepare data of GDP, population data, and 2019 IHME data. See more details in imputation.py. 

You can skip step 2 if generating tables and figures based on the original 141 countries' data.

```python
python imputation.py -i 'results/aggregate_results.csv' -o 'results/aggregate_results_imputed.csv'
```


### Step 3. Generate Tables or Figures
You can generate tables with all aggregate results generated by step 1 or use imputed results by step 2. We provide the script as follows for file "results/aggregate_results_imputed.csv", which including all 204 countries. 

#### Tables, saved in the folders 'tables'
```python
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -d 0
```
```python
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -d 0.02
```
```python
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -d 0.03
```

#### Figures, also saved in the folder 'tables'
To adjust the figure size, color, etc, we advise using jupyter notebook to generate figures.
If run jupyter notebook with Visual Studio Code, you should change "fig.show()" to "plot(fig)" to see figures.

```jupyter
jupyter notebook generate_figures-Choropleth.ipynb
```
```jupyter
jupyter notebook generate_figures-TC_proportion.ipynb
```

### Releasing in the future ...
Codes and scripts for data preparation. Maybe we will provide easy-to-use tools for data preparation if available.
#### Check
We have provided data preparation source code for data collection and preprocessing.
But we do not check it yet on other devices.
#### IHME
The IHME dataset "IHME_p_details.csv" is too large. You can download it at the HTML link. Download the IHME_p_details.csv.zip file, and unzip it to folder dataPreparation/Mortality_morbidity/.
```bash
wget http://file.zhongcao.fit/f/70a0cfbe6a7d4c08b5de/?dl=1 -O dataPreparation/Mortality_morbidity/IHME_p_details.csv.zip
cd dataPreparation/Mortality_morbidity/
unzip IHME_p_details.csv.zip
```

You can get all data files by codes and public datasets in this project. Some datasets are extracted from from papers or government websites.
```
"alpha.csv", capital share in the production function, from Penn World Table[4]
"gd.csv", age list used for HMM.
"dl1_countrycodeorg_country_name.csv". Country names from wiki and World bank.
```

### References

[1] Shrestha SS, Honeycutt AA, Yang W, Zhang P, Khavjou OA, Poehler DC, Neuwahl SJ, Hoerger TJ. Economic Costs Attributable to Diabetes in Each U.S. State. Diabetes Care. 2018 Dec;41(12):2526-2534. doi: 10.2337/dc18-1179. Epub 2018 Oct 10. PMID: 30305349; PMCID: PMC8851543.

[2] Langa KM, Vijan S, Hayward RA, Chernew ME, Blaum CS, Kabeto MU, Weir DR, Katz SJ, Willis RJ, Fendrick AM. Informal caregiving for diabetes and diabetic complications among elderly americans. J Gerontol B Psychol Sci Soc Sci. 2002 May;57(3):S177-86. doi: 10.1093/geronb/57.3.s177. PMID: 11983744.

[3] Chatterjee S, Riewpaiboon A, Piyauthakit P, Riewpaiboon W. Cost of informal care for diabetic patients in Thailand. Prim Care Diabetes. 2011 Jul;5(2):109-15. doi: 10.1016/j.pcd.2011.01.004. Epub 2011 Feb 18. PMID: 21334276.

[4] University of Groningen and University of California. Share of Labour Compensation in GDP at Current National Prices for United States [LABSHPUSA156NRUG], retrieved from FRED, Federal Reserve Bank of St. Louis Davis. 2021. https://fred.stlouisfed.org/series/LABSHPUSA156NRUG (accessed Jan 10 2021).




