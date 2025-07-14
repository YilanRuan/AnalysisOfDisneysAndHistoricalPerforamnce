# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:43:33 2022

@author: yilan
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
# import dowhy
# from dowhy import CausalModel
import warnings
import matplotlib.pyplot as plt
import dowhy
from dowhy import CausalModel
import warnings
from scipy.stats import chi2_contingency
from statsmodels.stats.power import tt_solve_power






#Question 1
disney = pd.read_csv("disney_movies_total_gross.txt", delimiter=('\t'))
# change type to float now or int, info says int
disney["total_gross"] = disney["total_gross"].str.replace('$',"").str.replace(',',"")
disney["total_gross"] = pd.to_numeric(disney["total_gross"]) 

# also do it for inflation_adj_gross
disney["inflation_adjusted_gross"] = disney["inflation_adjusted_gross"].str.replace('[\$\,]',"",regex =True)
disney["inflation_adjusted_gross"] = pd.to_numeric(disney["inflation_adjusted_gross"])

#  dropping the rows with nan
disney.dropna(inplace=True)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#print(disney)
# Data is now cleaned and ready for use


print("Question 1")

print(disney.head())
# Part A -- Descriptive Stat for our variable inflation adjusted gross
print(disney["inflation_adjusted_gross"].describe())   
print("sample size is", len(disney["inflation_adjusted_gross"]))
print("The mean inflated gross income is", np.mean(disney["inflation_adjusted_gross"]))
print("The median inflated gross income is",np.median(disney["inflation_adjusted_gross"]))
print("About 75% of the inflated gross income are less than or equal to",np.percentile(disney["inflation_adjusted_gross"],75))
print("About 90% of the inflated gross income are less than or equal to",np.percentile(disney["inflation_adjusted_gross"],90))
print("The range of the sample data is",np.ptp(disney["inflation_adjusted_gross"]))
print("The 3rd quartile in the sample data is",np.quantile(disney["inflation_adjusted_gross"],0.75))
print("The 1st quartile in the sample data is",np.quantile(disney["inflation_adjusted_gross"],0.25))
print("The interquartile range is",np.quantile(disney["inflation_adjusted_gross"],0.75)-np.quantile(disney["inflation_adjusted_gross"],0.25))
print("The variance of the inflated gross income is",np.var(disney["inflation_adjusted_gross"],ddof=1))
#print("The variance of the exam grades is",round(np.var(Grades,ddof=1),2))#to view two decimal places only
print("The standard deviation of the inflated gross income is",np.std(disney["inflation_adjusted_gross"],ddof=1))
print("The coefficient of variation is",100*np.std(disney["inflation_adjusted_gross"],ddof=1)/np.mean(disney["inflation_adjusted_gross"]),"%")

# Part B -- Hyp Testing

# Checking for 2 conditions before doing the hypothesis test: independance and sample size is 30+

# Check Indepdendance using Chi-Square
stat1, p1, dof1, expected1 = chi2_contingency(disney['inflation_adjusted_gross'])
alpha = 0.05
print("p value is " + str(p1))
if p1 <= alpha:
    print('We can reject the null hyptohesis and conclude that the samples are dependent')
else:
    print('We fail to reject null and cannot conclude that the samples are dependent (so they are independent)') 

# Testing to see if sample size is large enough to use CLT

lenofdata = len(disney['inflation_adjusted_gross'])
print("According to the CLT as long as the sample size is greater than 30, we can apply the CLT.")

#Therefore we have indepency and a sufficiently large sample size so we can do the Hypothesis Test)

#State null and alternative hypothesis

print("Null hyp - The average disney movie from 1937 to 2016 gross less than or equal to $302,872,154.")
print("Alt hyp - The average disney movie from 1937 to 2016 gross more than $302,872,154.")


hyp_val = 302872154
Teststat,pvalone =  stats.ttest_1samp(disney["inflation_adjusted_gross"],hyp_val, alternative="greater")

print("Test Stat is", Teststat)
print("P-val is", pvalone)

sign_level = 0.05
if pvalone <= sign_level:
    print("We reject h0 and can conclude that the average disney movie from 1937 to 2016 gross more than $302,872,154.")
else:
    print("We cannot reject h0 and cannot conclude that the average disney movie from 1937 to 2016 gross more than $302,872,154.")


desiredsamplesize2= tt_solve_power(.2, alpha =.05, power = .8, alternative = "two-sided")
print("the desired sample size is", desiredsamplesize2)

desiredsamplesize3= tt_solve_power(.5, alpha =.05, power = .8, alternative = "two-sided")
print("the desired sample size is", desiredsamplesize3)

desiredsamplesize4= tt_solve_power(.8, alpha =.05, power = .8, alternative = "two-sided")
print("the desired sample size is", desiredsamplesize4)
    
# # =============================================================================
print('Question 2')

# Step 1: Checking for the 3 assumptions to do the anova test :  normality, equal variances, and independance

# Test for normality using Shapiro Wilk Test:
    
shastat, shapval = (stats.shapiro(disney['inflation_adjusted_gross']))
print('pval for shapiro:', shapval)
if shapval <= 0.05:
      print("We reject null and conclude that there is evidence to support the fact that the data is not normally distributed.")
else:
      print("We fail to reject null and cannot conclude that the data is not normally distibuted so the data is normally distributed.")

# failed the normality test so we will need to normalize it:  
disney['log'] = np.log((disney['inflation_adjusted_gross'])) 
disney['cube'] = (disney['log'])  ** 3
print(disney.agg(['skew', 'kurtosis']).transpose())
# after normalizing we see that the skew for the cube column is -.247 so its normal    

disney['cube'].hist(grid=False,
          figsize=(10, 6),
          bins=30)
# plot the graph to see distribution

rows = disney.query('genre == "Adventure"')
r = rows["cube"]

rows1 = disney.query('genre == "Comedy"')
r1 = rows1["cube"]

rows2 = disney.query('genre == "Drama"')
r2 = rows2["cube"]

rows3 = disney.query('genre == "Action"')
r3 = rows3["cube"]

rows4 = disney.query('genre == "Thriller/Suspense"')
r4 = rows4["cube"]

rows5 = disney.query('genre == "Romantic Comedy"')
r5 = rows5["cube"]

rows6 = disney.query('genre == "Documentary"')
r6 = rows6["cube"]

rows7 = disney.query('genre == "Musical"')
r7 = rows7["cube"]

rows8 = disney.query('genre == "Western"')
r8 = rows8["cube"]

rows9 = disney.query('genre == "Horror"')
r9 = rows9["cube"]

rows10 = disney.query('genre == "Black Comedy"')
r10 = rows10["cube"]

rows11 = disney.query('genre == "Concert/Performance"')
r11 = rows11["cube"]

#Test for variability --> perform Bartlett's test 
teststat2, pval2 = stats.bartlett(r,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11)
print('pval for bartlette', pval2)
signlevel = 0.05
if pval2 <= signlevel:
    print("we reject null so each group has different variances")
else:
    print("we fail to reject null so each group has equal variances")
# The p-value (0.011275224549350652) from Bartlett’s test is less than α = .05, 
#Assumption of equal variances is false so we need to do  Welch’s ANOVA 

# independance test --> chi-squared
from scipy.stats import chi2_contingency 
stat3, p3, dof3, expected3 = chi2_contingency(disney['cube'])
print('pval for chi', p3)
if p3 <= alpha:
    print('We can reject the null hyptohesis and conclude that the grouped samples are dependent')
else:
    print('We fail to reject null and cannot conclude that the grouped samples are dependent (so they are independent)')
# Meets all the requirements

# run the Welch Anova

from pingouin import welch_anova
import pingouin as pg 

anovaTest = welch_anova(dv='cube', between='genre', data=disney)
print(anovaTest)
pval4 = 0.000035
if pval4 <= signlevel:
        print("We reject H0. There exists a difference in the inflation_adjusted_gross as the genre changes ")
else:
      print("We fail to reject H0. We cannot conclude that there exists a difference in the inflation_adjusted_gross as the genre changes")


# Tukey for Welch
tukey4diffvar= pg.pairwise_gameshowell(dv='cube', between='genre', data=disney)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(tukey4diffvar)


# print rows that are significant 
tukeyimportant = tukey4diffvar[(tukey4diffvar["pval"] <=  0.05)]
print(tukeyimportant)


# # =============================================================================
#Question 3

disneyhg = []

disneyhg = pd.read_csv("disneydowhy.txt", delimiter="\t")
print(disneyhg.head())
print(disneyhg["Age"].min())
print(disneyhg["Age"].max())

# Normalization (not needed)
# # copy the data
# df_min_max_scaled = disneyhg.copy()
  
# # apply normalization techniques by Column 1
# column = 'Age'
# df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
  
# # view normalized data
# display(df_min_max_scaled)

g1 = disneyhg[(disneyhg["Age"]< 40) & (disneyhg["Preference"] == "Villains")]
g2 = disneyhg[(disneyhg["Age"]>=40) & (disneyhg["Preference"] == "Heroes")]
g3 = disneyhg[(disneyhg["Age"]< 40) & (disneyhg["Preference"] == "Heroes")]
g4 = disneyhg[(disneyhg["Age"]>=40) & (disneyhg["Preference"] == "Villains")]

x =[[len(g1.index), len(g2.index)], [len(g3.index), len(g4.index)]]

print(x)

teststat, pval, dof, expected_counts = chi2_contingency(x)
print(expected_counts)
print("the pval is", pval)


# # =============================================================================

#Question 4

disneydata2 = pd.read_csv("disneydatabaycik2.txt", delimiter=('\t'))

dataframenew = pd.get_dummies(disneydata2, columns=['DisneyPlus Recommendation System'], drop_first=True)

dataframenew.rename(columns = {"DisneyPlus Recommendation System_Yes" : "RecSystemYes", "PopulationAverageAge " : "age", "Disney+ Subscribers (in thousands)" : "subscribers", "Average Annual Income (in thousands)": "Income"}, inplace = True)


column = 'age'
dataframenew["age"] = (dataframenew[column] - dataframenew[column].min()) / (dataframenew[column].max() - dataframenew[column].min())
#print(dataframenew["age"])
column1 = 'Income'
dataframenew["Income"] = (dataframenew[column1] - dataframenew[column1].min()) / (dataframenew[column1].max() - dataframenew[column1].min())
#print(dataframenew["Income"])
column2 = 'subscribers'
dataframenew["subscribers"] = (dataframenew[column2] - dataframenew[column2].min()) / (dataframenew[column2].max() - dataframenew[column2].min())
#print(dataframenew["subscribers"])
pd.set_option("display.max_columns", None)
# print(dataframenew.head())


dataframenew["subscribers"] = dataframenew["subscribers"] * (80 - 18) + 18
dataframenew["age"] = dataframenew["age"] * (50 - 30) + 30
dataframenew["Income"] = dataframenew["Income"] * (80 - 18) + 18
print(dataframenew.head())                                                                   
#print(dataframenew.describe())

dataframenewage = dataframenew.iloc[:,2]
dataframeneincome = dataframenew.iloc[:,3]
dataframenesub = dataframenew.iloc[:,4]
print(dataframeneincome.head())
dataframeneincome.hist(grid=False,
        figsize=(10, 6),
        bins=100)
plt.title("Annual Average Income Distribution")
plt.xlabel("Income (In thousands) Per Region")
plt.ylabel("Count")
plt.show()

dataframenewage.hist(grid=False,
          figsize=(10, 6),
          bins=50)
plt.title("Age Distribution")
plt.xlabel("Average Age Per Region")
plt.ylabel("Count")
plt.show()

dataframenesub.hist(grid=False,
        figsize=(10, 6),
        bins=100)
plt.title("Subscribers Distribution")
plt.xlabel("Number of Subscribers Per Region")
plt.ylabel("Count")
plt.show()



print(dataframenew.columns)

print(dataframenew.head())
X = dataframenew.iloc[:,6]
Y = dataframenew.iloc[:,4]
X = sm.add_constant(X)


datafit = sm.OLS(Y, X).fit()
print(datafit.summary())


print(dataframenew.head())
X2 = dataframenew.iloc[:,np.r_[6:7,3:4]]
Y2 = dataframenew.iloc[:,4]
print(Y2.head())
X2 = sm.add_constant(X2)   
est3 = sm.OLS(Y2,X2) #
est4 = est3.fit() #fits the data into the model
print(est4.summary())

print(dataframenew.columns)


print(dataframenew.head())
causal_graph = """digraph {


"RecSystemYes"; 
"Subscribers"; 
"Income"; 
       
   
"Income" -> "Subscribers";
"RecSystemYes"-> "Subscribers";

}"""





model1= CausalModel(
        data = dataframenew,
        graph = causal_graph.replace("\n", " "),
        treatment="RecSystemYes",
        outcome="Subscribers")
model1.view_model()

from IPython.display import Image, display
display(Image(filename="causal_model.png"))


# # =============================================================================

#Question 5

disneyplus = []
disneyplus = pd.read_csv("disneyplus1.txt", delimiter ='\t')
print(disneyplus.head())
disneyplus = disneyplus.dropna()

print(disneyplus.keys())






# disneyplus['DisneyPlus_Recommendation_System'] = disneyplus['DisneyPlus_Recommendation_System'].replace(['Yes'],'1' )
# disneyplus['DisneyPlus_Recommendation_System'] = disneyplus['DisneyPlus_Recommendation_System'].replace(['No'],'0') 
# disneyplus = pd.get_dummies(disneyplus, columns = ['DisneyPlus_Recommendation_System'],drop_first= True)
# disneyplus["DisneyPlus_Recommendation_System"]= disneyplus["DisneyPlus_Recommendation_System_Yes"].astype(bool)

# print(disneyplus)

# print(disneyplus["DisneyPlus_Recommendation_System"])


# disneyplus["Population_Average_Age1"]= disneyplus.iloc[:,2]
# print(disneyplus["Population_Average_Age1"])

disneyplus = pd.get_dummies(disneyplus, columns = ['DisneyPlus_Recommendation_System'],drop_first= True)
disneyplus["DisneyPlus_Recommendation_System"]= disneyplus["DisneyPlus_Recommendation_System_Yes"].astype(bool)

disneyplus.rename(columns = {"PopulationAverageAge " : "age", "DisneyPlus_Recommendation_System" : "RecSystem", "Disney+ Subscribers" : "subscribers", "Average_Annual_Income"  : "Income"}, inplace = True)
print(disneyplus.columns)

disneyplus.info()

disneyplus.head()

# copy the data
# df_min_max_scaled = disneyplus.copy()
  
# column = 'age'
# disneyplus["age"] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
# print(disneyplus["age"])
# column1 = 'Income'
# disneyplus["Income"] = (df_min_max_scaled[column1] - df_min_max_scaled[column1].min()) / (df_min_max_scaled[column1].max() - df_min_max_scaled[column1].min())
# print(disneyplus["Income"])
# column2 = 'subscribers'
# disneyplus["subscribers"] = (df_min_max_scaled[column2] - df_min_max_scaled[column2].min()) / (df_min_max_scaled[column2].max() - df_min_max_scaled[column2].min())
# print(disneyplus["subscribers"])
# pd.set_option("display.max_columns", None)
# print(disneyplus)

# shastat, shapval = (stats.shapiro(disneyplus["Income"]))
# print(shapval)
# if shapval <= 0.05:
#     print("We can reject The null hypothesis so your data is not normal")
# else:
#     print("Fail to reject null so pval is normal")



causal_graph = """
digraph {
age;
RecSystem;
subscribers;
Income;
RecSystem -> subscribers;
age -> Income;
age -> subscribers;
age -> RecSystem;
}
"""

from dowhy import CausalModel
from IPython.display import Image, display

model= CausalModel(
        data = disneyplus,
        graph=causal_graph.replace("\n", " "),
        treatment='RecSystem',
        outcome= 'subscribers')

model.view_model()
display(Image(filename="causal_model.png"))


estimands = model.identify_effect()
print(estimands)

causal_estimate_reg = model.estimate_effect(estimands, method_name="backdoor.linear_regression",test_significance=True)
print("the causal estimate reg value is",causal_estimate_reg.value)

refute_results_reg = model.refute_estimate(estimands, causal_estimate_reg, method_name="placebo_treatment_refuter", num_simulations=20)
print(refute_results_reg)

refutel = model.refute_estimate(estimands,causal_estimate_reg, method_name = "data_subset_refuter", num_simulations=20)
print(refutel)

causal_estimate_match = model.estimate_effect(estimands, method_name="backdoor.propensity_score_matching",target_units="ate",test_significance=True)
print(causal_estimate_match)
print(causal_estimate_match.value)

refutel_match = model.refute_estimate(estimands,causal_estimate_match,  method_name = "data_subset_refuter", num_simulations=20)
print(refutel_match)

refutel_match = model.refute_estimate(estimands,causal_estimate_match, method_name = "placebo_treatment_refuter", num_simulations=20)
print(refutel_match)

print(causal_estimate_reg)
print("the causal estimate reg value is", causal_estimate_reg.value)
print(causal_estimate_match)
print("the causal estimate match is",causal_estimate_match.value)

