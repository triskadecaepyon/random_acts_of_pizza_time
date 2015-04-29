# random_acts_of_pizza_time



### Model 1: Logistic Regression

Logistic Regression (train): 61.78% held set 50%


Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.2772  -0.7656  -0.6178  -0.3238   2.3117  

Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)            -1.99233    0.22990  -8.666  < 2e-16 ***
de_community_age       -0.12493    0.02063  -6.057 1.39e-09 ***
month_partTRUE          0.15430    0.10874   1.419   0.1559    
gratitudeFALSE         -0.11997    0.11401  -1.052   0.2927    
hyperlinkTRUE           0.37420    0.20365   1.837   0.0661 .  
reciprocityTRUE         0.31266    0.13463   2.322   0.0202 *  
sentiment_positiveTRUE  0.09301    0.12057   0.771   0.4405    
sentiment_negativeTRUE -0.04592    0.17999  -0.255   0.7986    
request_length          0.06431    0.01644   3.912 9.16e-05 ***
de_karma                0.14016    0.02333   6.009 1.87e-09 ***
posted_beforeTRUE       1.27015    0.21937   5.790 7.04e-09 ***
de_craving             -0.02924    0.02063  -1.417   0.1564    
de_family               0.02441    0.01947   1.253   0.2101    
de_job                  0.02551    0.01691   1.509   0.1313    
de_money                0.03824    0.01829   2.091   0.0365 *  
de_student             -0.01141    0.02031  -0.562   0.5744    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1




### Model 2: Gradient Boost Tree
Gradient Boost Tree (Best Train): 62.4%

Area under the curve: 0.6244
d_before  					 d_before  9.2894975
de_job                         de_job  3.7266598
de_family                   de_family  3.6107731
de_craving                 de_craving  3.2168256
de_student                 de_student  3.1493636
hyperlink                   hyperlink  2.2448240
reciprocity               reciprocity  2.1181661
month_part                 month_part  1.5735193
gratitude                   gratitude  1.4219262
sentiment_positive sentiment_positive  0.7915072
sentiment_negative sentiment_negative  0.5932222


