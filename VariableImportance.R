##example adopted from: Interpretable_Machine_Learning
## https://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/CompareRandomForestPackages.html#7_iml_package:_

##############################################################
rm(list=ls())
#options(java.parameters = "-Xmx10g")
#load R libraries
library(ggplot2)
library(iml)
library(ranger)

##load the data
load("Cohort_hpcData.RData")

lossname = 'MAE'

Y = test$avg_pdc 
X = test[ ,cov_list_dummy[[1]]]
X[,"comor_combined"] = 1*X[,"comor_combined"] 

m.lzn.qrf <- ranger(y= Y, x = X, 
                    data = test, 
                    quantreg = TRUE,
                   importance = 'permutation',
                    keep.inbag=TRUE,  # needed for QRF
                   scale.permutation.importance = TRUE,
                    mtry = 10, ntrees = 500 )


barplot(sort(m.lzn.qrf$variable.importance, decreasing=TRUE)[50:1], horiz = TRUE, las = 1,
        main = 'Variable Importance Measure', cex.names = 0.5)

predictor <- Predictor$new(model = m.lzn.qrf, data=X, y = Y)
############
groups = list(
 age = c("age"),  
 sex =  c("sex_Male"), 
height_weight = c("height_cm","weight_kg") ,                              
GDMT_class =c("ARNI_ACEI_ARB",  "MRA","BB", "SGLT2I","n_meds"),                                  
 days_on_gdmt = c("days_on_GDMT"),                            
 polypharmacy = c("polypharmacy_all"),                    
  pharmacy_traveltime = c("distance", "duration_car",
                          "duration_foot"),                           
 comorbidities = c("chf",  "carit","valv","pcd","pvd",                                    
                   "hypunc", "hypc", "ond", "cpd", "diabunc",                                 
                   "diabc", "hypothy" , "rf","ld", "solidtum",           
                   "rheumd"  ,  "coag" , "obes" , "wloss",
                   "fed" , "dane", "depre" ,"ischemic",'comor_combined'),
frequency_pastvisits = c("n_outpt_pastyr","n_hosp_pastyr",  "n_ED_pastyr") ,
vitals = c( "systolic_min_val" , "systolic_max_val", 
              "systolic_mean_value", 
            "diastolic_min_val","diastolic_max_val", 
                "diastolic_mean_value",
            "pulse_num_min_val", "pulse_num_max_val",
          "pulse_num_mean_value"),
frequency_vitals = c( "systolic_n_obs","diastolic_n_obs","pulse_num_n_obs"),
ejection_fraction = c("ejection_fraction_cat_EF_ov_40",
                      "ejection_fraction_cat_Missing"),
frequency_ef = c("ejection_fraction_num_n_obs"),
frequency_labs = c("sodium_num_n_obs",  "potassium_num_n_obs", 
         "iron_num_n_obs", "bnp_num_n_obs" ,
         "hemoglobin_num_n_obs", "creatinine_num_n_obs"),
Race_Ethnicity = c("race_ethnicity_Asian", "race_ethnicity_Black_or_African_American",
                   "race_ethnicity_Hispanic", "race_ethnicity_White"),
language = c("language_recode_English", "language_recode_Other",
             "language_recode_Russian"),
smoking = c("smoking_recode_Current_Former","smoking_recode_Never"),
insurance = c("insurance_Commercial","insurance_Medicaid","insurance_Medicare",  
              "if_medicaid"),
prior_pdc = c("prior_avg_pdc_cat_0_0.29",  "prior_avg_pdc_cat_0.29_0.56",
              "prior_avg_pdc_cat_0.56_0.74", "prior_avg_pdc_cat_0.74_0.84",
              "prior_avg_pdc_cat_0.84_0.91", "prior_avg_pdc_cat_0.91_0.96",
              "prior_avg_pdc_cat_0.96_0.99", "prior_avg_pdc_cat_0.99_1.00"),             
missed_appointments = c("if_noshow_No","if_noshow_No_visit"),                      
index_encounter_type = c("category_ED_or_Hospitalization", "category_Outpatient"),                     
COC_index = c("COC_quantile_COC_Q2_0.26_0.43","COC_quantile_COC_Q3_Q4__0.43_1",
              "COC_quantile_Missing"))                   
# neighborhood_poverty = c("pov_below_pc","pov_below_1.5pc","gini"),
# neighborhood_race_ethnicity = c("raceeth_nonhisp_white_pc","raceeth_nonhisp_black_pc",
#                                 "raceeth_nonhisp_asian_pc","hisp_pc", "speak_eng_pc"),
# neighborhood_housing_burden = c("rent_burden","hhvalue_med",  
#                                 "hhinc_med" ,    "hh_size_avg",                             
#                                 "owner_pc",                                
#                                 "renter_pc"),                          
# neighborhood_pharmacy_density = c("popden_6211"),
# neighborhood_unemployment = c("unemploy_pc","ft_employment_pc"),
# neighborhood_internet_access = c("internet_pc"),                             
# neighborhood_education = c("edu_lths_pc","edu_coll_pc2"),
# neighborhood_transit = c("transit_car_pc", "transit_public_pc","transit_walk_pc",
#                               "travel_time_ls30min_pc","travel_time_30to59min_pc",
#                                   "travel_time_60to89min_pc"),                
# neighborhood_public_assistance = c("public_assist_pc"),                          
# neighborhood_walkscore = c("Walk_Score"),                              
# neighborhood_parkaccess = c("park_access"),                             
# neighborhood_pollution = c("PM","Ozone") )


##using feature importance metric based on mae loss
imp.mae <- iml::FeatureImp$new(predictor, loss = "mae", features = groups)

plot(imp.mae)
