rm(list=ls())
gc()
library(lme4)
library(lmerTest)
library(multcomp)
library(dplyr)
library(report)
library(car)
library(dplyr)
library(e1071)
library(effectsize)
library(performance)
library(multtest)

test_norm <- function(model) {
  resids <- resid(model)
  kurt <- kurtosis(resids)
  skew <- skewness(resids)
  cat("Testing normality\n")
  cat('Shapiro-wilk test')
  print(shapiro.test(resid(model)))
  cat("Skewness should be between -3 and +3 (best around zero)\n")
  cat(skew, "\n\n")
  cat("Excess kurtosis (i.e., absolute kurtosis - 3) should be less than 4; ideally around zero\n")
  cat(kurt, "\n")
  
  return(list(kurt = kurt, skew = skew))
}
z_norm_the<-function(df, var){
  df <- df %>% filter(!is.na(.[[var]]))
  
  df[[var]]<-df[[var]]
  mean<-mean(df[[var]])
  std<-sd(df[[var]])
  df$centered_var<-df[[var]]-mean
  df$z_score<-df$centered_var/std 
  return(df)
}


sigcode<-function(pval){
  if(pval<0.001){
    sig='***'
  } else if(pval<0.01){
    sig='**'
  }else if(pval<0.05){
    sig='*'
  }else if(pval<0.1){
    sig='.'
  }else{
    sig=''
  }
  return(sig)
}




dir_foraging_data='/media/ll16598/IMAGINAL250/SZ/SZ_tob_utt_results/SZ_foraging_atoms/'
dir_save_p=paste0(dir_foraging_data,'stats_results/')



analysis='noun'


files<-'.csv'

min_steps=10
standard=10



transf_variable_list<-c('log',
                        'log',
                        'log',
                        'sqrt',
                        'none',
                        'none',
                        'log',
                        'none')
variable_list<-c( 
  'num_atoms',
  'repeat_ratio_contig',
  'entropy',
  'mean_cluster_length',
  'mean_atoms_visited',
  'all_switching_jumps',
  'all_within_clust_jumps',
  'mean_words_per_utterance'
  
  
)

#############START OF LOOP#########################
{

data_dir<-paste0(working_dir, 'topic_dfs/')
IPII_metadata_dir<-paste0(working_dir, 'IPII_database_for_IU_R01_prelim_data.csv')
print('Running stats on:')
print(base_name)
IPII_topic_run_data_dir<-paste0(dir_foraging_data, file)

IPII_metadata<-read.csv(IPII_metadata_dir)
IPII_topic_run_data<-read.csv(IPII_topic_run_data_dir)
nrow(IPII_topic_run_data)

IPII_topic_run_data$total_steps<-IPII_topic_run_data$length
colnames(IPII_topic_run_data)

IPII_topic_run_data$subject <- sub("_.*", "", IPII_topic_run_data$filename)
IPII_topic_run_data$subject<-as.numeric(IPII_topic_run_data$subject)
nrow(IPII_topic_run_data)

extra_cols <- setdiff(names(IPII_metadata), names(IPII_topic_run_data))
metadata_trimmed <- IPII_metadata %>%
  select(subject, all_of(extra_cols))


IPII_topic_run_data <- IPII_topic_run_data %>%
  left_join(metadata_trimmed, by = "subject")


nrow(IPII_topic_run_data)

IPII_topic_run_data$subject<-as.factor(IPII_topic_run_data$subject)
IPII_topic_run_data$diagnosis<-as.factor(IPII_topic_run_data$racial_ethnic_identity)


IPII_topic_run_data$total_steps<-IPII_topic_run_data$n_nouns
IPII_topic_run_data<-subset(IPII_topic_run_data, IPII_topic_run_data$total_steps>=min_steps)

p_vals_df <- data.frame(
                      variable = character(),
                      response = character(),
                      cognitive_p_value  = numeric(),
                      negative_p_value  = numeric(),
                      positive_p_value  = numeric(),
                      cognitive_sig_code=character(),
                      negative_sig_code=character(),
                      positive_sig_code=character(),
                      effect_r=numeric(),
                      r_p_value=character(),
                      coeff_cog  = numeric(),
                      coeff_neg  = numeric(),
                      coeff_pos  = numeric(),
                      cog_effect=numeric(),
                      cog_cilo=numeric(),
                      cog_cihi=numeric(),
                      neg_effect=numeric(),
                      neg_cilo=numeric(),
                      neg_cihi=numeric(),
                      pos_effect=numeric(),
                      pos_cilo=numeric(),
                      pos_cihi=numeric(),
                      cog_effect2=character,
                      neg_effect2=character,
                      pos_effect2=character,
                      
                      cognitive_f_value       = numeric,
                      negative_f_value =numeric,
                      positive_f_value=numeric,
                      
                      diagnosis_p_value=numeric(),
                      diagnosis_sig_code=character(),
                      kurtosis=numeric(),
                      skewness=numeric(),

                      N=numeric(),
                      stringsAsFactors = FALSE)

trans_id=0
for (var in variable_list) {
    trans_id=trans_id+1
    df_to_test<-IPII_topic_run_data
    df_to_test[[var]]<-as.numeric(df_to_test[[var]])
    df_to_test<-remove_bracs(df_to_test, var)
    df_to_test <- df_to_test %>% filter(!is.na(.[[var]]))
    df_to_test<-z_norm_the(df_to_test,var)
    df_to_test$variable <- df_to_test$z_score
    trans<-transf_variable_list[trans_id]
    if (trans=='log'){
    df_to_test$variable<-log(as.numeric(df_to_test$variable)+standard)}else if (trans=='sqrt'){
      df_to_test$variable<-sqrt(as.numeric(df_to_test$variable)+standard)}
    
    model <- lm(variable ~ PANSS_cognitive_sx_total_LysakerFactor+PANSS_negative_sx_total_LysakerFactor+PANSS_positive_sx_total_LysakerFactor +total_steps, data = df_to_test)

    vifs <- car::vif(model)
    plot(hist(resid(model)))
    
    if (any(vifs > 5)) {
      warning("⚠️ High multicollinearity detected: some VIFs > 5\n",
              paste(names(vifs)[vifs > 5], collapse = ", "))
    } else {
      message("✅ VIF check passed: all predictors < 5")
    }
    results <- test_norm(model)
    kurt<-results$kurt
    skew<-results$skew
    model_to_predict<-model

    anova_model <- car::Anova(model, type = "II")
    anova_model
    sum<-summary(model)
    sum
    final_model<-model
    if (MIXED==TRUE){
      cognitive_p_value <- anova_model[1, 3]
      negative_p_value <- anova_model[2, 3]
      
      positive_p_value <- anova_model[3, 3]
    }else{
    cognitive_p_value <- anova_model[1, 4]
    negative_p_value <- anova_model[2, 4]
    positive_p_value <- anova_model[3, 4]}
    
    
    cognitive_f_value <- anova_model[1, 3]
    negative_f_value <- anova_model[2, 3]
    positive_f_value <- anova_model[3, 3]
    raw_pvals <- c(cognitive_p_value, negative_p_value,positive_p_value)
    tsbh_pvals <- p.adjust(raw_pvals, method = "BH")
    
    names(tsbh_pvals) <- c("cognitive", "negative", "positive")
    cognitive_p_value <- unname(tsbh_pvals["cognitive"])
    negative_p_value  <- unname(tsbh_pvals["negative"])
    positive_p_value  <- unname(tsbh_pvals["positive"])

    std_coefs <- standardize_parameters(model)
    
    cog_effect<-std_coefs[2,2]
    cog_cilo<-std_coefs$CI_low[2]
    cog_cihi<-std_coefs$CI_high[2]
    
    neg_effect<-std_coefs[3,2]
    neg_cilo<-std_coefs$CI_low[3]
    neg_cihi<-std_coefs$CI_high[3]
    
    pos_effect<-std_coefs[4,2]
    pos_cilo<-std_coefs$CI_low[4]
    pos_cihi<-std_coefs$CI_high[4]
    if(traj_rf==TRUE){
    diagnosis_p_value <- anova_model[2, 3]}else{
      diagnosis_p_value=0
    }
    
    sum<-summary(model)
    coeff <- sum$coefficients
    coeff_cog <- coeff["PANSS_cognitive_sx_total_LysakerFactor", "Estimate"]
    coeff_neg <- coeff["PANSS_negative_sx_total_LysakerFactor", "Estimate"]
    coeff_pos <- coeff["PANSS_positive_sx_total_LysakerFactor", "Estimate"]
    fmt2 <- function(x) sprintf("%.2f", x)   # vectorised
    
    cog_effect2 <- sprintf("%s[%s, %s]",
                           fmt2(cog_effect),
                           fmt2(cog_cilo),
                           fmt2(cog_cihi))
    
    neg_effect2 <- sprintf("%s[%s, %s]",
                           fmt2(neg_effect),
                           fmt2(neg_cilo),
                           fmt2(neg_cihi))
    
    pos_effect2 <- sprintf("%s[%s, %s]",
                           fmt2(pos_effect),
                           fmt2(pos_cilo),
                           fmt2(pos_cihi))

    p_vals_df <- rbind(
      p_vals_df,
      data.frame(
        variable               = var,
        response               = 'NA',
        cognitive_p_value       = cognitive_p_value,
        negative_p_value =negative_p_value,
        positive_p_value=positive_p_value,
        cognitive_sig_code      = sigcode(cognitive_p_value),
        negative_sig_code      = sigcode(negative_p_value),
        positive_sig_code      = sigcode(positive_p_value),
        coeff_cog        = coeff_cog,
        coeff_neg=coeff_neg,
        coeff_pos=coeff_pos,
        cog_effect=cog_effect,
        cog_cilo=cog_cilo,
        cog_cihi=cog_cihi,
        neg_effect=neg_effect,
        neg_cilo=neg_cilo,
        neg_cihi=neg_cihi,
        pos_effect=pos_effect,
        pos_cilo=pos_cilo,
        pos_cihi=pos_cihi,
        
        cog_effect2=cog_effect2,
        neg_effect2=neg_effect2,
        pos_effect2=pos_effect2,
                           
        cognitive_f_value       = cognitive_f_value,
        negative_f_value =negative_f_value,
        positive_f_value=positive_f_value,
        effect_r=0,
        r_p_value=sigcode(1),
        diagnosis_p_value=diagnosis_p_value,
        diagnosis_sig_code=sigcode(diagnosis_p_value),
        kurtosis               = kurt,
        skewness               = skew,
        N=nrow(df_to_test),
        stringsAsFactors       = FALSE
        
      )
    )
  }


write.csv(p_vals_df, file = paste0(dir_save_p,base_name, "_stats.csv"), row.names = FALSE)

}



