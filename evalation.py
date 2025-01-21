from .pretraining import target_arr, labels
from julia import Main

Main.include("modelling.jl")



# geneformer benchmarking models to compare against

# bundle data for plotting
bundled_data = []
bundled_data += [(roc_auc2, roc_auc_sd2, mean_fpr2, mean_tpr2, "SVM rank", "purple", "solid")]
bundled_data += [(roc_auc0, roc_auc_sd0, mean_fpr0, mean_tpr0, "Random Forest rank", "blue", "solid")]
bundled_data += [(roc_auc1, roc_auc_sd1, mean_fpr1, mean_tpr1, "Logistic Regression rank", "green", "solid")]

# plot ROC
plot_ROC(bundled_data, 'Dosage Sensitive vs Insensitive TFs')
