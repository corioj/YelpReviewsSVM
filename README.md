# YelpReviewsSVM
SVM classifier trained to determine the emotional valence of a Yelp review based on contents, and thus determine whether an establishment is worth visiting.

Explored the effects of different regularization techniques, data engineering schemes, and evaluation metrics by examining the results on a randomized train/validation split. Furthermore, the techniques of accounting for class imbalance were implemented and examined. 

Final model put forth to be evaluated on an unseen test set was developed by myself and utilized data engineering techniques such as tracking word counts and removing emotionally neutral or unimportant words like conjunctions and pronouns (deemed "fluff"). Final model was chosen based on raw accuracy score after tuning various hyperparameters and evaluating different choices for the SVM kernel.

Self-designed model and data engineering techniques designed final performance achieved top 10% of class on testing data.
