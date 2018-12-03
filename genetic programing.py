import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import sklearn as sk

telescope = pd.read_csv("MAGIC Gamma Telescope Data.csv")

#clean the data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
#to shuffle the data

#give the data ordered index
tele = telescope_shuffle.reset_index(drop = True)
tele_class = tele["Class"]
tele_feature = tele.drop(columns = "Class")

#split training, testing and validation data
feature_train, feature_test, label_train, label_test = sk.model_selection.train_test_split(tele_feature, tele_class, test_size = 0.25,
                                    random_state= 42)

#fit the model genetic algo with the data
tpot_c = TPOTClassifier(generations = 5, verbosity= 2)
tpot_c.fit(feature_train, label_train)

print(tpot_c.score(feature_test, label_test))

tpot_c.export("output.py")






