import numpy as np
import lda

# in order to run me, call me from the root of the project like so:
# python -m features.example_feature_set

# 1. Import feature_set.FeatureSet
from feature_set import FeatureSet

# 2. Inherit from FeatureSet
class LDADocTopicFeatureSet(FeatureSet):

    def __init__(self):
        # 3. Initialize the FeatureSet super class, if necessary
        # By default, verbose and useBinaryFileMode are False
        # verbose prints debugging information about how the FeatureSet class
        #   extracts or loads features
        # useBinaryFileMode causes the outputFile to be opened for binary
        #   reading or writing. This is necessary for np.save() and np.Load()
        FeatureSet.__init__(self, verbose=True, useBinaryFileMode=True)

    # 4. Implement the _extract() method by extracting features from a dataset and optionally
    #   returning the extracted features
    def _extract(self, outputFile):
        X = lda.datasets.load_reuters()
        model = lda.LDA(n_topics=20, n_iter=10, random_state=1)
        model.fit(X)
        
        np.save(outputFile, model.doc_topic_)

        # this is optional, but it provides a good optimization :)
        return model.doc_topic_

    # 5. Implement the _load() method by loading the features from outputFile
    def _load(self, outputFile):
        return np.load(outputFile)

# 6. Instantiate your feature set object and call smartLoad() with an output file path
#   Note: ideally you should do this in a separate script that imports this script, otherwise
#   you won't achieve the benefits of smartLoad() when you start tweaking your model
featureSet = LDADocTopicFeatureSet()
X = featureSet.smartLoad('data_cache/LDA_features.npa')
print X.shape

# 7. Now you can tweak your model without having to reextract your features!!
