package ru.ml.rf.classifier;

import ru.ml.rf.datasets.DataSet;
import ru.ml.rf.datasets.Instance;
import ru.ml.rf.utils.SamplerUtils;

import java.util.ArrayList;
import java.util.List;

public class RandomForest {

    private int numTrees;
    private List<DecisionTree> decisionTrees;

    // size of sampling for each bootstrap step. 
    private int maxFeatures;

    private int minSamplesLeaf;

    // minimum number of samples for each node. If reached the minimum, we just make it as
    // leaf node without further splitting. 
    public static final int TREE_MIN_SIZE = 1;

    private DataSet dataset;

    public RandomForest(DataSet dataset, int numTrees, int maxFeatures, int minSamplesLeaf) {
        this.dataset = dataset;
        this.numTrees = numTrees;
        this.maxFeatures = maxFeatures;
        this.minSamplesLeaf = minSamplesLeaf;
        decisionTrees = new ArrayList<>(numTrees);
        createRandomForest();
    }

    public void createRandomForest() {
        for (int i = 0; i < numTrees; i++) {
            DecisionTree dt = new DecisionTree();
            dt.buildTree(getBootStrapData(), maxFeatures, minSamplesLeaf);

            decisionTrees.add(dt);
        }
    }


    /**
     * Get the predicted label for given feature vector.
     *
     * @param featureVector the input feature vector.
     * @return predicted label.
     */
    public int predictLabel(double[] featureVector) {
        double[] dist = predictDist(featureVector);
        int maxLabel = 0;
        double maxProb = 0;
        for (int i = 0; i < dist.length; i++) {
            if (dist[i] > maxProb) {
                maxProb = dist[i];
                maxLabel = i;
            }
        }
        return maxLabel;
    }

    /**
     * Get the prediction for the input feature vector.  Basically, it iterate
     * through each decision tree, and get prediction from each of them. Then aggregate
     * those predictions.
     *
     * @param featureVector the query input feature vector.
     * @return prediction, which is probability distribution for different
     * labels.
     */
    public double[] predictDist(double[] featureVector) {
        int totalNumLabels = 3;
        double[] finalPredict = new double[totalNumLabels];
        // iterate through each decision tree, and make prediction. 
        for (int i = 0; i < numTrees; i++) {
            double[] predict = decisionTrees.get(i).predictDist(featureVector);
            for (int j = 0; j < totalNumLabels; j++) {
                finalPredict[j] += predict[j];
            }
        }

        for (int i = 0; i < totalNumLabels; i++) {
            finalPredict[i] = finalPredict[i] / numTrees;
        }

        return finalPredict;
    }


    /**
     * Get bootstrap dataset, with replacement.
     */
    private DataSet getBootStrapData() {
        int[] indexs = SamplerUtils.bootStrap(dataset.getNumOfInstance());
        List<Instance> bootStrapSamples = new ArrayList<Instance>();
        for (int i = 0; i < indexs.length; i++) {
            bootStrapSamples.add(dataset.getInstance(indexs[i]));
        }

        return new DataSet(bootStrapSamples);
    }
}
