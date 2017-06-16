package ru.ml.rf.classifier;

import ru.ml.rf.datasets.DataSet;
import ru.ml.rf.datasets.Instance;
import ru.ml.rf.nodes.DecisionNode;
import ru.ml.rf.nodes.LeafNode;
import ru.ml.rf.nodes.TreeNode;
import ru.ml.rf.utils.EntropyUtils;
import ru.ml.rf.utils.SamplerUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;


public class DecisionTree {

    // root of the decision tree. 
    private TreeNode root;

    // for each node, we randomly select subset of features to consider for splitting
    // by default, we set the size as square root of total number of features.  
    private int maxFeatures;

    private int numLabels = 3;

    // minimum size of subtree, this value can be used as condition for termination.
    // by default, we set the size as 10.
    private int minSamplesLeaf = 10;

    DecisionTree() {
    }

    void buildTree(DataSet dataset, int maxFeatures, int minSamplesLeaf) {
        this.maxFeatures = maxFeatures;
        this.minSamplesLeaf = minSamplesLeaf;
        root = build(dataset);
    }

    double[] predictDist(double[] featureVector) {
        return predict(root, featureVector);
    }

    public int predictLabel(double[] featureVector) {
        double[] dist = predict(root, featureVector);
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

    private double[] predict(TreeNode node, double[] featureVector) {
        // if leaf node, then just return the distribution. 
        if (node instanceof LeafNode)
            return ((LeafNode) node).getLabelProbDist();

        // if current node is decision node, then go to left child or right child. 
        int featureIdx = ((DecisionNode) node).getSplittingFeatureIndex();
        double splittingValue = ((DecisionNode) node).getSplittingValue();
        double value = featureVector[featureIdx];
        if (value < splittingValue)
            return predict(((DecisionNode) node).getLeftChild(), featureVector);
        else
            return predict(((DecisionNode) node).getRightChild(), featureVector);
    }

    private TreeNode build(DataSet dataset) {
        // create a new leaf node if either of condition met. 
        if (dataset.getNumOfInstance() < minSamplesLeaf || hasSameLabel(dataset.getLabels())) {
            return new LeafNode(numLabels, dataset.getLabels());
        }

        // sub-sample the attributes. 
        int[] selectedFeatureIndexes = SamplerUtils.randSample(dataset.getNumOfFeatures(), maxFeatures);

        // select the best feature based on information gain
        int bestFeatureIndex = getBestFeatureIndex(selectedFeatureIndexes, dataset);

        // for numerical attribute, we  create left and right child. 
        return createDecisionNode(bestFeatureIndex, dataset);
    }


    private int getBestFeatureIndex(int[] candidateFeatureIndex, DataSet dataset) {
        double maxInfoGain = Double.MIN_VALUE;
        int bestFeatureIndex = 0;
        double entropy = EntropyUtils.getEntropy(dataset.getLabels());

        for (int i = 0; i < candidateFeatureIndex.length; i++) {
            int featureIndex = candidateFeatureIndex[i];
            double infoGain = getInformationGain(entropy, featureIndex, dataset);
            if (infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                bestFeatureIndex = i;
            }
        }
        return bestFeatureIndex;
    }


    private double getInformationGain(double entropy, int featureIndex, DataSet dataset) {
        int dataSize = dataset.getNumOfInstance();
        // get the mean
        double mean = 0;
        for (int i = 0; i < dataset.getNumOfInstance(); i++) {
            double[] featureVector = dataset.getInstance(i).getFeatureVector();
            mean += featureVector[featureIndex] / dataSize;
        }

        // divide the dataset into two subset, based on the mean value. 
        int leftSize = 0;
        for (int i = 0; i < dataSize; i++) {
            if ((dataset.getInstance(i).getFeatureVector())[featureIndex] < mean)
                leftSize++;
        }
        int rightSize = dataSize - leftSize;

        List<Integer> leftLabels = new ArrayList<>(leftSize);
        List<Integer> rightLabels = new ArrayList<>(rightSize);

        for (int i = 0; i < dataSize; i++) {
            if (dataset.getInstance(i).getFeatureVector()[featureIndex] < mean)
                leftLabels.add(dataset.getInstance(i).getLabelIndex());
            else
                rightLabels.add(dataset.getInstance(i).getLabelIndex());
        }

        double leftEntropy = EntropyUtils.getEntropy(leftLabels);
        double rightEntropy = EntropyUtils.getEntropy(rightLabels);

        return entropy - (leftSize * 1.0 / dataSize) * leftEntropy
                - (rightSize * 1.0 / dataSize) * rightEntropy;
    }

    private TreeNode createDecisionNode(int bestFeatureIdx, DataSet dataset) {
        // calculate the mean. 
        double mean = 0;
        int dataSize = dataset.getNumOfInstance();
        for (int i = 0; i < dataSize; i++) {
            double[] featureVector = dataset.getInstance(i).getFeatureVector();
            mean += featureVector[bestFeatureIdx] / dataSize;
        }

        List<Instance> leftDataSet = new ArrayList<>();
        List<Instance> rightDataSet = new ArrayList<>();
        // divide the datasets into two subset, based on the mean value. 
        for (int i = 0; i < dataSize; i++) {
            // smaller one goes to left. 
            if ((dataset.getInstance(i).getFeatureVector())[bestFeatureIdx] < mean)
                leftDataSet.add(dataset.getInstance(i));
            else
                rightDataSet.add(dataset.getInstance(i));
        }

        // create new decision node, and set the left child and right child. 
        TreeNode node = new DecisionNode(bestFeatureIdx, mean);
        if (leftDataSet.size() > 0) {
            ((DecisionNode) node).setLeftChild(build(new DataSet(leftDataSet)));
        } else {
            // create leaf node, with majority distribution. 
            TreeNode leafNode = new LeafNode(numLabels, dataset.getLabels());
            ((DecisionNode) node).setLeftChild(leafNode);
        }

        if (rightDataSet.size() > 0) {
            ((DecisionNode) node).setRightChild(build(new DataSet(rightDataSet)));
        } else {
            // create leaf node, with majority distribution. 
            TreeNode leafNode = new LeafNode(numLabels, dataset.getLabels());
            ((DecisionNode) node).setRightChild(leafNode);
        }

        return node;
    }

    private boolean hasSameLabel(List<Integer> labels) {
        for (int i = 1; i < labels.size(); i++) {
            if (!Objects.equals(labels.get(i), labels.get(i - 1)))
                return false;
        }
        return true;
    }

    public void setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
    }

    public void setTreeMinSize(int minTreeSize) {
        this.minSamplesLeaf = minTreeSize;
    }
}
