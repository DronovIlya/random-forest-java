package ru.ml.rf.datasets;

import java.util.Arrays;

public class Instance {

    private final double[] featureVector;
    private final int labelIndex;

    Instance(double[] featureVector, int labelIndex) {
        this.featureVector = featureVector;
        this.labelIndex = labelIndex;
    }

    public double[] getFeatureVector() {
        return this.featureVector;
    }

    public int getLabelIndex() {
        return labelIndex;
    }

    @Override
    public String toString() {
        return "Instance{" +
                "featureVector=" + Arrays.toString(featureVector) +
                ", labelIndex=" + labelIndex +
                '}';
    }
}
