package ru.ml.rf.datasets;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TestTrain {

    public final DataSet test;
    public final DataSet train;

    public TestTrain(DataSet input, int splitSize, Random rng) {
        List<Instance> train = new ArrayList<>();
        List<Instance> test = new ArrayList<>();

        for (int i = 0; i < input.getSize(); i++) {
            Instance sample = input.getInstance(i);
            if (rng.nextDouble() < (splitSize - train.size()) / (double) (input.getSize() - i)) {
                train.add(sample);
            } else {
                test.add(sample);
            }
        }

        this.train = new DataSet(train);
        this.test = new DataSet(test);
    }

}
