package ru.ml.rf.tasks;

import ru.ml.rf.classifier.RandomForest;
import ru.ml.rf.datasets.DataSet;
import ru.ml.rf.datasets.Instance;
import ru.ml.rf.datasets.TestTrain;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class IrisTask {

    private static final String TRAIN_DATA = "comparison/data_prepared/spine.csv";

    public static void main(String[] args) throws IOException {
        IrisTask irisTask = new IrisTask();

        Pair<List<double[]>, List<Integer>> allData = irisTask.getData(TRAIN_DATA);

//        List<String> uniqueLabels = new ArrayList<>(new HashSet<>(allData.second));
//        List<Integer> labels = allData.second.stream().map(uniqueLabels::indexOf).collect(Collectors.toList());

        DataSet all = new DataSet(allData.first, allData.second);
        TestTrain testTrain = new TestTrain(all, (int) (all.getSize() * 0.7), new Random(all.getSize()));

        System.out.println(testTrain.train.getSize() + ", test = " + testTrain.test.getSize());

        RandomForest rf = new RandomForest(new DataSet(testTrain.train.getTrainingData()), 100, 10);
        DataSet test = testTrain.test;

        int correct = 0;
        int total = 0;
        for (int i = 0; i < test.getSize(); i++){
            Instance sample = test.getInstance(i);
            int predictLabel = rf.predictLabel(sample.getFeatureVector());
            if (sample.getLabelIndex() == predictLabel) {
                correct++;
            }
            total++;
        }

        System.out.println("accuracy = " + ((double) correct / total));
    }

    private Pair<List<double[]>, List<Integer>> getData(String path) throws IOException {
        System.out.println("read dataset from " + path);

        try (BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path)))) {
            int count = -1;
            List<double[]> samples = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            String input;
            while ((input = in.readLine()) != null) {
                count++;
                if (count == 0) {
                    continue;
                }

                String[] features = input.split(",");
                double[] sample = new double[features.length - 1];
                for (int i = 0; i < features.length - 1; i++){
                    sample[i] = Double.parseDouble(features[i]);
                }
                samples.add(sample);

                labels.add(Integer.parseInt(features[features.length - 1]));
            }
            return new Pair<>(samples, labels);
        }
    }

    private static class Pair<K, V> {
        K first;
        V second;

        public Pair(K first, V second) {
            this.first = first;
            this.second = second;
        }

        public K getFirst() {
            return first;
        }

        public V getSecond() {
            return second;
        }
    }

}
