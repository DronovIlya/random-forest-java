package ru.ml.rf.utils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EntropyUtils {

    /*
     * calculate the entropy of response variables. 
     * H(X) = -\sum_{i=1}^{d} p(x_{i})log2(p(x_{i}))
     *
     */
    public static double getEntropy(List<Integer> labels) {
        int size = labels.size();
        Map<Integer, Integer> countMap = new HashMap<>();

        for (int i = 0; i < labels.size(); i++) {
            if (!countMap.containsKey(labels.get(i))) {
                countMap.put(labels.get(i), 1);
            } else {
                int currCnt = countMap.get(labels.get(i));
                countMap.put(labels.get(i), currCnt + 1);
            }
        }

        double entropy = 0;
        for (Integer label : countMap.keySet()) {
            double p = countMap.get(label) * 1.0 / size;
            entropy += p * log(p, 2);
        }

        return -1 * entropy;
    }

    private static double log(double x, int base) {
        return Math.log(x) / Math.log(base);
    }
}
