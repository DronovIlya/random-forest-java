package ru.ml.rf.utils;

import java.util.Random;

public class SamplerUtils {

    public static int[] randSample(int n, int m) {
        int[] indexes = new int[n];
        for (int i = 0; i < n; i++)
            indexes[i] = i;

        for (int i = 0; i < m; i++) {
            int randomNum = new Random().nextInt(n - i) + i;
            // swatch i and randomNum
            int tmp = indexes[i];
            indexes[i] = indexes[randomNum];
            indexes[randomNum] = tmp;
        }

        int[] result = new int[m];
        for (int i = 0; i < m; i++)
            result[i] = indexes[i];

        return result;
    }

    /**
     * bootstrap index of samples, with replacement.
     */
    public static int[] bootStrap(int n) {
        int[] bootstrapIndex = new int[n];
        for (int i = 0; i < n; i++) {
            bootstrapIndex[i] = new Random().nextInt(n);
        }

        return bootstrapIndex;
    }
}
