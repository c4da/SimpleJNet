package org.neuralnetwork;

import matrix.Matrix;

public class LossFunction {
    public static Matrix crossEntropy(Matrix expected, Matrix actual) {
        Matrix result = actual.apply((index, value) -> {
            return -expected.get(index) * Math.log(value);
        }).sumColumns();

        return result;
    }
}
