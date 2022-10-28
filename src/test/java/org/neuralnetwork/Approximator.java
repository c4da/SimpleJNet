package org.neuralnetwork;

import matrix.Matrix;

import java.util.function.Function;

public class Approximator {

    public static Matrix gradient(Matrix input, Function<Matrix, Matrix> transform){

        input.forEach((row, col, index, value) -> {
            System.out.printf("%12.5f", value);

            if (col == input.getCols() - 1){
                System.out.println();
            }
        });

        return null;
    }
}
