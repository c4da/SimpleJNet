package org.neuralnetwork;

import org.neuralnetwork.matrix.Matrix;

public class TrainingArrays {
    private double[] input;

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(double[] output) {
        this.output = output;
    }

    private double[] output;

    public TrainingArrays(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }


}
