package org.neuralnetwork;



import matrix.Matrix;

import java.util.LinkedList;

public class BatchResult {
    private LinkedList<Matrix> io = new LinkedList<>();
    private LinkedList<Matrix> weightErrors = new LinkedList<>();
    private LinkedList<Matrix> weightInputs = new LinkedList<>();
    private double loss;

    private double percentCorrect;
    public LinkedList<Matrix> getWeightErrors() {
        return weightErrors;
    }

    public void addWeightInput(Matrix input){
        weightInputs.add(input);
    }

    public LinkedList<Matrix> getWeightInputs(){
        return weightInputs;
    }

    public void addWeightErrors(Matrix weightError) {
        this.weightErrors.addFirst(weightError);
    }

    public Matrix getInputError() {
        return inputError;
    }

    public Matrix getOutput(){
        return io.getLast();
    }

    public void setInputError(Matrix inputError) {
        this.inputError = inputError;
    }

    private Matrix inputError;

    public LinkedList<Matrix> getIo(){
        return io;
    }

    public void addIo(Matrix m){
        io.add(m);
    }

    public void setLoss(double loss) {
        this.loss = loss;
    }

    public double getLoss() {
        return loss;
    }

    public void setPercentCorrect(double percentCorrect) {
        this.percentCorrect = percentCorrect;
    }

    public double getPercentCorrect(){
        return percentCorrect;
    }
}
