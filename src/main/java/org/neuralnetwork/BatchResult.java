package org.neuralnetwork;



import matrix.Matrix;

import java.util.LinkedList;

public class BatchResult {
    private LinkedList<Matrix> io = new LinkedList<>();
    private LinkedList<Matrix> weightErrors = new LinkedList<>();

    public LinkedList<Matrix> getWeightErrors() {
        return weightErrors;
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
}
