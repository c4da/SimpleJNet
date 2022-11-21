package org.neuralnetwork.App;

import org.neuralnetwork.BatchResult;
import org.neuralnetwork.Engine;
import org.neuralnetwork.Transform;
import org.neuralnetwork.loader.BatchData;
import org.neuralnetwork.loader.Loader;
import org.neuralnetwork.loader.MetaData;
import org.neuralnetwork.loader.test.TestLoader;
import org.neuralnetwork.matrix.Matrix;

public class NeuralNetwork {

    private Engine engine;
    private int epochs;
    private double initialLearningRate = 0.01;
    private double finalLearningRate = 0;
    private double learningRate = 0;
    private Object lock = new Object();

    public NeuralNetwork(){
        engine = new Engine();
    }

    public void add(Transform transform, double... params){
        engine.add(transform, params);
    }

    public void setLearningRate(double initialLearningRate, double finalLearningRate){
        this.initialLearningRate = initialLearningRate;
        this.finalLearningRate = finalLearningRate;
    }

    public void setEpochs(int epochs){
        this.epochs = epochs;
    }

    public void fit(Loader trainLoader, Loader evalLoader){
        learningRate = initialLearningRate;
        for (int epoch = 0; epoch < epochs; epoch++) {

            System.out.printf("Epoch %3d ", epoch);

            runEpoch(trainLoader, true);
            if (evalLoader != null){
                runEpoch(evalLoader, false);
            }

            learningRate -= (initialLearningRate - finalLearningRate) / epochs;
        }
    }

    private void runEpoch(Loader loader, boolean trainingMode) {
        loader.open();

        var queue = createBatchTask(loader, trainingMode);
        consumeBatchTask(queue, trainingMode);

        loader.close();
    }
    private void consumeBatchTask(Loader loader, boolean trainingMode) {
    }

    private Object createBatchTask(Loader loader, boolean trainingMode) {
        MetaData metaData = loader.getMetaData();
        int numberBatches = metaData.getNumberBatches();

        for (int i = 0; i < numberBatches; i++) {
            runBatch(loader, trainingMode);
        }

        return null;
    }

    private BatchResult runBatch(Loader loader, boolean trainingMode) {
        int batchSize = 33;

        MetaData metaData = loader.open();
        int numberItems = metaData.getNumberItems();

        BatchData batchData = loader.readBatch();

        int itemsRead = metaData.getItemsRead();
        int inputSize = metaData.getInputSize();
        int expectedSize = metaData.getExpectedSize();

        Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
        Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

        BatchResult batchResult = engine.runForwards(input);

        if (trainingMode){
            engine.runBackwards(batchResult, expected);
            synchronized (lock) {
                engine.adjust(batchResult, learningRate);
            }
        } else {
            engine.evaluate(batchResult, expected);
        }

        return batchResult;

    }

    @Override
    public String toString() {
        return engine.toString();
    }
}
