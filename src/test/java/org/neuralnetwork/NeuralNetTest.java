package org.neuralnetwork;

import org.neuralnetwork.matrix.Matrix;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class NeuralNetTest {

    @Test
    public void testTrainEngine(){
        var inputRows = 500;
        var cols = 32;
        var outputRows = 3;

        Engine engine = new Engine();
        engine.add(Transform.DENSE, 100, inputRows);
        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, outputRows);
        engine.add(Transform.SOFTMAX);

        RunningAverages runningAverages = new RunningAverages(2, 2, ((callNumber, averages) -> {
//            assertTrue(averages[0] < 6);
            System.out.printf("%d. Loss: %.3f -- Percent correct: %.2f\n", callNumber, averages[0], averages[1]);
        }));

//        System.exit(0);

        double initialLearningRate = 0.02;
        double learningRate = initialLearningRate;
        double iterations = 2000;

        for (int i = 0; i < iterations; i++) {
            var trainingMatrixes = Util.generateTrainingMatrixes(inputRows, outputRows, cols);
            var input = trainingMatrixes.getInput();
            var expected =trainingMatrixes.getOutput();

            BatchResult batchResult = engine.runForwards(input);
            engine.runBackwards(batchResult, expected);
            engine.adjust(batchResult, learningRate);
            engine.evaluate(batchResult, expected);

            runningAverages.add(batchResult.getLoss(), batchResult.getPercentCorrect());

            double learningRateDecrease = initialLearningRate / iterations;

            learningRate -= learningRateDecrease;
        }

    }

    private Random random = new Random();

    @Test
    public void testWeightGradient(){

        int inputRows = 4;
        int outputRows = 5;

        Matrix input = Util.generateInputMatrix(inputRows, 1);
        Matrix weights = new Matrix(outputRows, inputRows, i -> random.nextGaussian());
        Matrix expected = Util.generateExpectedMatrix(outputRows, 1);

        Matrix output = weights.multiply(input).softMax();
//        Matrix loss = LossFunctions.crossEntropy(expected, output);

        Matrix calculatedError = output.apply((index, value) -> value - expected.get(index));

        Matrix calculatedWeightGradients = calculatedError.multiply(input.transpose());

        Matrix approximatedWeightGradients = Approximator.weightGradient(
                weights,
                w -> {
                    Matrix out = w.multiply(input).softMax();
                    return LossFunctions.crossEntropy(expected, out);
                });

        calculatedWeightGradients.setTolerance(1e-2);
        assertTrue(calculatedWeightGradients.equals(approximatedWeightGradients));
    }

    @Test
    public void testEngine(){

        var inputRows = 5;
        var cols = 6;
        var outputRows = 5;

        Engine engine = new Engine();

        engine.add(Transform.DENSE, 8, 5);
//        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, 5);
//        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, 4);

        engine.add(Transform.SOFTMAX);
        engine.setStoreInputError(true);

        Matrix input = Util.generateInputMatrix(inputRows, cols);
        Matrix expected = Util.generateExpectedMatrix(outputRows, cols);

        Matrix approximatedError = Approximator.gradient(input, in -> {
            BatchResult output = engine.runForwards(in);
            return LossFunctions.crossEntropy(expected, output.getOutput());
        });

        BatchResult batchResult = engine.runForwards(input);
        engine.runBackwards( batchResult, expected);

        Matrix calculatedError = batchResult.getInputError();

        System.out.println(calculatedError);
        System.out.println(approximatedError);

        calculatedError.setTolerance(1e-3);
        assertTrue(calculatedError.equals(approximatedError));
    }

    @Test
    public void testBackprop(){

        interface NeuralNet {
            Matrix apply(Matrix m);
        }

        final int inputRows = 4;
        final int cols = 5;
        final int outputRows = 4;

        Matrix input = new Matrix(inputRows, cols, i-> random.nextGaussian()).softMax();
        Matrix expected = new Matrix(outputRows, cols, i -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(outputRows);
            expected.set(randomRow, col, 1);
        }

        Matrix weights = new Matrix(outputRows, inputRows, i -> random.nextGaussian());
        Matrix biases = new Matrix(outputRows, 1, i -> random.nextGaussian());

        NeuralNet neuralNet = m -> {
            Matrix out = m.apply((index, value) -> value > 0 ? value : 0);
            out = weights.multiply(out); //apply weights
            out = out.modify((row, col, value)->value+biases.get(row)); //add biases
            out = out.softMax(); //apply softmax
            return out;
        };

        //Little neural network with only one layer - forward pass
        Matrix softMaxOutput = neuralNet.apply(input);

        Matrix approximatedResult = Approximator.gradient(input, in -> {
            Matrix out = neuralNet.apply(in);
            return LossFunctions.crossEntropy(expected, out);
        });

        Matrix calculatedResult = softMaxOutput.apply((index, value) -> value - expected.get(index));

        //backwards pass - backpropagation
        calculatedResult = weights.transpose().multiply(calculatedResult);
        calculatedResult = calculatedResult.apply((index, value) -> input.get(index) > 0 ? value : 0);

        assertTrue(approximatedResult.equals(calculatedResult));

    }

    @Test
    public void testSoftMaxCrossEntropyGradient(){
        final int rows = 4;
        final int cols = 5;
        Matrix input = new Matrix(rows, cols, i-> random.nextGaussian()).softMax();
        Matrix expected = new Matrix(rows, cols, i -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);
            expected.set(randomRow, col, 1);
        }

        Matrix softMaxOutput = input.softMax();


        Matrix result = Approximator.gradient(input, in -> {
            return LossFunctions.crossEntropy(expected, in.softMax());
        });

        result.forEach((index, value) -> {
            double softmaxValue = softMaxOutput.get(index);
            double expectedValue = expected.get(index);
            assertTrue(Math.abs(value - (softmaxValue - expectedValue)) < 0.01);
        });
    }

    @Test
    public void testAddIncrement(){
        Matrix m = new Matrix(5, 8, i -> random.nextGaussian());

        int row = 3;
        int col = 2;
        double inc = 10;

        Matrix result = m.addIncrement(row, col, inc);

        double originalValue = m.get(row, col);
        double incrementedValue = result.get(row, col);
        assertTrue(Math.abs(incrementedValue - (originalValue + inc)) < 0.001);

        System.out.println(m);
        System.out.println(result);
    }

    @Test
    public void testApproximator(){
        final int rows = 4;
        final int cols = 5;
        Matrix input = new Matrix(rows, cols, i-> random.nextGaussian()).softMax();
        Matrix expected = new Matrix(rows, cols, i -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);
            expected.set(randomRow, col, 1);
        }

        Matrix result = Approximator.gradient(input, in -> {
            return LossFunctions.crossEntropy(expected, in);
        });

        input.forEach((index, value) ->{
            double resultValue = result.get(index);
            double expectedValue = expected.get(index);

            if (expectedValue < 1e-3){
                assertTrue(Math.abs(resultValue) < 1e-2);
            } else {
                assertTrue(Math.abs(resultValue + 1.0 / value) < 1e-2);
            }
        });

        System.out.println(result);
    }

    @Test
    public void testTemp(){

        int inputSize = 5;
        int layer1Size = 6;
        int layer2Size = 4;

        Matrix input = new Matrix(inputSize, 1, i-> random.nextGaussian());

        Matrix layer1Weights = new Matrix(inputSize, inputSize, i-> random.nextGaussian());
        Matrix layer1Biases = new Matrix(layer1Size, 1, i-> random.nextGaussian());

        Matrix layer2Weights = new Matrix(layer2Size, layer1Weights.getRows(), i-> random.nextGaussian());
        Matrix layer2Biases = new Matrix(layer2Size, 1, i-> random.nextGaussian());

        var output = input;
        System.out.println(output);

        output = layer1Weights.multiply(output);
        System.out.println(output);

        output = output.modify((row, col, value) -> value + layer1Biases.get(row));
        System.out.println(output);

        output = output.modify(value -> value > 0 ? value : 0);
        System.out.println(output);

        output = layer2Weights.multiply(output);
        System.out.println(output);

        output = output.modify((row, col, value) -> value + layer2Biases.get(row));
        System.out.println(output);

        output = output.softMax();
        System.out.println(output);

    }

    @Test
    public void testCrossEntropy(){
        double[] expectedValues = {1, 0, 0, 0, 0, 1, 0, 1, 0};
        Matrix expected = new Matrix(3, 3, i -> expectedValues[i]);

        Matrix actual = new Matrix(3, 3, i -> (0.05 * i * i)).softMax();

        Matrix result = LossFunctions.crossEntropy(expected, actual);

        actual.forEach((row, col, index, value) -> {
            double expectedValue = expected.get(index);
            double loss = result.get(col);
            if (expectedValue > 0.9){
                assertTrue(Math.abs(Math.log(value) + loss) < 0.001d);
            }

        });
    }

    @Test
    public void testAddBias(){

        Matrix input = new Matrix(3, 5, i -> (i + 1));
        Matrix weights = new Matrix(3, 3, i -> (i + 1));
        Matrix biases = new Matrix(3, 1, i -> (i + 1));

        Matrix result = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));

        double[] expectedValues = {31, 37, 43, 68, 83, 98, 105, 129, 153};

        Matrix expected = new Matrix(3, 3, i -> expectedValues[i]);
        assertTrue(expected.equals(result));

    }
    @Test
    public void testRELU(){

        final int numberNeurons = 5;
        final int numberInputs = 6;
        final int inputSize = 4;

        Matrix input = new Matrix(inputSize, numberInputs, i -> random.nextDouble());
        Matrix weights = new Matrix(numberNeurons, inputSize, i -> random.nextGaussian());
        Matrix biases = new Matrix(numberNeurons, 1, i -> random.nextGaussian());

        Matrix result1 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));
        Matrix result2 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row)).modify(value -> value > 0 ? value : 0);

        result2.forEach((index, value)->{
            double originalValue = result1.get(index);
            if (originalValue > 0){
                assertTrue(Math.abs(originalValue - value) < 1e-4);
            } else {
                assertTrue(Math.abs(value) < 1e-4);
            }
//            System.out.println(index + ", " + value);
        });


    }
}