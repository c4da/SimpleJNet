package org.neuralnetwork;

import junit.framework.TestCase;
import matrix.Matrix;
import org.junit.Test;

import java.util.Random;

public class NeuralNetTest extends TestCase {

    private Random random = new Random();

    @Test
    public void testApproximator(){
        final int rows = 4;
        final int cols = 5;
        Matrix input = new Matrix(rows, cols, i-> random.nextGaussian());
        Matrix expected = new Matrix(rows, cols, i -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);
            expected.set(randomRow, col, 1);
        }
        Approximator.gradient(input, in->{
            return LossFunction.crossEntropy(expected, in);
        });

        System.out.println();
        System.out.println(input);

        System.out.println(expected);
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

        Matrix result = LossFunction.crossEntropy(expected, actual);

        actual.forEach((row, col, index, value) -> {
            double expectedValue = expected.get(index);
            double loss = result.get(col);
            if (expectedValue > 0.9){
                assertTrue(Math.abs(Math.log(value) + loss) < 0.001d);
            }

        });
    }

//    @Test
    public void testEngine(){
        Engine engine = new Engine();

        engine.add(Transform.DENSE, 8, 5);
        engine.add(Transform.RELU);

        engine.add(Transform.DENSE, 5);
        engine.add(Transform.RELU);

        engine.add(Transform.DENSE, 4);
        engine.add(Transform.SOFTMAX);
        
        Matrix input = new Matrix(5, 1, i->random.nextGaussian());
        Matrix output = engine.runForwards(input);

        System.out.println(engine);
        System.out.println(output);
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
        Matrix result2 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row))
                .modify(value -> value > 0 ? value : 0);

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