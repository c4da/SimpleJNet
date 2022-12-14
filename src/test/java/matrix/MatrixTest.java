package matrix;

import org.junit.Test;
import org.neuralnetwork.matrix.Matrix;

import java.util.Random;

import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

public class MatrixTest {
    private Random random = new Random();

    @Test
    public void testGetGreatestRowNumber(){
        double[] values = {3, -6, 2, 7, 2, 6, 3, 11, -9, 1};
        Matrix m = new Matrix(3, 3, i->values[i]);

        double[] expectedValues = {1, 2, 1};
        Matrix expected = new Matrix(1, 3, i -> expectedValues[i]);
        Matrix result = m.getGreatestRowNumbers();
        assertTrue(expected.equals(result));

        System.out.println(m);
        System.out.println(result);
    }

    @Test
    public void testAverageColumn(){
        int rows = 7;
        int cols = 5;

        Matrix m = new Matrix(rows, cols, i -> 2 * i - 3);

        double averageIndex = (cols - 1)/2.0;

        Matrix expected = new Matrix(rows, 1);
        expected.modify((row, col, value) -> 2 * (row * cols + averageIndex) - 3);

        Matrix result = m.averageColumn();

        System.out.println(result);
        System.out.println(m);
        System.out.println(expected);

        assertTrue(expected.equals(result));
    }

    @Test
    public void testTranspose(){
        Matrix m = new Matrix(2, 3, i -> i);
        Matrix result = m.transpose();
        double[] expectedValues = {0, 3, 1, 4, 2, 5};
        Matrix expected = new Matrix(3, 2, i -> expectedValues[i]);
        assertTrue(expected.equals(result));

    }

    @Test
    public void testSetValue(){
        int rows = 8;
        int cols = 8;
        Matrix m = new Matrix(rows, cols, i -> 1);
        System.out.println(m);

        for (int i = 0; i < rows; i++) {
            m.set(i, 0, 0);
        }

        System.out.println(m);
        for (int i = 0; i < rows; i++) {
            int col = 0;
            assertTrue(m.get(i * cols + col) == 0);
        }
    }

    @Test
    public void testSoftMax(){
        Matrix m = new Matrix(5, 8, i -> random.nextGaussian());
        Matrix result = m.softMax();
        System.out.println(result);

        double[] colSums = new double[8];
        result.forEach((row, col, value)->{
            assertTrue(value >= 0 && value <= 1.0);
            colSums[col] += value;
        });

        for (double colSum : colSums) {
            assertTrue(Math.abs(colSum - 1.0) < 1e-5);
        }
    }

    @Test
    public void testSumColumns(){
        Matrix m = new Matrix(4, 5, i -> i);
        Matrix result = m.sumColumns();

        double[] expectedValues = {+30.00000,    +34.00000,    +38.00000,   +42.00000,  +46.00000};
        Matrix expected = new Matrix(1, 5, i->expectedValues[i]);

        assertTrue(expected.equals(result));
    }

    @Test
    public void testMultiply(){
        Matrix m1 = new Matrix(2, 3, i -> i);
        Matrix m2 = new Matrix(3, 2, i -> i);

        double[] expectedValues = {10, 13, 28, 40};
        Matrix expected = new Matrix(2, 2, i -> expectedValues[i]);

        Matrix result = m1.multiply(m2);
        assertTrue(expected.equals(result));
    }

    @Test
    public void testMultiplySpeed(){
        int rows = 500;
        int cols = 500;
        int mid = 50;

        Matrix m1 = new Matrix(rows, mid, i -> i);
        Matrix m2 = new Matrix(mid, cols, i -> i);

        var start = System.currentTimeMillis();
        var iterations = 100;
        for (int i = 0; i < iterations; i++) {
            m1.multiply(m2);
        }
        var end = System.currentTimeMillis();

        System.out.printf("Matrix multiplication time taken: %dms\n", end - start);
    }

    @Test
    public void testEquals(){
        Matrix m1 = new Matrix(3, 4, i -> 2 * (i - 6));
        Matrix m2 = new Matrix(3, 4, i -> 2 * (i - 6));
        Matrix m3 = new Matrix(3, 4, i -> 2 * (i - 6.2));

        assertTrue(m1.equals(m2));
        assertFalse(m1.equals(m3));
    }

    @Test
    public void testAddMatrices(){
        Matrix m1 = new Matrix(2, 2, i -> i);
        Matrix m2 = new Matrix(2, 2, i -> i * 1.5);
        Matrix expected = new Matrix(2, 2, i -> i * 2.5);

        Matrix result = m1.apply((index, value) -> value + m2.get(index));
        assertTrue(expected.equals(result));
    }

    @Test
    public void testMultiplyDouble(){
        Matrix m = new Matrix(3, 4, i -> 0.5 * (i - 6));
        double x = 0.5;

        Matrix expected = new Matrix(3, 4, i -> x * 0.5 * (i - 6));
        Matrix result = m.apply((index, value) -> x * value);

        System.out.println(result);
        System.out.println(expected);

        assertTrue(result.equals(expected));
        assertTrue(Math.abs(result.get(1) + 1.25) < 1e-4);
    }

    @Test
    public void testToString(){
        Matrix m = new Matrix(3, 4, i->i*2);

        String text = m.toString();
        double[] expected = new double[3*4];

        for (int i = 0; i < expected.length; i++) {
            expected[i] = i * 2;
        }
        var index = 0;
        var rows = text.split("\n");

        assertTrue(rows.length == 3);

        for (String row : rows) {
            var values = row.split("\\s+");

            for (var textValue:values){
                if (textValue.length() == 0){
                    continue;
                }
                textValue = textValue.replace(",", ".");
                var doubleValue = Double.valueOf(textValue);
                assertTrue(Math.abs(doubleValue - expected[index]) < 0.0001);
                ++index;
            }

        }
    }
}