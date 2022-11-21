package org.neuralnetwork.loader.test;

import junit.framework.TestCase;
import org.junit.Test;
import org.neuralnetwork.loader.BatchData;
import org.neuralnetwork.loader.Loader;
import org.neuralnetwork.loader.MetaData;
import org.neuralnetwork.matrix.Matrix;

import static org.junit.Assert.assertTrue;

public class TestLoaderTest {

    @Test
    public void test(){
        int batchSize = 33;
        Loader testLoader = new TestLoader(60_000, batchSize);

        MetaData metaData = testLoader.open();
        int numberItems = metaData.getNumberItems();
        int lastBatchSize = numberItems % batchSize;
        int numberBatches = metaData.getNumberBatches();

        for (int i = 0; i < numberBatches; i++) {
            BatchData batchData = testLoader.readBatch();

            assertTrue(batchData != null);

            int itemsRead = metaData.getItemsRead();
            int inputSize = metaData.getInputSize();
            int expectedSize = metaData.getExpectedSize();

            Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
            Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

            assertTrue(input.sum() != 0);
            assertTrue(expected.sum() == itemsRead);

            if (i == numberBatches - 1 && lastBatchSize != 0){
                assertTrue(itemsRead == lastBatchSize);
            } else {
                assertTrue(itemsRead == batchSize);
            }

        }

    }

}