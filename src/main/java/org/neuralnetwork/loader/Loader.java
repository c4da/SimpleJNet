package org.neuralnetwork.loader;

public interface Loader {
    public MetaData open();
    public void close();

    public MetaData getMetaData();
    public BatchData readBatch();
}
