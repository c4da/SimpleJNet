package matrix;

import java.util.Arrays;

public class Matrix {
    private static final String NUMBER_FORMAT = "%+12.5f ";
    private static final double TOLERANCE = 1e-6;
    private final int rows;
    private final int cols;
    public interface Producer{
        double produce(int index);
    }
    public interface IndexValueProducer {
        double produce(int index, double value);
    }

    public interface ValueProducer {
        double produce(double value);
    }

    public interface IndexValueConsumer{
        void consume(int index, double value);
    }
    public interface RowColProducer{
        double produce(int row, int col, double value);
    }

    public interface RowColValueConsumer{
        void consumer(int row, int col, double value);
    }

    public interface RowColIndexValueConsumer{
        void consumer(int row, int col, int index, double value);
    }
    private double[] a;
    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        a = new double[rows*cols];
    }

    public Matrix modify(ValueProducer producer){
        for (int i = 0; i < a.length; i++) {
            a[i] = producer.produce(a[i]);
        }
        return this;
    }

    public void forEach(RowColIndexValueConsumer consumer){
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                consumer.consumer(i, j, index,a[index++]);
            }
        }
    }
    public void forEach(RowColValueConsumer consumer){
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                consumer.consumer(i, j, a[index++]);
            }
        }
    }
    public void forEach(IndexValueConsumer consumer){
        for (int i = 0; i < a.length; i++) {
            consumer.consume(i, a[i]);
        }
    }

    public int getRows(){
        return rows;
    }

    public int getCols(){
        return cols;
    }
    public Matrix(int rows, int cols, Producer producer){
        this(rows, cols);

        for (int i = 0; i < a.length; i++) {
            a[i] = producer.produce(i);
        }
    }

    public Matrix apply(IndexValueProducer producer){
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < a.length; i++) {
            result.a[i] = producer.produce(i, a[i]);
        }

        return result;
    }

    public Matrix modify(RowColProducer producer){
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a[index] = producer.produce(i, j, a[index]);

                ++index;
            }
        }

        return this;
    }

    public Matrix softMax(){
        Matrix result = new Matrix(rows, cols, i -> Math.exp(a[i]));

        Matrix colSum = result.sumColumns();

        result.modify((row, col, value) -> {
            return value/colSum.get(col);
        });

        return result;
    }

    public Matrix multiply(Matrix m){
        Matrix result = new Matrix(rows, m.cols);

        assert cols == m.rows : "Cannot multiply. Wrong matrix dimensions.";

        for (int row = 0; row < result.rows; row++) {
            int index1 = row * result.cols;
            int index2 = row * cols;
            for (int n = 0; n < cols; n++) {
                int index3 = n * m.cols;
                for (int col = 0; col < result.cols; col++) {
                    result.a[index1 + col] += a[index2 + n] * m.a[col + index3];
                }
            }
        }

        return result;
    }

    public Matrix sumColumns(){
        Matrix result = new Matrix(1, cols);
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.a[j] += a[index++];
            }
        }
        return result;
    }

    public double get(int index){
        return a[index];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null){
            return false;
        }
        if (getClass() != o.getClass()){
            return false;
        }
        Matrix other = (Matrix) o;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - other.a[i]) > TOLERANCE){
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(a);
    }

    public String toString(boolean showValues){
        if (showValues){
            return toString();
        } else {
            return rows + "x" + cols;
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int col = 0; col < cols; ++col){
                sb.append(String.format(NUMBER_FORMAT, a[index]));
                ++index;
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
