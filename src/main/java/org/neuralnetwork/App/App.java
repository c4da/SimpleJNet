package org.neuralnetwork.App;

import org.neuralnetwork.Engine;
import org.neuralnetwork.Transform;
import org.neuralnetwork.loader.Loader;
import org.neuralnetwork.loader.test.TestLoader;

public class App {

    public static void main(String[] args) {


        String filename = "neural1.net";
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
        if (neuralNetwork == null){
            System.out.println("Unable to load neural network from saved. Creating from scratch.");

            int inputRows = 10;
            int outputRows = 3;

            neuralNetwork = new NeuralNetwork();
            neuralNetwork.add(Transform.DENSE, 100, inputRows);
            neuralNetwork.add(Transform.RELU);
            neuralNetwork.add(Transform.DENSE, outputRows);
            neuralNetwork.add(Transform.SOFTMAX);

            neuralNetwork.setEpochs(1);
            neuralNetwork.setLearningRate(0.02, 0.001);
        } else {
            System.out.println("Loaded from: " + filename);

        }
        System.out.println(neuralNetwork);

        Loader trainLoader = new TestLoader(60_000, 32);
        Loader testLoader = new TestLoader(10_000, 32);

        neuralNetwork.fit(trainLoader, testLoader);

        if (neuralNetwork.save("neuralNet.net")){
            System.out.println("Saved to " + filename);
        } else {
            System.out.println("Unable to save to " + filename);
        };

    }
}
