package org.example;// DeepLearning4j Image Recognition Template Project
// Author: Maurice Müller
// This file provides a clean starting point for training and using a CNN with DL4J.

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;

public class DL4JImageTemplate {

    // Basic image parameters
    static int height = 64;
    static int width = 64;
    static int channels = 3;
    static int batchSize = 32;
    static int epochs = 5;

    static void main() throws Exception {

        // ---- 1. Dataset Paths ----
        File trainData = new File("dataset/train");
        File testData = new File("dataset/test");

        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS);

        ParentPathLabelGenerator labels = new ParentPathLabelGenerator();

        // ---- 2. Image Record Reader ----
        ImageRecordReader trainReader = new ImageRecordReader(height, width, channels, labels);
        trainReader.initialize(trainSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(
                trainReader, batchSize, 1, trainReader.getLabels().size());

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        // ---- 3. Define CNN Model ----
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5,5)
                        .stride(1,1)
                        .nIn(channels)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(trainReader.getLabels().size())
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // ---- 4. Train Model ----
        System.out.println("Training model...");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainIter);
            System.out.println("Epoch " + (i+1) + " completed.");
        }

        // ---- 5. Evaluation ----
        System.out.println("Evaluating model...");
        ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labels);
        testReader.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize);
        testIter.setPreProcessor(scaler);

        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());

        // ---- 6. Example Prediction ----
        System.out.println("Running prediction...");
        File imgFile = new File("example.jpg");
        if (imgFile.exists()) {
            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
            INDArray image = loader.asMatrix(imgFile);
            scaler.transform(image);
            INDArray output = model.output(image);
            int predicted = Nd4j.argMax(output, 1).getInt(0);
            System.out.println("Predicted class: " + trainReader.getLabels().get(predicted));
        } else {
            System.out.println("No example.jpg found for prediction test.");
        }

    }
}