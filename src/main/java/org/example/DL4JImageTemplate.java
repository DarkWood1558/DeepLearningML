package org.example; // DeepLearning4j Image Recognition Template Project



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

    // Grundlegende Bildparameter
    static int height = 64; // Höhe der Bilder
    static int width = 64; // Breite der Bilder
    static int channels = 3; // Anzahl der Farbkanäle (RGB)
    static int batchSize = 32; // Anzahl der Bilder pro Batch
    static int epochs = 5; // Anzahl der Trainings-Epochen

    static void main() throws Exception {

        // ---- 1. Dataset-Pfade ----
        File trainData = new File("dataset/train"); // Pfad zu den Trainingsdaten
        File testData = new File("dataset/test"); // Pfad zu den Testdaten

        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS); // Aufteilen der Trainingsdaten
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS); // Aufteilen der Testdaten

        ParentPathLabelGenerator labels = new ParentPathLabelGenerator(); // Generierung von Labels basierend auf dem Pfad

        // ---- 2. Image Record Reader ----
        ImageRecordReader trainReader = new ImageRecordReader(height, width, channels, labels); // Initialisierung des Readers
        trainReader.initialize(trainSplit); // Initialisierung mit den Trainingsdaten

        DataSetIterator trainIter = new RecordReaderDataSetIterator(
                trainReader, batchSize, 1, trainReader.getLabels().size()); // Erstellen des DataSetIterators

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1); // Normalisierung der Bilddaten
        scaler.fit(trainIter); // Anpassen des Scalers an die Trainingsdaten
        trainIter.setPreProcessor(scaler); // Anwenden des Scalers auf den Iterator

        // ---- 3. CNN-Modell-Definition ----
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123) // Zufallszahl für Reproduzierbarkeit
                .updater(new Adam(1e-3)) // Optimierer
                .list()
                .layer(new ConvolutionLayer.Builder(5,5) // Faltungsschicht
                        .stride(1,1)
                        .nIn(channels)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Max-Pooling-Schicht
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(256) // Dichte Schicht
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Ausgabeschicht
                        .nOut(trainReader.getLabels().size())
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // Eingabetyp definieren
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config); // Modell erstellen
        model.init(); // Modell initialisieren
        model.setListeners(new ScoreIterationListener(10)); // Listener für die Ausgabe der Scores

        // ---- 4. Modelltraining ----
        System.out.println("Modell wird trainiert...");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainIter); // Modell anpassen
            System.out.println("Epoche " + (i+1) + " abgeschlossen.");
        }

        // ---- 5. Bewertung ----
        System.out.println("Modell wird bewertet...");
        ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labels); // Reader für Testdaten
        testReader.initialize(testSplit); // Initialisierung mit den Testdaten

        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize); // Iterator für Testdaten
        testIter.setPreProcessor(scaler); // Anwenden des Scalers auf den Test-Iterator

        Evaluation eval = model.evaluate(testIter); // Bewertung des Modells
        System.out.println(eval.stats()); // Ausgabe der Bewertungsergebnisse

        // ---- 6. Beispielvorhersage ----
        System.out.println("Vorhersage wird durchgeführt...");
        File imgFile = new File("example.jpg"); // Beispielbild für die Vorhersage
        if (imgFile.exists()) {
            NativeImageLoader loader = new NativeImageLoader(height, width, channels); // Loader für das Bild
            INDArray image = loader.asMatrix(imgFile); // Bild in Matrix umwandeln
            scaler.transform(image); // Normalisierung des Bildes
            INDArray output = model.output(image); // Vorhersage des Modells
            int predicted = Nd4j.argMax(output, 1).getInt(0); // Vorhergesagte Klasse ermitteln
            System.out.println("Vorhergesagte Klasse: " + trainReader.getLabels().get(predicted)); // Ausgabe der Vorhersage
        } else {
            System.out.println("Beispielbild example.jpg nicht gefunden.");
        }
    }
}
