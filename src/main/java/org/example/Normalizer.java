package org.example;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Normalizer {

    private static final int TARGET_WIDTH = 64;
    private static final int TARGET_HEIGHT = 64;

    public static void main(String[] args) throws Exception {
        File inputDir = new File("/Users/mauricemuller/Library/CloudStorage/SynologyDrive-sync/Schule/Studium/Master/Semester 1/Maschinelles_Lernen/Verkehrszeichen/");
        File outputDir = new File("dataset/train");
        normalizeDirectory(inputDir, outputDir);
    }

    public static void normalizeDirectory(File inputDir, File outputDir) throws IOException {
        if (!outputDir.exists()) outputDir.mkdirs();

        for (File file : inputDir.listFiles()) {
            if (file.isDirectory()) {
                // Ordnerstruktur beibehalten
                normalizeDirectory(file, new File(outputDir, file.getName()));
            } else {
                processImage(file, new File(outputDir,
                        file.getName().replaceFirst("[.][^.]+$", "") + ".bmp"));
            }
        }
    }

    private static void processImage(File inputFile, File outputFile) {
        try {
            BufferedImage img = ImageIO.read(inputFile);
            if (img == null) {
                System.out.println("Überspringe ungültiges Bild: " + inputFile);
                return;
            }

            // Auf 64x64 skalieren
            Image scaled = img.getScaledInstance(TARGET_WIDTH, TARGET_HEIGHT, Image.SCALE_SMOOTH);

            // Konsistente Farbtiefe
            BufferedImage normalized = new BufferedImage(
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    BufferedImage.TYPE_3BYTE_BGR
            );

            Graphics2D g = normalized.createGraphics();
            g.drawImage(scaled, 0, 0, null);
            g.dispose();

            // speichern als BMP
            ImageIO.write(normalized, "bmp", outputFile);

            System.out.println("Konvertiert: " + inputFile.getName() + " → " + outputFile.getName());

        } catch (Exception e) {
            System.err.println("Fehler beim Verarbeiten: " + inputFile.getName());
            e.printStackTrace();
        }
    }
}
