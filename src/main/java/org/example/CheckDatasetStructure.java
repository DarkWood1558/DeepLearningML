package org.example;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class CheckDatasetStructure {

    public static void main(String[] args) {
        File trainDir = new File("dataset/train");
        File testDir = new File("dataset/test");

        System.out.println("🔍 Prüfe Dataset-Struktur...");
        checkDataset(trainDir, testDir);
    }

    public static void checkDataset(File trainDir, File testDir) {
        if (!trainDir.exists() || !testDir.exists()) {
            System.out.println("❌ Train- oder Test-Ordner existiert nicht!");
            return;
        }

        Set<String> trainLabels = listLabelDirs(trainDir);
        Set<String> testLabels = listLabelDirs(testDir);

        System.out.println("\n📂 Train Labels: " + trainLabels);
        System.out.println("📂 Test Labels:  " + testLabels);

        // Labels die nur im Testset existieren
        Set<String> onlyInTest = new HashSet<>(testLabels);
        onlyInTest.removeAll(trainLabels);

        // Labels die nur im Trainset existieren
        Set<String> onlyInTrain = new HashSet<>(trainLabels);
        onlyInTrain.removeAll(testLabels);

        if (!onlyInTest.isEmpty()) {
            System.out.println("\n⚠️ Test-Ordner enthält zusätzliche Klassen:");
            onlyInTest.forEach(label -> System.out.println("   ➤ " + label));
        }

        if (!onlyInTrain.isEmpty()) {
            System.out.println("\n⚠️ Train-Ordner enthält Klassen, die im Test fehlen:");
            onlyInTrain.forEach(label -> System.out.println("   ➤ " + label));
        }

        if (onlyInTest.isEmpty() && onlyInTrain.isEmpty()) {
            System.out.println("\n✅ Struktur OK – gleiche Klassen in train und test.");
        }
    }

    private static Set<String> listLabelDirs(File parent) {
        Set<String> result = new HashSet<>();

        File[] files = parent.listFiles();
        if (files == null) return result;

        for (File f : files) {
            if (f.isDirectory()) {
                if (f.getName().equalsIgnoreCase(".DS_Store")) {
                    System.out.println("⚠️ Entferne .DS_Store in: " + parent.getPath());
                    f.delete();
                    continue;
                }
                result.add(f.getName());
            } else if (f.getName().equalsIgnoreCase(".DS_Store")) {
                System.out.println("⚠️ Entferne Datei: " + f.getPath());
                f.delete();
            } else {
                System.out.println("⚠️ Ungültige Datei: " + f.getPath());
            }
        }

        return result;
    }
}