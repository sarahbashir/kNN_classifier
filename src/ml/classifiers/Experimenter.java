package ml.classifiers;

import ml.data.*;

public class Experimenter {

    public static void percepTest() {
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        AveragePerceptronClassifier avgPercep = new AveragePerceptronClassifier();
        avgPercep.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,false);
            for (int j=0; j<100; j++){
                //normalize
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = avgPercep.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }


    public static void percepFeat() {
        FeatureNormalizer normalizer = new FeatureNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        AveragePerceptronClassifier avgPercep = new AveragePerceptronClassifier();
        avgPercep.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                normalizer.preprocessTrain(knnSplit.getTrain());
                normalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = avgPercep.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }

    public static void percepEx() {
        ExampleNormalizer normalizer = new ExampleNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        AveragePerceptronClassifier avgPercep = new AveragePerceptronClassifier();
        avgPercep.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                normalizer.preprocessTrain(knnSplit.getTrain());
                normalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = avgPercep.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }

    public static void percepExFeat() {
        ExampleNormalizer exNormalizer = new ExampleNormalizer();
        FeatureNormalizer featNormalizer = new FeatureNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        AveragePerceptronClassifier avgPercep = new AveragePerceptronClassifier();
        avgPercep.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                featNormalizer.preprocessTrain(knnSplit.getTrain());
                exNormalizer.preprocessTrain(knnSplit.getTrain());
                featNormalizer.preprocessTest(knnSplit.getTest());
                exNormalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = avgPercep.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }

    public static void knnTest() {
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10, true);
        KNNClassifier knn = new KNNClassifier();
        knn.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,false);
            for (int j=0; j<100; j++){
                //normalize
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = knn.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }


    public static void knnFeat() {
        FeatureNormalizer normalizer = new FeatureNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        KNNClassifier knn = new KNNClassifier();
        knn.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                normalizer.preprocessTrain(knnSplit.getTrain());
                normalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = knn.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }

    public static void knnEx() {
        ExampleNormalizer normalizer = new ExampleNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        KNNClassifier knn = new KNNClassifier();
        knn.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                normalizer.preprocessTrain(knnSplit.getTrain());
                normalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = knn.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }

    }

    public static void knnExFeat() {
        ExampleNormalizer exNormalizer = new ExampleNormalizer();
        FeatureNormalizer featNormalizer = new FeatureNormalizer();
        DataSet data = new DataSet("data/titanic-train.real.csv");
        CrossValidationSet knnSet = new CrossValidationSet(data,10);
        KNNClassifier knn = new KNNClassifier();
        knn.train(data);
        double[] accuracies = new double[10];
        for (int i=0; i<10; i++) {
            DataSetSplit knnSplit = knnSet.getValidationSet(i,true);
            for (int j=0; j<100; j++){
                //normalize
                featNormalizer.preprocessTrain(knnSplit.getTrain());
                exNormalizer.preprocessTrain(knnSplit.getTrain());
                featNormalizer.preprocessTest(knnSplit.getTest());
                exNormalizer.preprocessTest(knnSplit.getTest());
                double runAccuracy = 0.0;
                for(Example e: knnSplit.getTest().getData()) {
                    double prediction = knn.classify(e);
                    if (prediction == e.getLabel()){
                        runAccuracy++;
                    }

                }
                runAccuracy = runAccuracy/(knnSplit.getTest().getData().size());
                accuracies[i] += runAccuracy;
            }
            accuracies[i] = accuracies[i]/100;
            System.out.println(accuracies[i]);
        }



    }
    public static void main (String[] args) {



    }

}
