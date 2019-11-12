package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.*;
import java.util.stream.Collectors;

import static java.lang.Math.sqrt;

/**
 * Sarah Bashir
 *
 *
 * This class uses the kNN approach to predict
 * the label for a data point
 *
 */
public class KNNClassifier implements Classifier {
    private DataSet data;
    private int k = 3;
    @Override

    /**
     *
     * Train the data set (setter method)
     */
    public void train(DataSet data) {
        this.data=data;
    }

    @Override
    /**
     *
     * Label an example based on its
     * k nearest neighbors
     *
     */
    public double classify(Example example) {
        ArrayList<KNNScore> scores = new ArrayList<KNNScore>();
        ArrayList<Example> dataArray = data.getData();
        //look through each example in the data
        for (Example e: dataArray) {
            double distance = 0.0;
            //compare each feature in given example to examples in data array
            for (int i = 0; i < e.getFeatureSet().size(); i++) {
                //calculate the distances of the examples
                distance += Math.pow(example.getFeature(i) - e.getFeature(i), 2);

            }

            distance = sqrt(distance);
            //add a knn score object with the given distance to the list
            KNNScore newScore = new KNNScore(e, distance);
            scores.add(newScore);
        }
        //sort the list of knn scores (distances and associated examples)
        Collections.sort(scores);

        //        for (KNNScore score: scores) {
//            System.out.println(score.getDistance() + " the scores");
//        }

        //return the majority label of the k neighbors
        double majLabel = 0.0;
        for (int j=0; j<k; j++) {
            majLabel += scores.get(j).getExample().getLabel();
        }
        if (majLabel >= 0) {
            //label 1
            return 1.0;
        } else {
            //label -1
            return -1.0;
        }
    }


    /**
     *
     * Setter method for how many neighbors
     *
     * @param k
     */
    public void setK(int k) {
        this.k = k;
    }


    /**
     *
     * Helper class to create kNN score objects
     * (store example and distance)
     *
     */
    private class KNNScore implements Comparable<KNNScore>{
        private Example example;
        private double distance;

        /**
         *
         * Comparator method to find closest neighbors
         * and store example and distance
         *
         * @param knnScore
         * @return
         */
        public int compareTo(KNNScore knnScore) {
            //compare distance and knnScore.distance
            if (this.distance < knnScore.distance){
                return -1;
            }

            else if (this.distance == knnScore.distance){
                return 0;
            }

            else{
                return 1;
            }

        }

        /**
         * Getter method for example
         *
         * @return
         */
        public Example getExample(){
            return this.example;
        }

        /**
         * Getter method for distance
         *
         * @return
         */
        public double getDistance(){
            return this.distance;
        }


        /**
         *
         * KNN score object
         *
         * @param e
         * @param dist
         */
        private KNNScore(Example e, double dist) {
            this.example = e;
            this.distance = dist;
        }

    }

    public static void main (String[] args) {
        DataSet data = new DataSet("data/titanic-train.real.csv");
        Example ex = new Example(data.getData().get(0));
        KNNClassifier classifier = new KNNClassifier();
        classifier.train(data);
        classifier.classify(ex);
    }



}
