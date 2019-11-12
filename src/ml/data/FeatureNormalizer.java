package ml.data;

import java.util.ArrayList;

/**
 * Berkay Adanali and Sarah Bashir Assignment 4
 * This class normalizes feature values by centering
 * and variance scaling
 *
 */
public class FeatureNormalizer implements DataPreprocessor {
    private ArrayList<Double> means = new ArrayList<Double>();
    private ArrayList<Double> stdDevs = new ArrayList<Double>();

    /**
     *
     * This method calculates the standard deviation and mean of the
     * training data set and adjusts accordingly
     *
     * @param train
     */
    public void preprocessTrain(DataSet train) {
        ArrayList<Example> dataSet = train.getData();
        //calculate the means of each feature
        for (int i=0;i<dataSet.get(0).getFeatureSet().size();i++) {
            double average = 0.0;
            for(int j=0;j<dataSet.size();j++){
                average += dataSet.get(j).getFeature(i);
            }
            average = average / dataSet.size();
            //store the means
            means.add(average);
        }
        //centering
        for (Example e: dataSet) {
            for(int i=0;i<e.getFeatureSet().size();i++){
                //subtract the mean from each feature value
                e.setFeature(i,(e.getFeature(i)-means.get(i)));
            }
        }
        //variance scaling (stdDev)
        for(int i=0;i<dataSet.get(0).getFeatureSet().size();i++){
            double squared = 0.0;
            for (Example e: dataSet) {
                squared += Math.pow(e.getFeature(i),2);
            }
            squared = squared/dataSet.size();
            squared = Math.sqrt(squared);
            //store the standard deviations of the data set
            stdDevs.add(squared);
        }
        for (int i=0;i<dataSet.get(0).getFeatureSet().size();i++){
            for (Example e: dataSet) {
                //divide each feature value by the standard deviation
                e.setFeature(i,(e.getFeature(i)/stdDevs.get(i)));
            }
        }
    }
    @Override
    /**
     *  Normalizes the features according to the arithmetic means
     *  and standard deviations calculated from the training data.
     *
     */
    public void preprocessTest(DataSet test) {
        ArrayList<Example>dataSet = test.getData();
        //center
        for (Example e: dataSet) {
            for(int i=0;i<e.getFeatureSet().size();i++){
                //subtract the means from the training data
                e.setFeature(i,(e.getFeature(i)-means.get(i)));
            }
        }
        //variance scale
        for (int i=0;i<dataSet.get(0).getFeatureSet().size();i++){
            for (Example e: dataSet) {
                //divide by the standard deviations from the training data
                e.setFeature(i,(e.getFeature(i)/stdDevs.get(i)));
            }
        }
    }
}
