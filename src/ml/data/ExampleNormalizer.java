package ml.data;

import java.util.ArrayList;

/**
 * Berkay Adanali and Sarah Bashir Assignment 4
 *
 * This class normalizes each example so its length is 1
 *
 */
public class ExampleNormalizer implements DataPreprocessor {
    /**
     * Helper method that divides each feature value by the square root of the sums of each value
     *
     *
     * @param train
     */
    public void normalize(DataSet train){
        ArrayList<Example> examples = train.getData();
        //for each example in the data set
        for (Example e : examples){
            //calculate the square root of sum of feature values squared
            double featureSum = 0.0;
            for (int i = 0; i < e.getFeatureSet().size(); i++){
                featureSum += Math.pow(e.getFeature(i),2);
            }
            featureSum = Math.sqrt(featureSum);
            //divide each feature value and set the new number
            for (int j = 0; j < e.getFeatureSet().size(); j++){
                e.setFeature(j,(e.getFeature(j)/featureSum));
            }
        }


    }

    @Override
    /**
     * Normalize training data set
     *
     */
    public void preprocessTrain(DataSet train) {
        this.normalize(train);
    }


    @Override
    /**
     *
     * Normalize testing data set
     *
     */
    public void preprocessTest(DataSet test) {
        this.normalize(test);

    }
}
