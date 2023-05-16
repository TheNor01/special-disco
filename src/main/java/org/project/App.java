package org.project;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.SMOTE;
import weka.attributeSelection.InfoGainAttributeEval;


import java.io.File;
import java.util.Random;

/**
 * I punti 2-5 devo essere svolti scrivendo dei programmi Java utilizzando le API di Weka.
 *
 */
public class App
{
    public static void main( String[] args ) throws Exception {
        //DataSource source = new DataSource("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/1.csv");
        DataSource source = new DataSource("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/inputDS/positive1.csv");
        Instances training = source.getDataSet();
        // header to fix https://stackoverflow.com/questions/3517186/using-weka-java-code-how-convert-csv-without-header-row-to-arff-format

        training.renameAttribute(0,"sepal length");
        training.renameAttribute(1,"sepal width");
        training.renameAttribute(2,"petal length");
        training.renameAttribute(3,"petal width");
        training.renameAttribute(4,"class");

        if (training.classIndex() == -1) training.setClassIndex(training.numAttributes() - 1);

        //save arff
        File arfFIle = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/training1.arff");
        if(arfFIle.exists()) arfFIle.delete();
        ArffSaver rawtrainingSaver = new ArffSaver();
        rawtrainingSaver.setInstances(training);
        rawtrainingSaver.setFile(arfFIle);
        rawtrainingSaver.writeBatch();


        //DataSource sourceNegative_1 = new DataSource("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/2.csv");
        //DataSource sourceNegative_2 = new DataSource("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/3.csv");
        //Instances trainingNegative_1 = sourceNegative_1.getDataSet();
        //Instances trainingNegative_2 = sourceNegative_2.getDataSet();
        // header to fix https://stackoverflow.com/questions/3517186/using-weka-java-code-how-convert-csv-without-header-row-to-arff-format

        /*
        trainingNegative_1.renameAttribute(0,"sepal length");
        trainingNegative_2.renameAttribute(0,"sepal length");
        trainingNegative_1.renameAttribute(1,"sepal width");
        trainingNegative_2.renameAttribute(1,"sepal width");
        trainingNegative_1.renameAttribute(2,"petal length");
        trainingNegative_2.renameAttribute(2,"petal length");
        trainingNegative_1.renameAttribute(3,"petal width");
        trainingNegative_2.renameAttribute(3,"petal width");
        trainingNegative_1.renameAttribute(4,"class");
        trainingNegative_2.renameAttribute(4,"class");

         */

        /*
        int[] indicesToRemove = {4};
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(indicesToRemove);
        removeFilter.setInvertSelection(false);
        removeFilter.setInputFormat(trainingNegative_1);

        Instances newNegative1 = Filter.useFilter(trainingNegative_1, removeFilter);

        removeFilter.setInputFormat(trainingNegative_2);
        Instances newNegative2 = Filter.useFilter(trainingNegative_2, removeFilter);

        removeFilter.setInputFormat(training);
        Instances newTraining = Filter.useFilter(training, removeFilter);

        //add new attribute class "NOMINAL"
        Add filterADD = new Add();
        filterADD.setAttributeIndex("last");
        filterADD.setAttributeName("class");
        // filterADD.setNominalLabels("class");
        filterADD.setInputFormat(newNegative1);
        newNegative1 = Filter.useFilter(newNegative1, filterADD);

        //training nominal j48
        filterADD.setAttributeIndex("last");
        filterADD.setNominalLabels("positive,negative");
        filterADD.setAttributeName("class");
        filterADD.setInputFormat(newTraining);
        newTraining = Filter.useFilter(newTraining, filterADD);


        filterADD.setAttributeIndex("last");
        filterADD.setAttributeName("class");
        filterADD.setInputFormat(newNegative2);
        newNegative2 = Filter.useFilter(newNegative2, filterADD);

        for (int i = 0; i < newNegative1.numInstances(); i++) {
            // 2. numeric
            newNegative1.instance(i).setValue(newNegative1.numAttributes() - 1, 1);
            newNegative2.instance(i).setValue(newNegative2.numAttributes() - 1, 1);
        }

        Random rand = new Random(1);
        for (int i = 0; i < newTraining.numInstances(); i++) {
            // 2. nominal
            newTraining.instance(i).setValue(newTraining.numAttributes() - 1, rand.nextInt(1));

        }


        if (newNegative1.classIndex() == -1) newNegative1.setClassIndex(newNegative1.numAttributes() - 1);
        if (newNegative2.classIndex() == -1) newNegative2.setClassIndex(newNegative2.numAttributes() - 1);
        if (newTraining.classIndex() == -1) newTraining.setClassIndex(newTraining.numAttributes() - 1);



        File arfFIle1 = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/trainingNegative1.arff");
        if(arfFIle1.exists()) arfFIle1.delete();
        ArffSaver rawtrainingSaver1 = new ArffSaver();
        rawtrainingSaver1.setInstances(newNegative1);
        rawtrainingSaver1.setFile(arfFIle1);
        rawtrainingSaver1.writeBatch();


        File arfFIle2 = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/trainingNegative2.arff");
        if(arfFIle2.exists()) arfFIle2.delete();
        ArffSaver rawtrainingSaver2 = new ArffSaver();
        rawtrainingSaver2.setInstances(newNegative2);
        rawtrainingSaver2.setFile(arfFIle2);
        rawtrainingSaver2.writeBatch();

        //merge all

        //Instances tmp1 = Instances.mergeInstances(newNegative1,newNegative2);
        //Instances allTraining = Instances.mergeInstances(tmp1,training);


        //Remove percentage FROM TRAINING ALL
         */



        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(training);
        rp.setInvertSelection(true);
        rp.setPercentage(30);

        Instances testing = Filter.useFilter(training, rp);

        File arfFIletesting = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/testing.arff");
        if(arfFIletesting.exists()) arfFIletesting.delete();
        ArffSaver rawtrainingSavertesting = new ArffSaver();
        rawtrainingSavertesting.setInstances(testing);
        rawtrainingSavertesting.setFile(arfFIletesting);
        rawtrainingSavertesting.writeBatch();


        //Classifier section

        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options

        //tree.buildClassifier(training);   // build classifier

        //The crossValidateModel takes care of training and evaluating the classifier. (It creates a copy of the original classifier that you hand over to the crossValidateModel for each run of the cross-validation.)

        //j48 cannot handle numeric class??!

        int folds = 5;
        Evaluation eval = new Evaluation(training);
        eval.crossValidateModel(tree, testing, folds, new Random(1));


        //TPR, la precisione e la recall.
        //cross validation
        System.out.println(eval.toSummaryString("\nResults\n======\n", true));
        System.out.println("F-measure : "+eval.weightedFMeasure());
        System.out.println("precision : "+eval.weightedPrecision());
        System.out.println("recall : "+eval.weightedRecall());

        //multiclass j48

        DataSource sourceAll = new DataSource("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/inputDS/positive1.csv");
        Instances trainingAll= sourceAll.getDataSet();
        // header to fix https://stackoverflow.com/questions/3517186/using-weka-java-code-how-convert-csv-without-header-row-to-arff-format

        trainingAll.renameAttribute(0,"sepal length");
        trainingAll.renameAttribute(1,"sepal width");
        trainingAll.renameAttribute(2,"petal length");
        trainingAll.renameAttribute(3,"petal width");
        trainingAll.renameAttribute(4,"class");

        if (trainingAll.classIndex() == -1) trainingAll.setClassIndex(trainingAll.numAttributes() - 1);

        //save arff
        File arfFile_all = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/training_all.arff");
        if(arfFile_all.exists()) arfFile_all.delete();
        ArffSaver rawtrainingSaverAll = new ArffSaver();
        rawtrainingSaverAll.setInstances(trainingAll);
        rawtrainingSaverAll.setFile(arfFile_all);
        rawtrainingSaverAll.writeBatch();


        Instances testingAll = Filter.useFilter(trainingAll, rp);

        Evaluation evalAll = new Evaluation(trainingAll);
        evalAll.crossValidateModel(tree, testingAll, folds, new Random(1));

        System.out.println(evalAll.toSummaryString("\nResults\n======\n", true));
        System.out.println("F-measure : "+evalAll.weightedFMeasure());
        System.out.println("precision : "+evalAll.weightedPrecision());
        System.out.println("recall : "+evalAll.weightedRecall());


        //SMOTE positive vs negative, rebalance
        SMOTE smote=new SMOTE();
        smote.setInputFormat(training);
        Instances Trains_smote= Filter.useFilter(training, smote);

        File arfFile_smote = new File("/Users/thenor/Desktop/Aldo/Neodata/SaverioProjects/BinaryClassification/src/main/java/org/resources/irisSingleCl/training_smote_1.arff");
        if(arfFile_smote.exists()) arfFile_smote.delete();
        ArffSaver training_smote = new ArffSaver();
        training_smote.setInstances(Trains_smote);
        training_smote.setFile(arfFile_smote);
        training_smote.writeBatch();

        Evaluation evalAllSmote = new Evaluation(Trains_smote);
        evalAllSmote.crossValidateModel(tree, testingAll, folds, new Random(1));

        System.out.println(evalAllSmote.toSummaryString("\nResults\n======\n", true));
        System.out.println("F-measure : "+evalAllSmote.weightedFMeasure());
        System.out.println("precision : "+evalAllSmote.weightedPrecision());
        System.out.println("recall : "+evalAllSmote.weightedRecall());
        System.out.println(evalAllSmote.toMatrixString());

        //https://waikato.github.io/weka-wiki/performing_attribute_selection/#meta-classifier

        //attribute selecion (infogain+ ranker (valore infogain >0) e cfsSubsetEval+ best first) e il filtro SMOTE

        AttributeSelection selector = new AttributeSelection();
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        ranker.setThreshold(0.0);
        selector.setEvaluator(evaluator);
        selector.setSearch(ranker);



    }
}
