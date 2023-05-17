package org.project;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.SMOTE;
import weka.attributeSelection.InfoGainAttributeEval;


import java.io.File;
import java.util.ArrayList;
import java.util.Random;

/**
 * I punti 2-5 devo essere svolti scrivendo dei programmi Java utilizzando le API di Weka.
 *
 */
public class App
{
 private static void MetaClassiffier(SMOTE smote,int sourcesSize,int folds, J48 tree) throws Exception{
    for(int i=0;i<sourcesSize;i++){

            DataSource tmp = new DataSource("src/main/java/org/resources/irisSingleCl/inputDS/agg_positive"+i+".arff");
            Instances dataset = tmp.getDataSet();
            if (dataset.classIndex() == -1) dataset.setClassIndex(dataset.numAttributes() - 1);

            
            //Meta classification

            //https://waikato.github.io/weka-wiki/performing_attribute_selection/#meta-classifier

            //attribute selecion (infogain+ ranker (valore infogain >0) e cfsSubsetEval+ best first) e il filtro SMOTE (inglobato)

            //repeat 2) step using 

            //apply smote

            smote.setInputFormat(dataset);
            Instances Trains_smote= Filter.useFilter(dataset, smote);


            //1 meta
            AttributeSelectedClassifier classifier1 = new AttributeSelectedClassifier();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            ranker.setThreshold(0.1);

            classifier1.setClassifier(tree);
            classifier1.setEvaluator(evaluator);
            classifier1.setSearch(ranker);

            //2 meta
            AttributeSelectedClassifier classifier2 = new AttributeSelectedClassifier();
            CfsSubsetEval eva = new CfsSubsetEval();
            BestFirst bf = new BestFirst();
            classifier2.setClassifier(tree);
            classifier2.setEvaluator(eva);
            classifier2.setSearch(bf);

    
            //identificare le features selezionate

            System.out.println("FIRST META: infogain+ranker");

            Evaluation evaluation = new Evaluation(Trains_smote);
            evaluation.crossValidateModel(classifier1, Trains_smote, folds, new Random(1));
            System.out.println(evaluation.toSummaryString());

            classifier1.measureNumAttributesSelected();


            
            System.out.println("-----------");

            System.out.println("SECOND META: CfsSubsetEval + BestFirst");
            Evaluation evaluation2 = new Evaluation(Trains_smote);
            evaluation2.crossValidateModel(classifier2, Trains_smote, folds, new Random(1));
            System.out.println(evaluation2.toSummaryString());




        }
    }
    private static void RebalanceWithSmote(SMOTE smote,int sourcesSize,int folds, J48 tree) throws Exception{
        //SMOTE positive vs negative, rebalance
        for(int i=0;i<sourcesSize;i++){
            DataSource sourceSmote = new DataSource("src/main/java/org/resources/irisSingleCl/inputDS/agg_positive"+i+".arff");
            Instances trainingSmote = sourceSmote.getDataSet();
            if (trainingSmote.classIndex() == -1) trainingSmote.setClassIndex(trainingSmote.numAttributes() - 1);

            smote.setInputFormat(trainingSmote);
            Instances Trains_smote= Filter.useFilter(trainingSmote, smote);

            Evaluation evalAllSmote = new Evaluation(Trains_smote);
            evalAllSmote.crossValidateModel(tree, Trains_smote, folds, new Random(1));


            System.out.println("CLASSIFIER eval SMOTE:"+i);
            System.out.println(evalAllSmote.toSummaryString("\nResults\n======\n", true));
            System.out.println("F-measure : "+evalAllSmote.weightedFMeasure());
            System.out.println("precision : "+evalAllSmote.weightedPrecision());
            System.out.println("recall : "+evalAllSmote.weightedRecall());
            System.out.println(evalAllSmote.toMatrixString());

        }    
    }
    private static void CreateClassifierFromAllCsvJoined(String allCsvJoined ,int folds,J48 tree) throws Exception{
        CSVLoader loader = new CSVLoader();
            loader.setNoHeaderRowPresent(true);
            loader.setSource(new File(allCsvJoined));
            Instances sourceIs=loader.getDataSet();

            if (sourceIs.classIndex() == -1) sourceIs.setClassIndex(sourceIs.numAttributes() - 1);

            NumericToNominal numericToNominal = new NumericToNominal();
            String[] optionsNumeric = new String[2];
            optionsNumeric[0] = "-R";
            optionsNumeric[1] = "last";
            numericToNominal.setOptions(optionsNumeric);
            numericToNominal.setInputFormat(sourceIs);

            Instances nominalALL = Filter.useFilter(sourceIs, numericToNominal);
            if (nominalALL.classIndex() == -1) nominalALL.setClassIndex(nominalALL.numAttributes() - 1);


            Evaluation evalAll = new Evaluation(nominalALL);
            evalAll.crossValidateModel(tree, nominalALL, folds, new Random(1));

            System.out.println("ALL CLASSIFIER");

            System.out.println(evalAll.toSummaryString("\nResults\n======\n", true));
            System.out.println("F-measure : "+evalAll.weightedFMeasure());
            System.out.println("precision : "+evalAll.weightedPrecision());
            System.out.println("recall : "+evalAll.weightedRecall());
    }

    private static void CreateModel(int sourcesSize,int folds,J48 tree) throws Exception{
        

        for(int i=0;i<sourcesSize;i++){
            DataSource source = new DataSource("src/main/java/org/resources/irisSingleCl/inputDS/agg_positive"+i+".arff");
            Instances training = source.getDataSet();
            if (training.classIndex() == -1) training.setClassIndex(training.numAttributes() - 1);
            //The crossValidateModel takes care of training and evaluating the classifier. (It creates a copy of the original classifier that you hand over to the crossValidateModel for each run of the cross-validation.)
            //j48 cannot handle numeric class??!
            Evaluation eval = new Evaluation(training);
            eval.crossValidateModel(tree, training, folds, new Random(1));

            //TPR, la precisione e la recall.
            //cross validation
            System.out.println("CLASSIFIER eval:"+i);
            System.out.println(eval.toSummaryString("\nResults\n======\n", true));
            System.out.println("F-measure : "+eval.weightedFMeasure());
            System.out.println("precision : "+eval.weightedPrecision());
            System.out.println("recall : "+eval.weightedRecall());
        }
    }

    private static void FormatDataset(ArrayList<String> sources) throws Exception{

        ArrayList<Instances> inputInstances = new ArrayList<>();
        ArrayList<Instances> outputInstances = new ArrayList<>();
        
        //create Instances from DataSources
        for(String s : sources){
            CSVLoader loader = new CSVLoader();
            loader.setNoHeaderRowPresent(true);
            loader.setSource(new File(s));
            inputInstances.add(loader.getDataSet());
        }

        for(int i=0;i<inputInstances.size();i++){

            int positiveIndex = i;
            Instances positiveInstance = inputInstances.get(positiveIndex);

            ArrayList<Instances> allInstances  = inputInstances;
            //negativeInstance.remove(positive);
            
            //remove last columns
            int[] indicesToRemove = {positiveInstance.numAttributes()-1};
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(indicesToRemove);
            removeFilter.setInvertSelection(false);
        

            ArrayList<Instances> removedClassInstance  = new ArrayList<>();
            for(Instances is : allInstances){
                removeFilter.setInputFormat(is);
                Instances tmp = Filter.useFilter(is, removeFilter);
                removedClassInstance.add(tmp);
            }


            //add new attribute class "NOMINAL"
            Add filterADD = new Add();
            filterADD.setAttributeIndex("last");
            filterADD.setAttributeName("class");

            ArrayList<Instances> addedClassInstance  = new ArrayList<>();
            for(Instances is : removedClassInstance){
                filterADD.setInputFormat(is);
                Instances tmp = Filter.useFilter(is, filterADD);
                if (tmp.classIndex() == -1) tmp.setClassIndex(tmp.numAttributes() - 1);
                addedClassInstance.add(tmp);
            }
            
            Instances positiveIs = addedClassInstance.remove(positiveIndex);


            //set negative to all classes
            for(Instances is : addedClassInstance){
                for (int j = 0; j < is.numInstances(); j++) {
                    // 2. nominal
                    is.instance(j).setValue(is.numAttributes() - 1, 1);
                }
            }

            //positive instance
            for (int j = 0; j < positiveIs.numInstances(); j++) {
                // 2. nominal
                positiveIs.instance(j).setValue(positiveIs.numAttributes() - 1, 0);
            }

            //create aggregate Dataset

            for(Instances its: addedClassInstance){
                for(Instance is : its) positiveIs.add(is);
            }

            if (positiveIs.classIndex() == -1) positiveIs.setClassIndex(positiveIs.numAttributes() - 1);


            //numeric to nominal
            NumericToNominal numericToNominal = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = "last";
            numericToNominal.setOptions(options);
            numericToNominal.setInputFormat(positiveIs);


            Instances nominalIs = Filter.useFilter(positiveIs, numericToNominal);
            outputInstances.add(nominalIs);
            
            System.out.println("TOTAL inside records: "+positiveIs.numInstances());
            System.out.println("Attribute number: "+positiveIs.numAttributes());
        } //end cycle


        for (int i=0;i<outputInstances.size();i++){
            File arfFIle = new File("src/main/java/org/resources/irisSingleCl/inputDS/agg_positive"+i+".arff");
            if(arfFIle.exists()) arfFIle.delete();
            ArffSaver rawtrainingSaver = new ArffSaver();
            rawtrainingSaver.setInstances(outputInstances.get(i));
            rawtrainingSaver.setFile(arfFIle);
            rawtrainingSaver.writeBatch();
        }
        return;
    }

    public static void main( String[] args ) throws Exception {


        ArrayList<String> sources = new ArrayList<>();
        sources.add("src/main/java/org/resources/irisSingleCl/1.csv");
        sources.add("src/main/java/org/resources/irisSingleCl/2.csv");
        sources.add("src/main/java/org/resources/irisSingleCl/3.csv");

        /*
        Punto 1
            Scrivere un programma in Java prenda in ingresso un N file csv (contenuti in una cartella)
            contenenti un dataset con una sola classe (esempio C1.csv, contiene solo istanze della classe
            C1, ha tante righe quante le istanze e tante colonne quanti sono gli attributi che descrivono le
            istanze + la colonna della classe ) e restituisca N dataset binari, contenuti in N file .arff (file1,
            fil2, …fileN). Ciascun dataset binario, contenuto nel file i-esimo, deve avere come istanze della
            classe positiva tutte le istanze della classe i-esima Ci (descritta e contenuta nell’i-esimo file
            CSV). Le istanze della classe negativa devono essere tutte quelle delle classi restanti (contenute
            e descritte nei file CSV restanti). Ad esempio, si supponga di avere tre file csv (c1.csv, c2.csv,
            c3.csv). Il primo dataset binario avrà come istanze della classe positiva tutte quelle della classe
            1 (in 1.csv) e come istanze della classe negativa quelle della classe 2 e 3 (in c2.csv e c3.csv). Il
            secondo dataset binario avrà come istanze della classe positiva tutte quelle della classe 2 e
            come istanze della classe negativa quelle della classe 1 e 3. Infine, il terzo dataset binario avrà
            come istanze della classe positiva tutte quelle della classe 3 e come istanze della classe
            negativa quelle della classe 3 e 2.
        */
        FormatDataset(sources);

        /*
        Punto 2
            Costruire N modelli di classificazione utilizzando gli N dataset generati al punto 1. Valutare la
            bontà di ciascuno dei modelli di classificazione in cross validation (5 fold) e riportare per
            ciascun di essi il TPR, la precisione e la recall. Usare come classificatore il J48 
        */
       //Classifier section
        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        int folds = 5;

        CreateModel(sources.size(),folds,tree);


        /*
        Punto 3
            Costruire un classificatore J48 utilizzando il dataset che si ottiene mettendo assieme tutte le
            istanze presenti nei file csv forniti (si crea quindi un dataset multi classe con N classi).
            Valutare le prestazioni, sempre in cross validation, estraendo per ciascuna classe il la
            precisione e la recall. Confrontare i risultati ottenuti con quelli ottenuti nel punto 2).
        */
       
        //DA MIGLIORARE RENDENDO AUTOMATICA L'UNIONE DEI CSV        
        String allCsvJoined="src/main/java/org/resources/irisSingleCl/inputDS/allClasses.csv";
        CreateClassifierFromAllCsvJoined(allCsvJoined,folds,tree);
        
        /*
        Punto 4
            Ripetere il punto 2) ribilanciando la classe positiva usando il filtro SMOTE. 
            Ripetere quindi i confronti con il classificatore del punto 3).
        */
        SMOTE smote=new SMOTE();
        RebalanceWithSmote(smote,sources.size(),folds,tree);
        

        /*
        Punto 5
            Ripetere il punto 2 usando due metaclassificatori che inglobino ciascuno un metodo diverso di
            attribute selecion (infogain+ ranker (valore infogain >0) e cfsSubsetEval+ best first) e il filtro
            SMOTE come al punto 4. Per ogni classe (quindi per ogni modello, identificare quali sono le
            caratteristiche selezionate).
        */
        MetaClassiffier(smote,sources.size(),folds,tree);
        
    }
}
