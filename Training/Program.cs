using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Models;
using System;
using System.Threading.Tasks;

namespace Training
{
    class Program
    {
        //Training and test data ratio: 7:3
        //There is a format error that I could not find the reason
        const string trainingDataFile = @".\Data\all_news_train.tsv";
        const string testDataFile = @".\Data\all_news_test.tsv";
        const string modelPath = @".\Data\Model.zip"; //Make sure to check model after every change made in the dataset. 
                                                      //The model that is in the "Learned" folder does not change on its own.
                                                      //Copy the model that is in Training\bin\x64\Debug\Data to the Learned folder to make predictions.

        //Main function for evaluation
        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                // Get a model trained to use for evaluation
                Console.WriteLine("Training Data Set");
                Console.WriteLine("-----------------");
                var model = await TrainAsync(trainingDataFile, modelPath); //Calling train function

                Console.WriteLine();
                Console.WriteLine("Evaluating Training Results");
                Console.WriteLine("---------------------------");
                Evaluate(model, testDataFile); //Calling evaluate function

                Console.WriteLine("Press any key to quit.");
                Console.ReadKey();

                
            }).GetAwaiter().GetResult();
        }

        /// Trains a model using the configured data file in trainingDataFile path
        /// and outputs a model as configured in the modelPath parameter.
        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>>
            TrainAsync(string trainingDataFile, string modelPath)
        {
            //Creating a pipeline
            var pipeline = new LearningPipeline();

            //Loading the training data
            pipeline.Add(new TextLoader(trainingDataFile).CreateFrom<ClassificationData>());

            //Taking the label column and maping it to a number 
            pipeline.Add(new Dictionarizer("Label"));

            //Coverting words into vectors, tokenizing them and using N-Gram algorithm to decide weights
            pipeline.Add(new TextFeaturizer("Features", "News")
            {
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }  //Changing the legth to take a better accuracy
            });

            //Different classifiers as below can be applied but StochasticDualCoordinateAscentClassifier was the best classifier for this program
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            //pipeline.Add(new LogisticRegressionClassifier());
            //pipeline.Add(new NaiveBayesClassifier());
            //pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 5 });

            //Converting vectors back to normal 
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            //Start training
            PredictionModel<ClassificationData, ClassPrediction> model = pipeline.Train<ClassificationData, ClassPrediction>();
            await model.WriteAsync(modelPath);
            return model;
        }

        /// Evaluates the trained model for quality assurance against a second independent test data set.
        internal static void Evaluate(
            PredictionModel<ClassificationData, ClassPrediction> model,
            string testDatafile)
        {
            //Loading test data
            var testData = new TextLoader(testDatafile).CreateFrom<ClassificationData>();

            //Deciding which evaluator will be used
            var evaluator = new ClassificationEvaluator();

            //Start evaluation
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("------------------------------------------");
            Console.WriteLine("  Accuracy Macro: {0:P2}", metrics.AccuracyMacro);
            Console.WriteLine("  Accuracy Micro: {0:P2}", metrics.AccuracyMicro);
            Console.WriteLine("         LogLoss: {0:P2}", metrics.LogLoss);
            Console.WriteLine();
            Console.WriteLine(" PerClassLogLoss:");
            for (int i = 0; i < metrics.PerClassLogLoss.Length; i++)
                Console.WriteLine("       Class: {0} - {1:P2}", i, metrics.PerClassLogLoss[i]);
        }
    }
}
