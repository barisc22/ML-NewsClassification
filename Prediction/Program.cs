using Microsoft.ML;
using Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Prediction
{
    class Program2
    {
        static readonly string[] classNames = { "Business", "Entertainment", "Politics", "Tech", "Sport" };

        static readonly IEnumerable<ClassificationData> predictSentimentsData = new[]
        {
            new ClassificationData
            {
                News = "Dubai’s financial regulator is investigating allegations of mismanagement at private equity firm Abraaj, which is on the verge of financial collapse after a scandal over its use of investor money, two sources familiar with the matter said." //Business
            },
            new ClassificationData
            {
                News = "Seven-goal classics, unbelievable comebacks and England winning a penalty shootout - one of the best World Cups of recent times enters the quarter-final stage on Friday." //Sports
            },
            new ClassificationData
            {
                News = "Thousands of people watched a film posted in its entirety to YouTube by its US distributor before the apparent mistake was tackled."//Tech
            },
            new ClassificationData
            {
                News = "President Donald Trump has a good point when he rages about America's NATO partners failing to meet even their own burden sharing targets for the alliance's common defense." //Politics
            },
            new ClassificationData
            {
                News = "Attorney General Jeff Sessions on Tuesday withdrew a number of policy guidance documents from past administrations related to immigration that he found unnecessary, outdated, inconsistent with existing law, or otherwise improper." //Politics
            },
            new ClassificationData
            {
                News = "Ireland signed off from their Grand Slam season with one last high as they edged a tense decider 20-16 in Sydney to claim a first series win in Australia since 1979." //Sport
            },
            new ClassificationData
            {
                News = "Elon Musk's electric car firm Tesla, for instance, has already highlighted just how important the Chinese market is to it.But it imports all of its products to China and so would see a 25% tariff placed on its cars sold in China - on top of the 15% tax imported vehicles already face there. This would inevitably push up prices for Tesla in China, making its vehicles less competitive than they already are, relative to others.." //Business
            },
            new ClassificationData
            {
                News = "One of China's biggest technology companies has declared it has begun mass production of a self-driving bus.." //Tech
            },
            new ClassificationData
            {
                News = "It is recounted in a new book by New York Times journalist Amy Chozick, who said Ms Clinton was concerned at the time that voters did not perceive her as authentic, and was also intent on keeping her personal dislike of Mr Trump out of her public appearances." //Politics
            }
        };

        const string modelPath = @".\Learned\Model.zip";

        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                //Start the prediction
                //If you want to add new sententences, just change one sentence from above
                var model = await PredictAsync(modelPath, classNames, predictSentimentsData);

                Console.WriteLine("Press any key to end program...");
                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        //Prediction class for the sentences above
        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>> PredictAsync(
            string modelPath, //in the Learned folder
            string[] classNames,
            IEnumerable<ClassificationData> predicts = null,
            PredictionModel<ClassificationData, ClassPrediction> model = null){

            if (model == null){
                model = await PredictionModel.ReadAsync<ClassificationData, ClassPrediction>(modelPath); //Wait for the model
            }

            //Making predictions
            IEnumerable<ClassPrediction> predictions = model.Predict(predicts);

            //Printing the predictions
            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (sentiment, prediction) to use in below 
            IEnumerable<(ClassificationData sentiment, ClassPrediction prediction)> sentimentsAndPredictions =
                predicts.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                string textDisplay = item.sentiment.News;
                if (textDisplay.Length > 100)
                    textDisplay = textDisplay.Substring(0, 100) + "...";

                string predictedClass = classNames[(uint)item.prediction.Class]; //Name of the class
                
                Console.WriteLine("Prediction: {0}-{1} | Test: '{2}'",
                    item.prediction.Class, predictedClass, textDisplay); //Class number - Name of the class - News
            }
            Console.WriteLine();
            return model;
        }
    }
}
