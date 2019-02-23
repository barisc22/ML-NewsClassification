namespace Models
{
    using Microsoft.ML.Runtime.Api;

    /// This is the class used for prediction after the model has been trained.
    /// It has a single boolean and a PredictedLabel
    /// ColumnName attribute.
    /// 
    /// The Label is used to create and train the model, and it's also used with a
    /// second dataset to evaluate the model. The PredictedLabel is used during
    /// prediction and evaluation. For evaluation, an input with training data,
    /// the predicted values, and the model are used.
    public class ClassPrediction
    {
        [ColumnName("PredictedLabel")]
        public float Class;
    }
}
