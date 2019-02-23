namespace Models
{
    using Microsoft.ML.Runtime.Api;

    /// This is the input dataset class and has a float (Sentiment) that has
    /// a value for sentiment of either positive or negative, and a string for
    /// the comment.
    /// 
    /// Both fields have Column attributes attached to them.
    /// This attribute describes the order of each field in the data file,
    /// and which is the Label field.
    public class ClassificationData
    {
        [Column(ordinal: "0", name: "Label")]
        public float newsCategory;

        [Column(ordinal: "1")]
        public string News;
    }
}
