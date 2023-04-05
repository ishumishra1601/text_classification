# text_classification

                                    
### The goal of this project is to build a text classification model using the Hugging Face library to classify a dataset of text into one of multiple categories. 

Dataset: https://www.kaggle.com/datasets/crawford/20-newsgroups 

This dataset is a collection newsgroup documents. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.
We have taken the following newsgroups as classes.
['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'sci.space', 'rec.sport.baseball']

### Pre-processing Steps: Following pre-processing steps have been done with the data
1.	Removed email-ids.
2.	Removed URLs
3.	Removed special characters.
4.	Removed stopwords

### Model Used: ‘distilbert-base-uncased’
DistilBERT is a smaller, distilled version of the popular BERT (Bidirectional Encoder Representations from Transformers) model. The "base-uncased" version of DistilBERT has 6 layers and 66 million parameters, compared to BERT's 12 layers and 110 million parameters. This reduction in size makes DistilBERT much faster and more efficient than BERT, while still maintaining a high level of accuracy on a variety of natural language processing tasks.

### The benefits of using DistilBERT include:
1. Faster inference: DistilBERT is much faster than BERT because of its smaller size, making it more practical for real-world applications where speed is a concern.
2. Reduced memory requirements: Because DistilBERT is smaller than BERT, it requires less memory to store and use, which can be a significant advantage in resource-constrained environments.
3. High accuracy: Despite its smaller size, DistilBERT performs almost as well as BERT on many natural language processing tasks, making it a good choice for applications where accuracy is crucial.
4. Transfer learning: Like BERT, DistilBERT is pre-trained on a large corpus of text and can be fine-tuned for specific NLP tasks with a relatively small amount of task-specific data, allowing for effective transfer learning.

### Evaluation of the model:

 ![image](https://user-images.githubusercontent.com/82132543/228598020-1affb270-19ec-4a3e-a648-63a13aeebfed.png)


### Possible ways to improve the performance of the DistilBERT model:

•	Data augmentation: Augmenting the training data with additional examples can also improve the model's performance. This can be done by adding noise, perturbing the data, or creating synthetic examples to increase the diversity of the training data.

•	Larger architecture: Increasing the size of the architecture can also improve the model's performance. This can be done by adding more layers or increasing the number of parameters in the model.

•	Ensemble methods: Combining multiple DistilBERT models or models of different architectures can improve performance. This can be done by training several models on the same task and then aggregating their predictions.

•	Domain-specific pre-training: Pre-training the model on a domain-specific corpus of text can improve its performance on tasks related to that domain. For example, pre-training the model on medical texts can improve its performance on medical NLP tasks.


### Sample predictions
 
 ![image](https://user-images.githubusercontent.com/82132543/228598265-7a1b3740-663f-4794-8574-3aa5530bd433.png)


### Observation

Here we can see that we have taken a sample from test dataset, and another we have given it a new sentence and our model has correctly predicted its category.




