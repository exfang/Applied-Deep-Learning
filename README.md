# Applied Deep Learning

Computer vision and natural language processing are two areas that have seen great advances due to deep learning.

This project has 2 parts:
1. Find a suitable computer vision dataset to train a deep learning model. You will (i) build and train your own deep learning model (ii) train another deep learning model (of your choice) using transfer learning. You are also expected to compare and comment on the performance of both models.
2. Use the Twitter US airline sentiment data to build a text classifier. The dataset for Part 2 is modified from https://www.kaggle.com/datasets/crowdflower/twitter-airlinesentiment?resource=download. You will train your own deep learning model with this dataset. Additionally, you should look for another sentiment dataset and use this dataset to train a second deep learning model with the same architecture/layers as the first model. You are expected to comment on the performance of both models on the airline tweets dataset.

# Part 1

## Business Scenario

In the heart of Africa, the majestic rhinoceroses and elephants are under constant threat from poachers seeking to profit from their ivory horns and tusks. To combat this crisis and protect these magnificent creatures, I have developed an image classification system. This system will use advanced AI algorithms to detect and identify rhinos and elephants in real-time, allowing for quicker response and protection against poaching activities.

Objective: The primary objective of this project is to create a robust image classification system capable of identifying rhinoceroses and elephants from photographs and video footage. By accurately recognizing these animals in their natural habitat, wildlife conservationists and authorities can respond swiftly to any potential threats, such as poachers or natural disasters, and implement necessary protective measures.

## Model Information

- Tensorflow Keras CNN Model (For building from scratch)
- Tensorflow Keras VGG-16 Model (For transfer learning)

## Model Outcome

Below is the model that has been built from scratch. 

![image](https://github.com/exfang/Applied-Deep-Learning/assets/98097965/4275529c-fcdb-477d-8e8c-fc404b460916)

- The model did quite well in predicting whether an image is a rhino or elephant.
- The training and validation loss from 2.5 epochs and above flatlines around less than 1. This shows that the model isn't over or under-fitting since the loss is roughly similar.
- The training and validation accuracies are also increasing to more than 65% from 5 epochs onward.

Below uses Transfer Learning with VGG-16

![image](https://github.com/exfang/Applied-Deep-Learning/assets/98097965/605e9c3f-3b1e-4fc3-a994-0077a9d856c3)

- Looking at the pretrained model's loss and accuracy, it definitely performs better than my manually tuned model. This could be due to the fact that the model was already trained on a large number of animal data that could potentially include rhinos and elephants.
- Additionally, the model does not show any signs of overfitting as the loss and accuracy for training and validation are roughly similar.

Therefore, pretrained model is better at classification of the rhino/elephant.

# Part 2

## Business Scenario

In the current digital era, social media platforms have emerged as a popular forum for people to share their thoughts and experiences, including their dealings United airlines. Twitter, one of the most popular channels for in-the-moment updates and communication, is a priceless resource for opinions and sentiments on airline services. Text sentiment analysis of tweets about airlines can yield valuable information and benefits for both United Airlines' business and passengers.

- Customer Insights: Sentiment analysis enables United Airlines to obtain a deeper understanding of their customers' attitudes and feelings. United airlines can determine their strengths and areas for improvement by classifying tweets as good, negative, or neutral. For data-driven decision-making and improving their overall customer experience, understanding customer sentiment is essential.
- Reputation Management: United Airlines is aware of the importance of maintaining a positive internet reputation. An airline's reputation can suffer and potential customers may be turned off by negative comments on social media. Sentiment analysis offers a proactive method for handling problems swiftly, reducing bad press, and successfully managing brand impression.
- Enhancing Customer Satisfaction: United Airlines can raise overall customer satisfaction by proactively addressing problems discovered through sentiment research. Customers who are satisfied are more inclined to stick with a brand and recommend it to others.

In conclusion, United Airlines' business can benefit greatly from sentiment analysis of tweets mentioning airlines. It offers practical insights that help improve client satisfaction, brand reputation, and competitiveness in a market that is continually changing. Airlines can keep aware of client sentiment, adjust to shifting tastes, and make data-driven decisions that are advantageous to both customers and the airline itself by utilizing sentiment analysis.

## Model Information

- Tensorflow Keras Sequential Model making use of LSTM

## Model Outcome

![image](https://github.com/exfang/Applied-Deep-Learning/assets/98097965/64d1cfd6-da27-4334-b523-d3ebc23bbf90)

Having tried several model layer tunings, my best model is as above. Although the model is not the best and there is clear overfitting, it still performs slightly better than random guessing at above 60% accuracy.

Prior to obtaining this result, I had tried several hyperparameter tunings:

1. Adding More Units in Bidirectional and Dense Layers:
- Reason: Increasing the number of units in these layers can potentially help the model capture more complex patterns in the data.

2. Adding More Bidirectional and Dense Layers:
- Reason: Stacking more layers can increase the model's capacity to learn hierarchical features from the data. Deeper networks are often better at capturing intricate patterns.

3. Adding L2 Regularization to Dense Layers:
- Reason: L2 regularization helps prevent overfitting by adding a penalty term to the loss function that discourages large weights.

4. Using Dropout and SpatialDropout1D:
- Reason: Dropout and SpatialDropout1D are regularization techniques that randomly drop a fraction of units during training. This helps prevent co-adaptation of neurons and can reduce overfitting by providing a form of model averaging.

5. Adding Batch Normalization:
- Reason: Batch normalization normalizes the activations within a layer during training, which can help stabilize and speed up training. It may improve the model's convergence and generalization.

6. Adjusting the Vocabulary Size:
- Reason: The vocabulary size determines the granularity of text representations. Increasing it allows the model to capture more unique words and can help balance representation quality and model complexity.

7. Adjusting learning rate
- Reason: Learning rate affects the speed and stability of training. I adjust it to help the model find the right balance between fast convergence and avoiding overshooting the optimal weights.

![image](https://github.com/exfang/Applied-Deep-Learning/assets/98097965/1de72984-53d7-48fb-a171-09f48ce01d71)

Trying the same model on Amazon text reviews. Just like the tweet data, the model is overfitted as the training and validation loss are very different.

Conclusion: 

- The model works terribly after being trained on the new dataset. This could be due to the difference in topics and words used.
- Seen in the classification matrix, the f1 score for all classes are around 40%, showing the model's inability to work well with the 3 sentiment classes.
- This shows that building task specific models may be better than reusing models trained for other tasks.
