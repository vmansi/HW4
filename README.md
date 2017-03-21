# HW4
Aim: 

We want to build a chatbot that can have conversation on day to day topics in Star Trek manner. This is continued as our final project at https://github.com/GJena/CIS-700-7_Chatbot-Project

Dataset: We tried 2 approaches:

1.Train a Seq2Seq model on Star Trek dataset. SInce we had small dataset, we tried including context, and augmenting the data.
So, if the conversation flow was  P1 -> P2 -> P1 -> P3, the post reply pairs that were generated were as follows:

P1, P2

P2, P3

P1.P2 , P3

P3, P4

P2,P3, P4

2.Train a Seq2Seq model on Cornell Movie datset. We also created a datset of 120 very commonly used dialogs in Star Trek series. The Seq2Seq model generates a reply to the input post. The Star Trek datset is used to find a dialog that has the maximum liklihood (unigrams) of follwoing the dialog the Seq2Seq model generates. Not good results. (A few sentences getting repeated.)

Tried to take 5 most liklely sentences, and then chose 1 at random.

Also, tried to build a very high quality Star Trek dialogs set, containing dialogs that are very generic and might go along with any sentence.

These things improved the final output. But still not satisfactory.

Model Evaluation:

We created a dialog set containing 15 dialogs, and tested both models on that. 

Train and Test Loss:

From the last assignment, we found out that train and cross validation loss are not good metric of evaluating chatbot. Even here, although both train and test loss kept decresing, after a point of time, our chatbot strted giving wierd replies.

So we kept a set of 10 dialogs and kept tesing our model after few hours with them. The checkpoint that generated best replies was saved.

For the model which was trained using Star Trek dataset, the perplexity was lower since there was repetition in data. 

Train Perplexity: 7.01

Cross Validation Perplexity:

Bucket 0: 6.12
 
Bucket 1: 8.38

Bucket 2: 9.59

Bucket 3: 15.26


For the model which was trained using Cornell Movie dataset:

Train Perplexity: 12.86

Cross Validation Perplexity:

Bucket 0: 60.12
 
Bucket 1: 8.38

Bucket 2: 9.59

Bucket 3: 15.26


