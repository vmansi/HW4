# HW4
Aim: We want to build a chatbot that can have conversation on day to day topics in Star Trek manner.

Dataset: We tried 2 approaches:

1. Train a Seq2Seq model on Star Trek dataset.

2. Train a Seq2Seq model on Cornell Movie datset. We also created a datset of 120 very commonly used dialogs in Star Trek series. The Seq2Seq model generates a reply to the input post. The Star Trek datset is used to find a dialog that has the maximum liklihood (unigrams)
of follwoing the dialog the Seq2Seq model generates. 
