## Final Project Proposal


Overview: Inspired by the Naive Bayes MP, for this final project, I plan to implement some more advanced
machine learning models to improve the predicting accuracy of digits classification, and do something interesting if time allowed. For now, I plan to implement a discriminative model for classification,
maybe I'll choose from multinomial logistic regression, SVM and Deep neural networks. Furthermore, I'd also like to implement a generative model (maybe Generative Adversarial Nets, GAN) to simulate the handwritten style of digits
and produce some high-quality fake images of digits. This would be meaningful because it can provide more useful training data.

Motivation: I indeed have some background knowledge and interest in machine learning models. Since I've just taken CS 446 last semester, these models are still
clearly in my mind. Also, I've implemented these models in Python before, so I think it would also provide some guide when I implement it in C++. 

Rough timeline: 

- Week1: Implement loading data, training model, etc.

- Week2: Implement classification of the trained model, and evaluate model performance based on accuracy on validation set. (maybe involve more evaluating tricks like F-score, whether overfitting or underfitting, etc), and connect the implementation to cinder visualization.

- Week3: Implement GAN to generate some interesting fake image of digits.

- Week4: Put everything together, and maybe explore more interesting things like improving the "lying" ability of GAN, introduce some techniques to avoid overfitting, etc.

Further exploration: If the above goals were finished earlier than expected, I would try using some ensemble methods like combining classifiers (bagging, random forests, Adaboost) to improve the model's classification performance. 

