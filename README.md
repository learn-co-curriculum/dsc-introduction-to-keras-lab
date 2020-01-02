
# Keras - Lab

## Introduction

In this lab you'll once again build a neural network, but this time you will be using Keras to do a lot of the heavy lifting.


## Objectives

You will be able to:

- Build a neural network using Keras 
- Evaluate performance of a neural network using Keras 

## Required Packages

We'll start by importing all of the required packages and classes.


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import optimizers
```

    Using TensorFlow backend.


## Load the data

In this lab you will be classifying bank complaints available in the `'Bank_complaints.csv'` file. 


```python
# Import data
df = None

# Inspect data
print(df.info())
df.head()
```

As mentioned earlier, your task is to categorize banking complaints into various predefined categories. Preview what these categories are and what percent of the complaints each accounts for. 


```python
# Your code here
```

## Preprocessing

Before we build our neural network, we need to do several preprocessing steps. First, we will create word vector counts (a bag of words type representation) of our complaints text. Next, we will change the category labels to integers. Finally, we will perform our usual train-test split before building and training our neural network using Keras. With that, let's start munging our data! 

## One-hot encoding of the complaints

Our first step again is to transform our textual data into a numerical representation. As we saw in some of our previous lessons on NLP, there are many ways to do this. Here, we'll use the `Tokenizer()` class from the `preprocessing.text` sub-module of the Keras package.   

As with our previous work using NLTK, this will transform our text complaints into word vectors. (Note that the method of creating a vector is different from our previous work with NLTK; as you'll see, word order will be preserved as opposed to a bag of words representation). In the below code, we'll only keep the 2,000 most common words and use one-hot encoding. 




```python
# As a quick preliminary, briefly review the docstring for keras.preprocessing.text.Tokenizer
Tokenizer?
```


```python
# ⏰ This cell may take about thirty seconds to run

# Raw text complaints
complaints = df['Consumer complaint narrative'] 

# Initialize a tokenizer 
tokenizer = Tokenizer(num_words=2000) 

# Fit it to the complaints
tokenizer.fit_on_texts(complaints) 

# Generate sequences
sequences = tokenizer.texts_to_sequences(complaints) 
print('sequences type:', type(sequences))

# Similar to sequences, but returns a numpy array
one_hot_results= tokenizer.texts_to_matrix(complaints, mode='binary') 
print('one_hot_results type:', type(one_hot_results))

# Useful if we wish to decode (more explanation below)
word_index = tokenizer.word_index 

# Tokens are the number of unique words across the corpus
print('Found %s unique tokens.' % len(word_index)) 

# Our coded data
print('Dimensions of our coded results:', np.shape(one_hot_results)) 
```

    sequences type: <class 'list'>
    one_hot_results type: <class 'numpy.ndarray'>
    Found 50110 unique tokens.
    Dimensions of our coded results: (60000, 2000)


## Decoding Word Vectors 

As a note, you can also decode these vectorized representations of the reviews. The `word_index` variable, defined above, stores the mapping from the label number to the actual word. Somewhat tediously, we can turn this dictionary inside out and map it back to our word vectors, giving us roughly the original complaint back. (As you'll see, the text won't be identical as we limited ourselves to top 2000 words.)

## Python Review / Mini Challenge

While a bit tangential to our main topic of interest, we need to reverse our current dictionary `word_index` which maps words from our corpus to integers. In decoding our `one_hot_results`, we will need to create a dictionary of these integers to the original words. Below, take the `word_index` dictionary object and change the orientation so that the values are keys and the keys values. In other words, you are transforming something of the form {A:1, B:2, C:3} to {1:A, 2:B, 3:C}. 


```python
# Your code here
reverse_index = None
```

## Back to Decoding Our Word Vectors...


```python
comment_idx_to_preview = 19
print('Original complaint text:')
print(complaints[comment_idx_to_preview])
print('\n\n')

# The reverse_index cell block above must be complete in order for this cell block to successively execute 
decoded_review = ' '.join([reverse_index.get(i) for i in sequences[comment_idx_to_preview]])
print('Decoded review from Tokenizer:')
print(decoded_review)
```

    Original complaint text:
    I have already filed several complaints about AES/PHEAA. I was notified by a XXXX XXXX let @ XXXX, who pretended to be from your office, he said he was from CFPB. I found out this morning he is n't from your office, but is actually works at XXXX. 
    
    This has wasted weeks of my time. They AES/PHEAA confirmed and admitted ( see attached transcript of XXXX, conversation at XXXX ( XXXX ) with XXXX that proves they verified the loans are not mine ) the student loans they had XXXX, and collected on, and reported negate credit reporting in my name are in fact, not mine. 
    They conclued their investigation on XXXX admitting they made a mistake and have my name on soneone elses loans. I these XXXX loans total {$10000.00}, original amount. My XXXX loans I got was total {$3500.00}. We proved by providing AES/PHEAA, this with my original promissary notes I located recently, the XXXX of my college provided AES/PHEAA with their original shoeinf amounts of my XXXX loans which show different dates and amounts, the dates and amounts are not even close to matching these loans they have in my name, The original lender, XXXX XXXX Bank notifying AES/PHEAA, they never issued me a student loan, and original Loan Guarantor, XXXX, notifying AES/PHEAA, they never were guarantor of my loans. 
    
    XXXX straight forward. But today, this person, XXXX XXXX, told me they know these loans are not mine, and they refuse to remove my name off these XXXX loan 's and correct their mistake, essentially forcing me to pay these loans off, bucause in XXXX they sold the loans to XXXX loans. 
    
    This is absurd, first protruding to be this office, and then refusing to correct their mistake. 
    
    Please for the love of XXXX will soneone from your office call me at XXXX, today. I am a XXXX vet and they are knowingly discriminating against me. 
    Pretending to be you.
    
    
    
    Decoded review from Tokenizer:
    i have already filed several complaints about aes i was notified by a xxxx xxxx let xxxx who to be from your office he said he was from cfpb i found out this morning he is n't from your office but is actually works at xxxx this has weeks of my time they aes confirmed and admitted see attached of xxxx conversation at xxxx xxxx with xxxx that they verified the loans are not mine the student loans they had xxxx and on and reported credit reporting in my name are in fact not mine they their investigation on xxxx they made a mistake and have my name on loans i these xxxx loans total 10000 00 original amount my xxxx loans i got was total 00 we by providing aes this with my original notes i located recently the xxxx of my college provided aes with their original amounts of my xxxx loans which show different dates and amounts the dates and amounts are not even close to these loans they have in my name the original lender xxxx xxxx bank notifying aes they never issued me a student loan and original loan xxxx notifying aes they never were of my loans xxxx forward but today this person xxxx xxxx told me they know these loans are not mine and they refuse to remove my name off these xxxx loan 's and correct their mistake essentially me to pay these loans off in xxxx they sold the loans to xxxx loans this is first to be this office and then refusing to correct their mistake please for the of xxxx will from your office call me at xxxx today i am a xxxx and they are against me to be you


## Convert the Products to Numerical Categories

On to step two of our preprocessing: converting our descriptive categories into integers.


```python
product = df['Product']

# Initialize
le = preprocessing.LabelEncoder() 
le.fit(product)
print('Original class labels:')
print(list(le.classes_))
print('\n')
product_cat = le.transform(product)  

# If you wish to retrieve the original descriptive labels post production
# list(le.inverse_transform([0, 1, 3, 3, 0, 6, 4])) 

print('New product labels:')
print(product_cat)
print('\n')

# Each row will be all zeros except for the category for that observation 
print('One hot labels; 7 binary columns, one for each of the categories.') 
product_onehot = to_categorical(product_cat)
print(product_onehot)
print('\n')

print('One hot labels shape:')
print(np.shape(product_onehot))
```

    Original class labels:
    ['Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Credit card', 'Credit reporting', 'Mortgage', 'Student loan']
    
    
    New product labels:
    [6 6 6 ... 4 4 4]
    
    
    One hot labels; 7 binary columns, one for each of the categories.
    [[0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     ...
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]]
    
    
    One hot labels shape:
    (60000, 7)


## Train-test split

Now for our final preprocessing step: the usual train-test split. 


```python
random.seed(123)
test_index = random.sample(range(1,10000), 1500)

test = one_hot_results[test_index]
train = np.delete(one_hot_results, test_index, 0)

label_test = product_onehot[test_index]
label_train = np.delete(product_onehot, test_index, 0)

print('Test label shape:', np.shape(label_test))
print('Train label shape:', np.shape(label_train))
print('Test shape:', np.shape(test))
print('Train shape:', np.shape(train))
```

    Test label shape: (1500, 7)
    Train label shape: (58500, 7)
    Test shape: (1500, 2000)
    Train shape: (58500, 2000)


## Building the network

Let's build a fully connected (Dense) layer network with relu activation in Keras. You can do this using: `Dense(16, activation='relu')`. 

In this example, use two hidden layers with 50 units in the first layer and 25 in the second, both with a `'relu'` activation function. Because we are dealing with a multiclass problem (classifying the complaints into 7 categories), we use a use a `'softmax'` classifier in order to output 7 class probabilities per case.  


```python
# Initialize a sequential model
model = None

# Two layers with relu activation



# One layer with softmax activation 

```

## Compiling the model

Now, compile the model! This time, use `'categorical_crossentropy'` as the loss function and stochastic gradient descent, `'SGD'` as the optimizer. As in the previous lesson, include the accuracy as a metric.


```python
# Compile the model

```

## Training the model

In the compiler, you'll be passing the optimizer (SGD = stochastic gradient descent), loss function, and metrics. Train the model for 120 epochs in mini-batches of 256 samples.

_Note:_ ⏰ _Your code may take about one to two minutes to run._


```python
# Train the model 
history = None
```

Recall that the dictionary `history` has two entries: the loss and the accuracy achieved using the training set.


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'acc'])



## Plot the results

As you might expect, we'll use our `matplotlib` for graphing. Use the data stored in the `history_dict` above to plot the loss vs epochs and the accuracy vs epochs. 


```python
# Plot the loss vs the number of epoch
```


```python
# Plot the training accuracy vs the number of epochs

```

It seems like we could just keep on going and accuracy would go up!

## Make predictions

Finally, it's time to make predictions. Use the relevant method discussed in the previous lesson to output (probability) predictions for the test set.


```python
# Output (probability) predictions for the test set 
y_hat_test = None
```

## Evaluate Performance

Finally, print the loss and accuracy for both the train and test sets of the final trained model.


```python
# Print the loss and accuracy for the training set 
results_train = None
results_train
```


```python
# Print the loss and accuracy for the test set 
results_test = None
results_test
```

We can see that the training set results are really good, but the test set results lag behind. We'll talk a little more about this in the next lesson, and discuss how we can get better test set results as well!


## Additional Resources 

- https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb 
- https://catalog.data.gov/dataset/consumer-complaint-database 

## Summary 

Congratulations! In this lab, you built a neural network thanks to the tools provided by Keras! In upcoming lessons and labs we'll continue to investigate further ideas regarding how to tune and refine these models for increased accuracy and performance.
