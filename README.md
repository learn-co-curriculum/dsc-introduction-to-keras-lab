# Keras - Lab

## Introduction

In this lab you'll once again build a neural network, but this time you will be using Keras to do a lot of the 
heavy lifting.


## Objectives

You will be able to:

- Build a neural network using Keras 
- Evaluate performance of a neural network using Keras 

**Keep in Mind:** Keras provide verbose (detailed) outputs that explain how it is using the hardware in your computer to run your neural network. They look similar to the message in the image below. The appearance of these warnings is not an indication that the code is broken. 

<table>
    <tbody>
        <tr>
            <td><img src="https://curriculum-content.s3.amazonaws.com/data-science/images/images/keras-warning.png" alt="This is the alt-text for the image." height="350/" /></td>
        </tr>
    </tbody>
</table>

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

## Load the data

In this lab you will be classifying bank complaints available in the `'Bank_complaints.csv'` file. 


```python
# Import data
df = pd.read_csv('Bank_complaints.csv')

# Inspect data
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60000 entries, 0 to 59999
    Data columns (total 2 columns):
     #   Column                        Non-Null Count  Dtype 
    ---  ------                        --------------  ----- 
     0   Product                       60000 non-null  object
     1   Consumer complaint narrative  60000 non-null  object
    dtypes: object(2)
    memory usage: 937.6+ KB
    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product</th>
      <th>Consumer complaint narrative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Student loan</td>
      <td>In XX/XX/XXXX I filled out the Fedlaon applica...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Student loan</td>
      <td>I am being contacted by a debt collector for p...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Student loan</td>
      <td>I cosigned XXXX student loans at SallieMae for...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Student loan</td>
      <td>Navient has sytematically and illegally failed...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Student loan</td>
      <td>My wife became eligible for XXXX Loan Forgiven...</td>
    </tr>
  </tbody>
</table>
</div>



As mentioned earlier, your task is to categorize banking complaints into various predefined categories. Preview what these categories are and what percent of the complaints each accounts for. 


```python
df['Product'].value_counts(normalize=True)
```




    Product
    Student loan                   0.190067
    Credit card                    0.159000
    Consumer Loan                  0.157900
    Mortgage                       0.138867
    Bank account or service        0.138483
    Credit reporting               0.114400
    Checking or savings account    0.101283
    Name: proportion, dtype: float64



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
reverse_index = dict([(value, key) for (key, value) in word_index.items()])
```

## Back to Decoding Our Word Vectors...


```python
comment_idx_to_preview = 19
print('Original complaint text:')
print(complaints[comment_idx_to_preview])
print('\n\n')

#The reverse_index cell block above must be complete in order for this cell block to successively execute.
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
model = models.Sequential()

# Two layers with relu activation
model.add(layers.Dense(50, activation='relu', input_shape=(2000,)))
model.add(layers.Dense(25, activation='relu'))

# One layer with softmax activation 
model.add(layers.Dense(7, activation='softmax'))
```

    Metal device set to: Apple M1 Pro


    2023-04-21 15:25:53.968193: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2023-04-21 15:25:53.969008: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)


## Compiling the model

Now, compile the model! This time, use `'categorical_crossentropy'` as the loss function and stochastic gradient descent, `'SGD'` as the optimizer. As in the previous lesson, include the accuracy as a metric.


```python
# Compile the model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['acc'])
```

## Training the model

In the compiler, you'll be passing the optimizer (SGD = stochastic gradient descent), loss function, and metrics. Train the model for 120 epochs in mini-batches of 256 samples.

_Note:_ ⏰ _Your code may take about one to two minutes to run._


```python
# Train the model 
history = model.fit(train,
                    label_train,
                    epochs=120,
                    batch_size=256)
```

    Epoch 1/120


    2023-04-21 15:25:55.390710: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
    2023-04-21 15:25:55.525203: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    229/229 [==============================] - 4s 10ms/step - loss: 1.8846 - acc: 0.2323
    Epoch 2/120
    229/229 [==============================] - 2s 9ms/step - loss: 1.6618 - acc: 0.4029
    Epoch 3/120
    229/229 [==============================] - 2s 9ms/step - loss: 1.3276 - acc: 0.5922
    Epoch 4/120
    229/229 [==============================] - 2s 10ms/step - loss: 1.0392 - acc: 0.6776
    Epoch 5/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.8629 - acc: 0.7130
    Epoch 6/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.7652 - acc: 0.7332
    Epoch 7/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.7074 - acc: 0.7458
    Epoch 8/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.6690 - acc: 0.7562
    Epoch 9/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.6415 - acc: 0.7643
    Epoch 10/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.6202 - acc: 0.7722
    Epoch 11/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.6028 - acc: 0.7788
    Epoch 12/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5881 - acc: 0.7839
    Epoch 13/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5752 - acc: 0.7890
    Epoch 14/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5638 - acc: 0.7931
    Epoch 15/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5540 - acc: 0.7974
    Epoch 16/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5442 - acc: 0.8016
    Epoch 17/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.5363 - acc: 0.8052
    Epoch 18/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5281 - acc: 0.8085
    Epoch 19/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.5210 - acc: 0.8102
    Epoch 20/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.5142 - acc: 0.8136
    Epoch 21/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.5081 - acc: 0.8163
    Epoch 22/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.5022 - acc: 0.8188
    Epoch 23/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4967 - acc: 0.8205
    Epoch 24/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4912 - acc: 0.8228
    Epoch 25/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4860 - acc: 0.8247
    Epoch 26/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4817 - acc: 0.8264
    Epoch 27/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4773 - acc: 0.8297
    Epoch 28/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4730 - acc: 0.8294
    Epoch 29/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4692 - acc: 0.8311
    Epoch 30/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4652 - acc: 0.8330
    Epoch 31/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4616 - acc: 0.8349
    Epoch 32/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.4578 - acc: 0.8362
    Epoch 33/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.4548 - acc: 0.8373
    Epoch 34/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4514 - acc: 0.8382
    Epoch 35/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4484 - acc: 0.8399
    Epoch 36/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4454 - acc: 0.8410
    Epoch 37/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4427 - acc: 0.8426
    Epoch 38/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.4396 - acc: 0.8434
    Epoch 39/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4370 - acc: 0.8445
    Epoch 40/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4342 - acc: 0.8461
    Epoch 41/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4317 - acc: 0.8457
    Epoch 42/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4292 - acc: 0.8481
    Epoch 43/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4271 - acc: 0.8489
    Epoch 44/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4244 - acc: 0.8500
    Epoch 45/120
    229/229 [==============================] - 2s 11ms/step - loss: 0.4224 - acc: 0.8501
    Epoch 46/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4201 - acc: 0.8507
    Epoch 47/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4181 - acc: 0.8513
    Epoch 48/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.4159 - acc: 0.8523
    Epoch 49/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.4138 - acc: 0.8529
    Epoch 50/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4120 - acc: 0.8534
    Epoch 51/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4105 - acc: 0.8552
    Epoch 52/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4084 - acc: 0.8551
    Epoch 53/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4064 - acc: 0.8559
    Epoch 54/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4047 - acc: 0.8562
    Epoch 55/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4028 - acc: 0.8573
    Epoch 56/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.4012 - acc: 0.8571
    Epoch 57/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3995 - acc: 0.8575
    Epoch 58/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3977 - acc: 0.8588
    Epoch 59/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3964 - acc: 0.8591
    Epoch 60/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3947 - acc: 0.8595
    Epoch 61/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3928 - acc: 0.8609
    Epoch 62/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3917 - acc: 0.8609
    Epoch 63/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3899 - acc: 0.8615
    Epoch 64/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3885 - acc: 0.8624
    Epoch 65/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3874 - acc: 0.8627
    Epoch 66/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3860 - acc: 0.8623
    Epoch 67/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3844 - acc: 0.8632
    Epoch 68/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3831 - acc: 0.8635
    Epoch 69/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3815 - acc: 0.8647
    Epoch 70/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3802 - acc: 0.8640
    Epoch 71/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3789 - acc: 0.8651
    Epoch 72/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3777 - acc: 0.8654
    Epoch 73/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3763 - acc: 0.8655
    Epoch 74/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3750 - acc: 0.8665
    Epoch 75/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3736 - acc: 0.8669
    Epoch 76/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3726 - acc: 0.8670
    Epoch 77/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3713 - acc: 0.8677
    Epoch 78/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3701 - acc: 0.8679
    Epoch 79/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3690 - acc: 0.8685
    Epoch 80/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3679 - acc: 0.8690
    Epoch 81/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3664 - acc: 0.8689
    Epoch 82/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3656 - acc: 0.8699
    Epoch 83/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3641 - acc: 0.8700
    Epoch 84/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3630 - acc: 0.8706
    Epoch 85/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3620 - acc: 0.8709
    Epoch 86/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3610 - acc: 0.8716
    Epoch 87/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3597 - acc: 0.8724
    Epoch 88/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3587 - acc: 0.8716
    Epoch 89/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3578 - acc: 0.8720
    Epoch 90/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3565 - acc: 0.8731
    Epoch 91/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3556 - acc: 0.8726
    Epoch 92/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3543 - acc: 0.8735
    Epoch 93/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3530 - acc: 0.8739
    Epoch 94/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3520 - acc: 0.8747
    Epoch 95/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3514 - acc: 0.8744
    Epoch 96/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3502 - acc: 0.8752
    Epoch 97/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3492 - acc: 0.8757
    Epoch 98/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3480 - acc: 0.8763
    Epoch 99/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3470 - acc: 0.8761
    Epoch 100/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3458 - acc: 0.8770
    Epoch 101/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3452 - acc: 0.8766
    Epoch 102/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3440 - acc: 0.8777
    Epoch 103/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3429 - acc: 0.8777
    Epoch 104/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3421 - acc: 0.8783
    Epoch 105/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3408 - acc: 0.8786
    Epoch 106/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3400 - acc: 0.8792
    Epoch 107/120
    229/229 [==============================] - 2s 11ms/step - loss: 0.3386 - acc: 0.8796
    Epoch 108/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3379 - acc: 0.8797
    Epoch 109/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3370 - acc: 0.8803
    Epoch 110/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3357 - acc: 0.8804
    Epoch 111/120
    229/229 [==============================] - 2s 10ms/step - loss: 0.3349 - acc: 0.8809
    Epoch 112/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3337 - acc: 0.8816
    Epoch 113/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3330 - acc: 0.8816
    Epoch 114/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3316 - acc: 0.8826
    Epoch 115/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3309 - acc: 0.8825
    Epoch 116/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3298 - acc: 0.8828
    Epoch 117/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3285 - acc: 0.8833
    Epoch 118/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3276 - acc: 0.8835
    Epoch 119/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3273 - acc: 0.8842
    Epoch 120/120
    229/229 [==============================] - 2s 9ms/step - loss: 0.3260 - acc: 0.8847


Recall that the dictionary `history` has two entries: the loss and the accuracy achieved using the training set.


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'acc'])



## Plot the results

As you might expect, we'll use our `matplotlib` for graphing. Use the data stored in the `history_dict` above to plot the loss vs epochs and the accuracy vs epochs. 


```python
history_dict = history.history
loss_values = history_dict['loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'g', label='Training loss')

plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](index_files/index_28_0.png)
    


It seems like we could just keep on going and accuracy would go up!


```python
# Plot the training accuracy vs the number of epochs

acc_values = history_dict['acc'] 

plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```


    
![png](index_files/index_30_0.png)
    


## Make predictions

Finally, it's time to make predictions. Use the relevant method discussed in the previous lesson to output (probability) predictions for the test set.


```python
# Output (probability) predictions for the test set 
y_hat_test = model.predict(test) 
```

    47/47 [==============================] - 0s 2ms/step


    2023-04-21 15:30:12.058031: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


## Evaluate Performance

Finally, print the loss and accuracy for both the train and test sets of the final trained model.


```python
# Print the loss and accuracy for the training set 
results_train = model.evaluate(train, label_train)
results_train
```

      14/1829 [..............................] - ETA: 14s - loss: 0.0826 - acc: 0.9732

    2023-04-21 15:30:12.863364: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    1829/1829 [==============================] - 14s 8ms/step - loss: 0.3187 - acc: 0.8887





    [0.31868523359298706, 0.8887008428573608]




```python
# Print the loss and accuracy for the test set 
results_test = model.evaluate(test, label_test)
results_test
```

    47/47 [==============================] - 0s 8ms/step - loss: 0.2274 - acc: 0.9353





    [0.22738108038902283, 0.9353333115577698]



We can see that the training set results are really good, and the test set results seem to be even better. In general, this type of result will be rare, as train set results are usually at least a bit better than test set results.


## Additional Resources 

- https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb 
- https://catalog.data.gov/dataset/consumer-complaint-database 

## Summary 

Congratulations! In this lab, you built a neural network thanks to the tools provided by Keras! In upcoming lessons and labs we'll continue to investigate further ideas regarding how to tune and refine these models for increased accuracy and performance.
