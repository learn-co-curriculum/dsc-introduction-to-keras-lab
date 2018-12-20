
# Keras - Lab

## Introduction

In this lab you'll once again build a neural network but this time with much less production time since we will be using Keras to do a lot of the heavy lifting building blocks which we coded from hand previously.  Our use case will be classifying Bank complaints.


## Objectives

You will be able to:
* Build a neural network using Keras

## Loading Required Packages

Here we'll import all of the various packages that we'll use in this code along. We'll point out where these imports were used as they come up in the lab.


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

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


## Loading the data

As usual, we will start our data science process by importing the data itself.  
Load and preview as a pandas dataframe.   
The data is stored in a file **Bank_complaints.csv**.


```python
#Your code here
#import pandas as pd #As reference; already imported above

df = pd.read_csv('Bank_complaints.csv')
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60000 entries, 0 to 59999
    Data columns (total 2 columns):
    Product                         60000 non-null object
    Consumer complaint narrative    60000 non-null object
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



## Further Preview of the Categories

As we said, our task here is to categorize banking complaints into various predefined categories. Preview what these categories are and what percent of the complaints each accounts for.


```python
#Your code here
df["Product"].value_counts(normalize=True)
```




    Student loan                   0.190067
    Credit card                    0.159000
    Consumer Loan                  0.157900
    Mortgage                       0.138867
    Bank account or service        0.138483
    Credit reporting               0.114400
    Checking or savings account    0.101283
    Name: Product, dtype: float64



## Preprocessing

Before we build our neural network, we need to do several preprocessing steps. First, we will create word vector counts (a bag of words type representation) of our complaints text. Next, we will change the category labels to integers. Finally, we will perform our usual train-test split before building and training our neural network using Keras. With that, let's start munging our data!

## One-hot encoding of the complaints

Our first step again is to transform our textual data into a numerical representation. As we've started to see in some of our previous lessons on NLP, there are many ways to do this. Here, we'll use the `Tokenizer` method from the `preprocessing` module of the Keras package.   

As with our previous work using NLTK, this will transform our text complaints into word vectors. (Note that the method of creating a vector is different from our previous work with NLTK; as you'll see, word order will be preserved as oppossed to a bag of words representation. In the below code, we'll only keep the 2,000 most common words and use one-hot encoding.

Note that the code block below takes advantage of the following package import from our first code cell above.  
`from keras.preprocessing.text import Tokenizer`


```python
#As a quick preliminary, briefly review the docstring for the Keras.preprocessing.text.Tokenizer method:
Tokenizer?
```


```python
#Now onto the actual code recipe...
complaints = df["Consumer complaint narrative"] #Our raw text complaints

tokenizer = Tokenizer(num_words=2000) #Initialize a tokenizer.

tokenizer.fit_on_texts(complaints) #Fit it to the complaints

sequences = tokenizer.texts_to_sequences(complaints) #Generate sequences
print('sequences type:', type(sequences))

one_hot_results= tokenizer.texts_to_matrix(complaints, mode='binary') #Similar to sequences, but returns a numpy array
print('one_hot_results type:', type(one_hot_results))

word_index = tokenizer.word_index #Useful if we wish to decode (more explanation below)

print('Found %s unique tokens.' % len(word_index)) #Tokens are the number of unique words across the corpus


print('Dimensions of our coded results:', np.shape(one_hot_results)) #Our coded data
```

    sequences type: <class 'list'>
    one_hot_results type: <class 'numpy.ndarray'>
    Found 50110 unique tokens.
    Dimensions of our coded results: (60000, 2000)


## Decoding our Word Vectors
As a note, you can also decode these vectorized representations of the reviews. The `word_index` variable, defined above, stores the mapping from the label number to the actual word. Somewhat tediously, we can turn this dictionary inside out and map it back to our word vectors, giving us roughly the original complaint back. (As you'll see, the text won't be identical as we limited ourselves to 200 words.)

## Python Review / Mini Challenge

While a bit tangential to our main topic of interest, we need to reverse our current dictionary `word_index` which maps words from our corpus to integers. In decoding our one_hot_results, we will need to create a dictionary of these integers to the original words. Below, take the `word_index` dictionary object and change the orientation so that the values are keys and the keys values. In other words, you are transforming something of the form {A:1, B:2, C:3} to {1:A, 2:B, 3:C}


```python
#Your code here
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
product = df["Product"]

le = preprocessing.LabelEncoder() #Initialize. le used as abbreviation fo label encoder
le.fit(product)
print("Original class labels:")
print(list(le.classes_))
print('\n')
product_cat = le.transform(product)  
#list(le.inverse_transform([0, 1, 3, 3, 0, 6, 4])) #If you wish to retrieve the original descriptive labels post production

print('New product labels:')
print(product_cat)
print('\n')


print('One hot labels; 7 binary columns, one for each of the categories.') #Each row will be all zeros except for the category for that observation.
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


## Train - test split

Now for our final preprocessing step: the usual train-test split.


```python
import random
random.seed(123)
test_index = random.sample(range(1,10000), 1500)

test = one_hot_results[test_index]
train = np.delete(one_hot_results, test_index, 0)


label_test = product_onehot[test_index]
label_train = np.delete(product_onehot, test_index, 0)

print("Test label shape:", np.shape(label_test))
print("Train label shape:", np.shape(label_train))
print("Test shape:", np.shape(test))
print("Train shape:", np.shape(train))
```

    Test label shape: (1500, 7)
    Train label shape: (58500, 7)
    Test shape: (1500, 2000)
    Train shape: (58500, 2000)


## Building the network

Let's build a fully connected (Dense) layer network with relu activations in Keras. You can do this using: `Dense(16, activation='relu')`.

In this examples, use 2 hidden with 50 units in the first layer and 25 in the second, both with a `relu` activation function. Because we are dealing with a multiclass problem (classifying the complaints into 7 ), we use a use a softmax classifyer in order to output 7 class probabilities per case.  

The previous imports that you'll use here are:  

```from keras import models
from keras import layers```


```python
#Your code here; initialize a sequential model with 3 layers; 
#two hidden relu and the final classification output using softmax
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
```

## Compiling the model and look at the results

Now, compile the model! This time, use `'categorical_crossentropy'` as the loss function and stochastic gradient descent, `'SGD'` as the optimizer. As in the previous lesson, include the accuracy as a metric.


```python
#Your code here
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Training the model

In the compiler, you'll be passing the optimizer (SGD = stochastic gradient descent), loss function, and metrics. Train the model for 120 epochs in mini-batches of 256 samples.


```python
#Your code here
history = model.fit(train,
                    label_train,
                    epochs=120,
                    batch_size=256)
```

    Epoch 1/120
    58500/58500 [==============================] - 1s 19us/step - loss: 1.8774 - acc: 0.2442
    Epoch 2/120
    58500/58500 [==============================] - 1s 15us/step - loss: 1.6151 - acc: 0.4546
    Epoch 3/120
    58500/58500 [==============================] - 1s 16us/step - loss: 1.2645 - acc: 0.6099
    Epoch 4/120
    58500/58500 [==============================] - 1s 16us/step - loss: 1.0055 - acc: 0.6786
    Epoch 5/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.8523 - acc: 0.7135
    Epoch 6/120
    58500/58500 [==============================] - 1s 16us/step - loss: 0.7625 - acc: 0.7328
    Epoch 7/120
    58500/58500 [==============================] - 1s 16us/step - loss: 0.7068 - acc: 0.7449
    Epoch 8/120
    58500/58500 [==============================] - 1s 16us/step - loss: 0.6691 - acc: 0.7557
    Epoch 9/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.6414 - acc: 0.7640
    Epoch 10/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.6200 - acc: 0.7703
    Epoch 11/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.6024 - acc: 0.7771
    Epoch 12/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5875 - acc: 0.7819
    Epoch 13/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5747 - acc: 0.7871
    Epoch 14/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5634 - acc: 0.7926
    Epoch 15/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5531 - acc: 0.7963
    Epoch 16/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5440 - acc: 0.8009
    Epoch 17/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5353 - acc: 0.8044
    Epoch 18/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5275 - acc: 0.8081
    Epoch 19/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5202 - acc: 0.8103
    Epoch 20/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5135 - acc: 0.8124
    Epoch 21/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.5074 - acc: 0.8159
    Epoch 22/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.5015 - acc: 0.8186
    Epoch 23/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4959 - acc: 0.8208
    Epoch 24/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4906 - acc: 0.8232
    Epoch 25/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4858 - acc: 0.8249
    Epoch 26/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4809 - acc: 0.8270
    Epoch 27/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4768 - acc: 0.8286
    Epoch 28/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4725 - acc: 0.8309
    Epoch 29/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4685 - acc: 0.8315
    Epoch 30/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4650 - acc: 0.8339
    Epoch 31/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4611 - acc: 0.8349
    Epoch 32/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4577 - acc: 0.8362
    Epoch 33/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4544 - acc: 0.8374
    Epoch 34/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4513 - acc: 0.8389
    Epoch 35/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4483 - acc: 0.8391
    Epoch 36/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4454 - acc: 0.8409
    Epoch 37/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4425 - acc: 0.8417
    Epoch 38/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4400 - acc: 0.8424
    Epoch 39/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4375 - acc: 0.8438
    Epoch 40/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4349 - acc: 0.8438
    Epoch 41/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4325 - acc: 0.8462
    Epoch 42/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4302 - acc: 0.8462
    Epoch 43/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4277 - acc: 0.8482
    Epoch 44/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4256 - acc: 0.8479
    Epoch 45/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4236 - acc: 0.8488
    Epoch 46/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4214 - acc: 0.8489
    Epoch 47/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4193 - acc: 0.8508
    Epoch 48/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4175 - acc: 0.8511
    Epoch 49/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4156 - acc: 0.8519
    Epoch 50/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4134 - acc: 0.8528
    Epoch 51/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4121 - acc: 0.8529
    Epoch 52/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4103 - acc: 0.8534
    Epoch 53/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4085 - acc: 0.8546
    Epoch 54/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4071 - acc: 0.8541
    Epoch 55/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4052 - acc: 0.8555
    Epoch 56/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4040 - acc: 0.8562
    Epoch 57/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4022 - acc: 0.8564
    Epoch 58/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.4006 - acc: 0.8576
    Epoch 59/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3991 - acc: 0.8577
    Epoch 60/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3976 - acc: 0.8583
    Epoch 61/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3965 - acc: 0.8585
    Epoch 62/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3951 - acc: 0.8596
    Epoch 63/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3935 - acc: 0.8609
    Epoch 64/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3921 - acc: 0.8596
    Epoch 65/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3909 - acc: 0.8608
    Epoch 66/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3896 - acc: 0.8613
    Epoch 67/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3883 - acc: 0.8619
    Epoch 68/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3872 - acc: 0.8620
    Epoch 69/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3859 - acc: 0.8625
    Epoch 70/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3847 - acc: 0.8630
    Epoch 71/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3836 - acc: 0.8637
    Epoch 72/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3821 - acc: 0.8649
    Epoch 73/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3813 - acc: 0.8641
    Epoch 74/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3803 - acc: 0.8644
    Epoch 75/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3790 - acc: 0.8652
    Epoch 76/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3780 - acc: 0.8645
    Epoch 77/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3769 - acc: 0.8658
    Epoch 78/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3756 - acc: 0.8659
    Epoch 79/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3747 - acc: 0.8661
    Epoch 80/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3737 - acc: 0.8668
    Epoch 81/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3726 - acc: 0.8676
    Epoch 82/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3720 - acc: 0.8673
    Epoch 83/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3706 - acc: 0.8675
    Epoch 84/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3701 - acc: 0.8682
    Epoch 85/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3689 - acc: 0.8681
    Epoch 86/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3681 - acc: 0.8688
    Epoch 87/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3673 - acc: 0.8691
    Epoch 88/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3663 - acc: 0.8695
    Epoch 89/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3652 - acc: 0.8702
    Epoch 90/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3646 - acc: 0.8699
    Epoch 91/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3636 - acc: 0.8701
    Epoch 92/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3627 - acc: 0.8707
    Epoch 93/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3618 - acc: 0.8712
    Epoch 94/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3607 - acc: 0.8707
    Epoch 95/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3600 - acc: 0.8713
    Epoch 96/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3592 - acc: 0.8718
    Epoch 97/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3582 - acc: 0.8725
    Epoch 98/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3575 - acc: 0.8720
    Epoch 99/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3566 - acc: 0.8728
    Epoch 100/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3559 - acc: 0.8730
    Epoch 101/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3551 - acc: 0.8731
    Epoch 102/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3543 - acc: 0.8733
    Epoch 103/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3533 - acc: 0.8731
    Epoch 104/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3529 - acc: 0.8742
    Epoch 105/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3519 - acc: 0.8744
    Epoch 106/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3509 - acc: 0.8740
    Epoch 107/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3504 - acc: 0.8746
    Epoch 108/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3494 - acc: 0.8752
    Epoch 109/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3487 - acc: 0.8752
    Epoch 110/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3481 - acc: 0.8762
    Epoch 111/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3471 - acc: 0.8751
    Epoch 112/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3463 - acc: 0.8760
    Epoch 113/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3457 - acc: 0.8764
    Epoch 114/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3451 - acc: 0.8767
    Epoch 115/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3439 - acc: 0.8770
    Epoch 116/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.3438 - acc: 0.8771
    Epoch 117/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3425 - acc: 0.8775
    Epoch 118/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3419 - acc: 0.8781
    Epoch 119/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3410 - acc: 0.8781
    Epoch 120/120
    58500/58500 [==============================] - 1s 14us/step - loss: 0.3401 - acc: 0.8788


The dictionary `history` has two entries: the loss and the accuracy achieved using the training set.


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'acc'])



## Plot the results

As you might expect, we'll use our import matplotlib.pyplot as plt for graphing. Use the data stored in the history_dict above to plot the loss vs epochs and the accurcay vs epochs.


```python
#Your code here; plot the loss vs the number of epoch

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


![png](index_files/index_31_0.png)


From the alternative perspective, we can also plot the successive accuracy of our model as the model is tuned:


```python
#Your code here; plot the training accuracy vs the number of epochs

acc_values = history_dict['acc'] 

plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_33_0.png)


It seems like we could just keep on going and accuracy would go up!

## Make predictions

Finally, it's time to output. Use the method discussed in the previous lesson to output (probability) predictions for the test set.


```python
y_hat_test = model.predict(test) #Your code here; Output (probability) predictions for the test set.
```

## Evaluate Performance

Finally, print the loss and accuracy for both the train and test sets of the final trained model.


```python
#Your code here; print the loss and accuracy for the training set.
results_train = model.evaluate(train, label_train)
results_train
```

    58500/58500 [==============================] - 1s 20us/step





    [0.3385159788148271, 0.8770256410256411]




```python
#Your code here; print the loss and accuracy for the test set.
results_test = model.evaluate(test, label_test)
results_test
```

    1500/1500 [==============================] - 0s 25us/step





    [0.2747621349096298, 0.9200000001589457]



We can see that the training set results are really good (a 89.4% classification accuracy!), but the test set results lag behind. In the next lab. We'll talk a little more about this in the next lecture, and will discuss how we can get better test set results as well!

## Additional Resources

https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb

https://catalog.data.gov/dataset/consumer-complaint-database

## Summary 

Congratulations! In this lab, you built a neural network with much less production time thanks to the tools provided by Keras! In upcoming lessons and labs we'll continue to investigate further ideas regarding how to tune and refine these models for increased accuracy and performance.
