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


```python
# __SOLUTION__ 
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

    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
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


```python
# __SOLUTION__ 
# Import data
df = pd.read_csv('Bank_complaints.csv')

# Inspect data
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



As mentioned earlier, your task is to categorize banking complaints into various predefined categories. Preview what these categories are and what percent of the complaints each accounts for. 


```python
# Your code here
```


```python
# __SOLUTION__ 
df['Product'].value_counts(normalize=True)
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

Our first step again is to transform our textual data into a numerical representation. As we saw in some of our previous lessons on NLP, there are many ways to do this. Here, we'll use the `Tokenizer()` class from the `preprocessing.text` sub-module of the Keras package.   

As with our previous work using NLTK, this will transform our text complaints into word vectors. (Note that the method of creating a vector is different from our previous work with NLTK; as you'll see, word order will be preserved as opposed to a bag of words representation). In the below code, we'll only keep the 2,000 most common words and use one-hot encoding. 




```python
# As a quick preliminary, briefly review the docstring for keras.preprocessing.text.Tokenizer
Tokenizer?
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
reverse_index = dict([(value, key) for (key, value) in word_index.items()])
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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
# Initialize a sequential model
model = models.Sequential()

# Two layers with relu activation
model.add(layers.Dense(50, activation='relu', input_shape=(2000,)))
model.add(layers.Dense(25, activation='relu'))

# One layer with softmax activation 
model.add(layers.Dense(7, activation='softmax'))
```

## Compiling the model

Now, compile the model! This time, use `'categorical_crossentropy'` as the loss function and stochastic gradient descent, `'SGD'` as the optimizer. As in the previous lesson, include the accuracy as a metric.


```python
# Compile the model

```


```python
# __SOLUTION__ 
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
history = None
```


```python
# __SOLUTION__ 
# Train the model 
history = model.fit(train,
                    label_train,
                    epochs=120,
                    batch_size=256)
```

    Epoch 1/120
    58500/58500 [==============================] - 2s 40us/step - loss: 1.8825 - acc: 0.2408
    Epoch 2/120
    58500/58500 [==============================] - 2s 36us/step - loss: 1.6170 - acc: 0.4628
    Epoch 3/120
    58500/58500 [==============================] - 2s 31us/step - loss: 1.2554 - acc: 0.6184
    Epoch 4/120
    58500/58500 [==============================] - 2s 32us/step - loss: 0.9964 - acc: 0.6802
    Epoch 5/120
    58500/58500 [==============================] - 2s 28us/step - loss: 0.8516 - acc: 0.7086
    Epoch 6/120
    58500/58500 [==============================] - 2s 29us/step - loss: 0.7686 - acc: 0.7263
    Epoch 7/120
    58500/58500 [==============================] - 2s 28us/step - loss: 0.7161 - acc: 0.7390
    Epoch 8/120
    58500/58500 [==============================] - 2s 33us/step - loss: 0.6796 - acc: 0.7493
    Epoch 9/120
    58500/58500 [==============================] - 2s 32us/step - loss: 0.6521 - acc: 0.7575
    Epoch 10/120
    58500/58500 [==============================] - 2s 29us/step - loss: 0.6302 - acc: 0.7662
    Epoch 11/120
    58500/58500 [==============================] - 2s 30us/step - loss: 0.6125 - acc: 0.7724
    Epoch 12/120
    58500/58500 [==============================] - 1s 22us/step - loss: 0.5973 - acc: 0.7782
    Epoch 13/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.5844 - acc: 0.7827
    Epoch 14/120
    58500/58500 [==============================] - 1s 25us/step - loss: 0.5728 - acc: 0.7871
    Epoch 15/120
    58500/58500 [==============================] - 1s 25us/step - loss: 0.5622 - acc: 0.7917
    Epoch 16/120
    58500/58500 [==============================] - 2s 28us/step - loss: 0.5527 - acc: 0.7965: 0s - loss: 0.5514 - acc:
    Epoch 17/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.5440 - acc: 0.7994
    Epoch 18/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.5359 - acc: 0.8037
    Epoch 19/120
    58500/58500 [==============================] - 1s 22us/step - loss: 0.5286 - acc: 0.8072
    Epoch 20/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.5215 - acc: 0.8098
    Epoch 21/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.5152 - acc: 0.8128
    Epoch 22/120
    58500/58500 [==============================] - 1s 25us/step - loss: 0.5092 - acc: 0.8148
    Epoch 23/120
    58500/58500 [==============================] - 1s 22us/step - loss: 0.5036 - acc: 0.8170
    Epoch 24/120
    58500/58500 [==============================] - 1s 24us/step - loss: 0.4977 - acc: 0.8198
    Epoch 25/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.4930 - acc: 0.8224
    Epoch 26/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.4882 - acc: 0.8238
    Epoch 27/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4836 - acc: 0.8256
    Epoch 28/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4793 - acc: 0.8275
    Epoch 29/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4752 - acc: 0.8296
    Epoch 30/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.4712 - acc: 0.8305
    Epoch 31/120
    58500/58500 [==============================] - 1s 22us/step - loss: 0.4676 - acc: 0.8319
    Epoch 32/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.4638 - acc: 0.8342
    Epoch 33/120
    58500/58500 [==============================] - 1s 25us/step - loss: 0.4607 - acc: 0.8351
    Epoch 34/120
    58500/58500 [==============================] - 1s 24us/step - loss: 0.4577 - acc: 0.8362
    Epoch 35/120
    58500/58500 [==============================] - 2s 28us/step - loss: 0.4543 - acc: 0.8380
    Epoch 36/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4511 - acc: 0.8389
    Epoch 37/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4483 - acc: 0.8397
    Epoch 38/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4454 - acc: 0.8407
    Epoch 39/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4425 - acc: 0.8426
    Epoch 40/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4401 - acc: 0.8434
    Epoch 41/120
    58500/58500 [==============================] - 1s 17us/step - loss: 0.4375 - acc: 0.8435
    Epoch 42/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4349 - acc: 0.8449
    Epoch 43/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4328 - acc: 0.8457
    Epoch 44/120
    58500/58500 [==============================] - 1s 17us/step - loss: 0.4303 - acc: 0.8469
    Epoch 45/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4283 - acc: 0.8480
    Epoch 46/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4262 - acc: 0.8475
    Epoch 47/120
    58500/58500 [==============================] - 1s 17us/step - loss: 0.4237 - acc: 0.8490
    Epoch 48/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.4221 - acc: 0.8495
    Epoch 49/120
    58500/58500 [==============================] - 1s 18us/step - loss: 0.4201 - acc: 0.8508
    Epoch 50/120
    58500/58500 [==============================] - 1s 16us/step - loss: 0.4182 - acc: 0.8508
    Epoch 51/120
    58500/58500 [==============================] - 1s 15us/step - loss: 0.4163 - acc: 0.8511
    Epoch 52/120
    58500/58500 [==============================] - 1s 18us/step - loss: 0.4140 - acc: 0.8518
    Epoch 53/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4125 - acc: 0.8532
    Epoch 54/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4106 - acc: 0.8531
    Epoch 55/120
    58500/58500 [==============================] - 1s 18us/step - loss: 0.4091 - acc: 0.8544
    Epoch 56/120
    58500/58500 [==============================] - 1s 18us/step - loss: 0.4077 - acc: 0.8546
    Epoch 57/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4057 - acc: 0.8550
    Epoch 58/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4040 - acc: 0.8558
    Epoch 59/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.4029 - acc: 0.8564
    Epoch 60/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.4009 - acc: 0.8565
    Epoch 61/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.3998 - acc: 0.8577
    Epoch 62/120
    58500/58500 [==============================] - 1s 22us/step - loss: 0.3984 - acc: 0.8578
    Epoch 63/120
    58500/58500 [==============================] - 1s 17us/step - loss: 0.3966 - acc: 0.8584
    Epoch 64/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3953 - acc: 0.8593
    Epoch 65/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3939 - acc: 0.8591
    Epoch 66/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3931 - acc: 0.8602
    Epoch 67/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3913 - acc: 0.8602
    Epoch 68/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3900 - acc: 0.8608
    Epoch 69/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3889 - acc: 0.8614
    Epoch 70/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3873 - acc: 0.8613
    Epoch 71/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3861 - acc: 0.8622
    Epoch 72/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3851 - acc: 0.8624
    Epoch 73/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3838 - acc: 0.8630
    Epoch 74/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3826 - acc: 0.8631
    Epoch 75/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3815 - acc: 0.8640
    Epoch 76/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3806 - acc: 0.8642
    Epoch 77/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3792 - acc: 0.8640
    Epoch 78/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3780 - acc: 0.8648
    Epoch 79/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3770 - acc: 0.8650
    Epoch 80/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3758 - acc: 0.8655
    Epoch 81/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3750 - acc: 0.8650
    Epoch 82/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3742 - acc: 0.8665
    Epoch 83/120
    58500/58500 [==============================] - 1s 24us/step - loss: 0.3727 - acc: 0.8660
    Epoch 84/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3720 - acc: 0.8671
    Epoch 85/120
    58500/58500 [==============================] - 1s 18us/step - loss: 0.3710 - acc: 0.8673
    Epoch 86/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3700 - acc: 0.8667
    Epoch 87/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3690 - acc: 0.8682
    Epoch 88/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3680 - acc: 0.8685
    Epoch 89/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3673 - acc: 0.8693
    Epoch 90/120
    58500/58500 [==============================] - 1s 25us/step - loss: 0.3660 - acc: 0.8686
    Epoch 91/120
    58500/58500 [==============================] - 1s 24us/step - loss: 0.3653 - acc: 0.8693
    Epoch 92/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3643 - acc: 0.8695
    Epoch 93/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3635 - acc: 0.8705
    Epoch 94/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3623 - acc: 0.8706
    Epoch 95/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3617 - acc: 0.8705
    Epoch 96/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3608 - acc: 0.8710
    Epoch 97/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3600 - acc: 0.8715
    Epoch 98/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3590 - acc: 0.8711
    Epoch 99/120
    58500/58500 [==============================] - 1s 23us/step - loss: 0.3579 - acc: 0.8724
    Epoch 100/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3573 - acc: 0.8725
    Epoch 101/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3563 - acc: 0.8722
    Epoch 102/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3555 - acc: 0.8726
    Epoch 103/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3549 - acc: 0.8729
    Epoch 104/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3539 - acc: 0.8734
    Epoch 105/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3531 - acc: 0.8743
    Epoch 106/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3522 - acc: 0.8745
    Epoch 107/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3514 - acc: 0.8746
    Epoch 108/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3507 - acc: 0.8750
    Epoch 109/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3499 - acc: 0.8747
    Epoch 110/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3487 - acc: 0.8750
    Epoch 111/120
    58500/58500 [==============================] - 1s 20us/step - loss: 0.3484 - acc: 0.8752
    Epoch 112/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3475 - acc: 0.8752
    Epoch 113/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3468 - acc: 0.8759
    Epoch 114/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3462 - acc: 0.8761
    Epoch 115/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3453 - acc: 0.8775
    Epoch 116/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3444 - acc: 0.8764
    Epoch 117/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3438 - acc: 0.8762
    Epoch 118/120
    58500/58500 [==============================] - 1s 21us/step - loss: 0.3431 - acc: 0.8775
    Epoch 119/120
    58500/58500 [==============================] - 1s 16us/step - loss: 0.3420 - acc: 0.8775
    Epoch 120/120
    58500/58500 [==============================] - 1s 19us/step - loss: 0.3418 - acc: 0.8774


Recall that the dictionary `history` has two entries: the loss and the accuracy achieved using the training set.


```python
history_dict = history.history
history_dict.keys()
```


```python
# __SOLUTION__ 
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
# __SOLUTION__ 
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


    
![png](index_files/index_41_0.png)
    



```python
# Plot the training accuracy vs the number of epochs

```

It seems like we could just keep on going and accuracy would go up!


```python
# __SOLUTION__ 
# Plot the training accuracy vs the number of epochs

acc_values = history_dict['acc'] 

plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```


    
![png](index_files/index_44_0.png)
    


## Make predictions

Finally, it's time to make predictions. Use the relevant method discussed in the previous lesson to output (probability) predictions for the test set.


```python
# Output (probability) predictions for the test set 
y_hat_test = None
```


```python
# __SOLUTION__ 
# Output (probability) predictions for the test set 
y_hat_test = model.predict(test) 
```

## Evaluate Performance

Finally, print the loss and accuracy for both the train and test sets of the final trained model.


```python
# Print the loss and accuracy for the training set 
results_train = None
results_train
```


```python
# __SOLUTION__ 
# Print the loss and accuracy for the training set 
results_train = model.evaluate(train, label_train)
results_train
```

    58500/58500 [==============================] - 2s 29us/step





    [0.33621186860045815, 0.8808376068376068]




```python
# Print the loss and accuracy for the test set 
results_test = None
results_test
```


```python
# __SOLUTION__ 
# Print the loss and accuracy for the test set 
results_test = model.evaluate(test, label_test)
results_test
```

    1500/1500 [==============================] - 0s 34us/step





    [0.23364664735396704, 0.9266666668256124]



We can see that the training set results are really good, and the test set results seem to be even better. In general, this type of result will be rare, as train set results are usually at least a bit better than test set results.

## Summary 

Congratulations! In this lab, you built a neural network thanks to the tools provided by Keras! In upcoming lessons and labs we'll continue to investigate further ideas regarding how to tune and refine these models for increased accuracy and performance.
