import pandas as pd
commands=pd.read_csv('TextCommands.csv’)
commands.columns = ['text','label','misc']           
commands.head()

#Data Preprocessing
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
MAX_SEQUENCE_LENGTH = 10
MAX_NUM_WORDS = 5000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(commands['text'])
sequences = tokenizer.texts_to_sequences(commands['text'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(commands['label']))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

#Data Splitting
VALIDATION_SPLIT = 0.1
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

#Model Building
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.initializers import Constant
EMBEDDING_DIM = 60
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_layer = Embedding(num_words,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH,trainable=True)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(64, 3, activation='relu')(embedded_sequences)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x=Flatten()(x)
x = Dense(100, activation='relu')(x)
preds = Dense(27, activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
model.summary()


#Model Training and Evaluation
s=0.0
for i in range (1,50):
    model.fit(x_train, y_train,batch_size=50, epochs=30, validation_data=(x_val, y_val))
    # evaluate the model
    scores = model.evaluate(x_val, y_val, verbose=0)
    s=s+(scores[1]*100)

# evaluate the model
scores = model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#classify a new unseen text command
# new instance where we do not know the answer
Xnew=["kindly undo the changes","Can you please undo the last paragraph","Make bold this","Would you be kind enough to bold the last word?","Please remove bold from the last paragraph","Kindly unbold the selected text","Kindly insert comment here","Can you please put a comment here","Can you please centre align this text","Can you please position this text in the middle"]
sequences_new = tokenizer.texts_to_sequences(Xnew)
data = pad_sequences(sequences_new, maxlen=MAX_SEQUENCE_LENGTH)
# make a prediction
yprob = model.predict(data)
yclasses=yprob.argmax(axis=-1)
# show the inputs and predicted outputs
print("X=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%snX=%s, Predicted=%s" % (Xnew[0], yclasses[0],Xnew[1],yclasses[1],Xnew[2],yclasses[2],Xnew[3],yclasses[3],Xnew[4],yclasses[4],Xnew[5],yclasses[5],Xnew[6],yclasses[6],Xnew[7],yclasses[7],Xnew[8],yclasses[8],Xnew[9],yclasses[9]))
