# -*- coding: utf-8 -*-

pip install python_speech_features

!pip3 install --upgrade speechrecognition
import speech_recognition as sr
sr.__version__

pip install pydub

from os import path
from pydub import AudioSegment

#converter codes 
'''
from pydub import AudioSegment

#flac to wav converter
flac_audio = AudioSegment.from_file("a1.flac", "flac")
flac_audio.export("a1.wav", format="wav")

#amr to wav converter
amr_audio = AudioSegment.from_file("k.amr", "amr")
amr_audio.export("k.wav", format="wav")

#mp3 to wav converter
flac_audio = AudioSegment.from_file("xyz.mp3", "mp3")
flac_audio.export("xyz.wav", format="wav")


#m4a to wav converter
m4a_audio = AudioSegment.from_file("uday1.m4a", "m4a")
m4a_audio.export("uday1.wav", format="wav")
'''

"""from pydub import AudioSegment


```
# This is formatted as code
converter code
#flac to wav converter
flac_audio = AudioSegment.from_file("a1.flac", "flac")
flac_audio.export("a1.wav", format="wav")

#amr to wav converter
amr_audio = AudioSegment.from_file("k.amr", "amr")
amr_audio.export("k.wav", format="wav")

#mp3 to wav converter
flac_audio = AudioSegment.from_file("xyz.mp3", "mp3")
flac_audio.export("xyz.wav", format="wav")
```
"""

#mp3 to wav converter code
'''
from os import path
from pydub import AudioSegment
'''
# files                                                                         
src = "sivram.mp3"   #filename 
dst = "A.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
'''
#sample 1
#m4a to wav converter
m4a_audio = AudioSegment.from_file("uday1.m4a", "m4a")
m4a_audio.export("B.wav", format="wav")

#sample 2
#m4a to wav converter


m4a_audio = AudioSegment.from_file("uday2.m4a", "m4a")
m4a_audio.export("B.wav", format="wav")

'''
#not in use

#mp3 to wav converter code

from os import path
from pydub import AudioSegment

# files                                                                         
src = "voice_sobiga.mp3"
dst = "B.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")

#!zip -r /content/spring_struts_JDs.zip /content/spring_struts
#from google.colab import files
#files.download("/content/CloudJDs.zip")


# Import necessary libraries 
from pydub import AudioSegment 
import speech_recognition as sr 

# Input audio file to be sliced 
audio = AudioSegment.from_wav("/content/A.wav") 

''' 
Step #1 - Slicing the audio file into smaller chunks. 
'''
# Length of the audiofile in milliseconds 
n = len(audio) 

# Variable to count the number of sliced chunks 
counter = 1

# Text file to write the recognized audio 
fh = open("recognized1.txt", "w+") 

# Interval length at which to slice the audio file. 
# If length is 22 seconds, and interval is 5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 5 - 10 seconds 
# chunk3 : 10 - 15 seconds 
# chunk4 : 15 - 20 seconds 
# chunk5 : 20 - 22 seconds 
interval = 5 * 1000

# Length of audio to overlap. 
# If length is 22 seconds, and interval is 5 seconds, 
# With overlap as 1.5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 3.5 - 8.5 seconds 
# chunk3 : 7 - 12 seconds 
# chunk4 : 10.5 - 15.5 seconds 
# chunk5 : 14 - 19.5 seconds 
# chunk6 : 18 - 22 seconds 
overlap = 1.5 * 0  #creating overlap equals to zero

# Initialize start and end seconds to 0 
start = 0
end = 0

# Flag to keep track of end of file. 
# When audio reaches its end, flag is set to 1 and we break 
flag = 0

# Iterate from 0 to end of the file, 
# with increment = interval 
for i in range(0, 2 * n, interval): 
	
	# During first iteration, 
	# start is 0, end is the interval 
	if i == 0: 
		start = 0
		end = interval 

	# All other iterations, 
	# start is the previous end - overlap 
	# end becomes end + interval 
	else: 
		start = end - overlap 
		end = start + interval 

	# When end becomes greater than the file length, 
	# end is set to the file length 
	# flag is set to 1 to indicate break. 
	if end >= n: 
		end = n 
		flag = 1

	# Storing audio file from the defined start to end 
	chunk = audio[start:end] 

	# Filename / Path to store the sliced audio 
	filename = 'a'+str(counter)+'.wav'

	# Store the sliced audio file to the defined path 
	chunk.export(filename, format ="wav") 
	# Print information about the current chunk 
	print("Processing chunk "+str(counter)+". Start = "
						+str(start)+" end = "+str(end)) 

	# Increment counter for the next chunk 
	counter = counter + 1
	
	# Slicing of the audio file is done. 
	# Skip the below steps if there is some other usage 
	# for the sliced audio files. 

  #''' 
  #Step #2 - Recognizing the chunk and writing to a file. 
  #'''

	# Here, Google Speech Recognition is used 
	# to take each chunk and recognize the text in it. 

	# Specify the audio file to recognize 

	AUDIO_FILE = filename 

	# Initialize the recognizer 
	r = sr.Recognizer() 

	# Traverse the audio file and listen to the audio 
	with sr.AudioFile(AUDIO_FILE) as source: 
		audio_listened = r.listen(source) 

	# Try to recognize the listened audio 
	# And catch expections. 
	try:	 
		rec = r.recognize_google(audio_listened) 
		
		# If recognized, write into the file. 
		fh.write(rec+" ") 
	
	# If google could not understand the audio 
	except sr.UnknownValueError: 
		print("Could not understand audio") 

	# If the results cannot be requested from Google. 
	# Probably an internet connection error. 
	except sr.RequestError as e: 
		print("Could not request results.") 

	# Check for flag. 
	# If flag is 1, end of the whole audio reached. 
	# Close the file and break. 
	if flag == 1: 
		fh.close() 
		break

#!zip -r /content/spring_struts_JDs.zip /content/spring_struts



#from google.colab import files

#files.download("/content/CloudJDs.zip")


# Import necessary libraries 
from pydub import AudioSegment 
import speech_recognition as sr 

# Input audio file to be sliced 
audio = AudioSegment.from_wav("/content/B.wav") 

''' 
Step #1 - Slicing the audio file into smaller chunks. 
'''
# Length of the audiofile in milliseconds 
n = len(audio) 

# Variable to count the number of sliced chunks 
counter = 1

# Text file to write the recognized audio 
fh = open("recognized1.txt", "w+") 

# Interval length at which to slice the audio file. 
# If length is 22 seconds, and interval is 5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 5 - 10 seconds 
# chunk3 : 10 - 15 seconds 
# chunk4 : 15 - 20 seconds 
# chunk5 : 20 - 22 seconds 
interval = 5 * 1000 

# Length of audio to overlap. 
# If length is 22 seconds, and interval is 5 seconds, 
# With overlap as 1.5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 3.5 - 8.5 seconds 
# chunk3 : 7 - 12 seconds 
# chunk4 : 10.5 - 15.5 seconds 
# chunk5 : 14 - 19.5 seconds 
# chunk6 : 18 - 22 seconds 
overlap = 1.5 * 0 # KEEPING OVERLAP 0 HERE 

# Initialize start and end seconds to 0 
start = 0
end = 0

# Flag to keep track of end of file. 
# When audio reaches its end, flag is set to 1 and we break 
flag = 0

# Iterate from 0 to end of the file, 
# with increment = interval 
for i in range(0, 2 * n, interval): 
	
	# During first iteration, 
	# start is 0, end is the interval 
	if i == 0: 
		start = 0
		end = interval 

	# All other iterations, 
	# start is the previous end - overlap 
	# end becomes end + interval 
	else: 
		start = end - overlap 
		end = start + interval 

	# When end becomes greater than the file length, 
	# end is set to the file length 
	# flag is set to 1 to indicate break. 
	if end >= n: 
		end = n 
		flag = 1

	# Storing audio file from the defined start to end 
	chunk = audio[start:end] 

	# Filename / Path to store the sliced audio 
	filename = 'b'+str(counter)+'.wav'

	# Store the sliced audio file to the defined path 
	chunk.export(filename, format ="wav") 
	# Print information about the current chunk 
	print("Processing chunk "+str(counter)+". Start = "
						+str(start)+" end = "+str(end)) 

	# Increment counter for the next chunk 
	counter = counter + 1
	
	# Slicing of the audio file is done. 
	# Skip the below steps if there is some other usage 
	# for the sliced audio files. 

  #''' 
  #Step #2 - Recognizing the chunk and writing to a file. 
  #'''

	# Here, Google Speech Recognition is used 
	# to take each chunk and recognize the text in it. 

	# Specify the audio file to recognize 

	AUDIO_FILE = filename 

	# Initialize the recognizer 
	r = sr.Recognizer() 

	# Traverse the audio file and listen to the audio 
	with sr.AudioFile(AUDIO_FILE) as source: 
		audio_listened = r.listen(source) 

	# Try to recognize the listened audio 
	# And catch expections. 
	try:	 
		rec = r.recognize_google(audio_listened) 
		
		# If recognized, write into the file. 
		fh.write(rec+" ") 
	
	# If google could not understand the audio 
	except sr.UnknownValueError: 
		print("Could not understand audio") 

	# If the results cannot be requested from Google. 
	# Probably an internet connection error. 
	except sr.RequestError as e: 
		print("Could not request results.") 

	# Check for flag. 
	# If flag is 1, end of the whole audio reached. 
	# Close the file and break. 
	if flag == 1: 
		fh.close() 
		break

#for reference for prediction part code piece
'''
print('Extracting features from audio')  #we approach each of our audio file very presizely
for fn in tqdm(os.listdir("audio_dir")):   #fn is file name we are iterating list directory inside that audio directory
  rate, wav =wavfile.read(os.path.join(audio_dir, fn))   #joining audio dir and file name f and reading it
  label = fn2class[fn]      #in order to calculate accuracy matrix
  c= classes.index(label)   #so here classes is js a list with index 0 to 9 with all of our individual classes ## indetifying to whom this audio belongs
  y_prob =[]                # here we took y_prob equals to a list

  for i in range(0,wave.shape[0]-config.step,config.stop):     #now we we will go each specific index we gonna walk thru each audio file
    sample = wav[i:i+config.step]                         #sample is equal to wav from i to i+config.step and this sample will be 1/10th of sec of audio file
    x = mfcc(sample, rate, numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)  #we have used it earliar as well in EDA
    x = (x-config.min)/(config.max-config.min)   # scaling the data above line

    if config.mode =='conv':    # for conv
      x =x.reshape(1,x.shape[0],x.shape[1],1)        #so we r reshaping our array to 1*1st feature*2nd feature size sine its a convolution model we need single grey scale channel for our array
    elif config.mode == 'time':    #recurrent
      x = np.expand_dims(x, axis =0)       #if we want to reshape the array we can add single demention on some thing we can use "np.expand_dim" and we can expand dimention to some (array x in the axis =0)
    y_hat = model.predict(x)            #prediction one of the sample
    y_prob.append(y_hat)
    y_pred.append(np.argmax(y_hat))       # from argmax we will get maximum value from y_hat 
    y_true.append(c)
  fn_prob[fn] =np.mean(y_prob, axis =0).flatten()  # so for each file name we want to store associated probability of the classification of that file $ we take mean (i.e np.mean)of every single thing Y_prob list and we have stored all prediction in it ".flatten" to flatten out the array
  return y_true, y_pred, fn_prob   
  '''

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt


from scipy.io import wavfile       #it will also detect sampling rate for us
from tqdm import tqdm
from keras.layers import Conv2D, MaxPool2D,Flatten ,LSTM
from keras.layers import Dropout, Dense  , TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical

import librosa
from scipy.io import wavfile
from sklearn.utils.class_weight import compute_class_weight
import pickle   # we are string our data in pickle that we have learned from training and called up when required rather than running whole model again which is to much time consuming 

from cfg import config

#saving model part1 

from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint #saving our model from keras so that we can load up later and make prediction

from python_speech_features import mfcc, logfbank


#fig.suptitle('This is the figure title', fontsize=12) it will help us to nam eplot and font size

1# we will remove dead space from signal that very low aplitude part we will run it js before caclc_fft code 


def envelope(y, rate, threshold):
  mask= []    # we will create a mask
  y = pd.Series(y).apply(np.abs)
  y_mean = y.rolling(window=int(rate/10),min_periods=1, center=True).mean()

  for mean in y_mean:
    if mean > threshold:
      mask.append(True)
    else:
      mask.append(False)

  return mask

#creating Calc_fft functin to calculate fft so that we can get to know spacing between our datapoint 
def calc_fft(y,rate):                                  # y is signal
  n= len(y)    
  #fft has two part magnitude and frequency we use np.fft ".rfftfreq" rfft = real fft frequency so that we can get frequency and np.fft.rfft to get value of magnitude   predefined method in np
  freq = np.fft.rfftfreq(n, d=1/rate)          #d is spacing between all of the individual samples which is equal to 1/sampling rate
  Y = abs(np.fft.rfft(y)/n )   # signal is in (a+ib) form so  as to take magnitude we use "abs" which do sqrt(a^2+b^2), and we will divide by n so as to normaliise the function so that magnitude scaled aproiately based on the scale of 
  return(Y,freq)

#loading data frame in this one is audio recording another is a test.csv file contains labeling wrt to audio file

df = pd.read_csv('merge_splitAB.csv')
df.set_index('fname', inplace =True)   #setting index from fname
print(df)

for f in df.index:
  rate, signal = wavfile.read(f)            #in desktop app use path in place of 'f' in '.read(f)'
  print(signal.shape[0])
  df.at[f,'length'] = signal.shape[0]/rate   #df.at[] acess individual element
  print(signal.shape[0]/rate)

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean() #we are groupby on the basis of label
                                               # in that we are choosing length and taking mean of it

df.reset_index(inplace=True)

#we are plotting the distribution of each class or instrumnet means amoutf data dat we are having for the respt instrument
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)       #if we do not use equal pie chart will look like ellipse
ax.axis('equal')
plt.show()

#four dictionary to store the data from  class  c metioned below we need this data to plot later on
signals = {}
fft = {}
fbank = {}
mfccs = {}

#there a lot of file can b possible in 1 instrument we are taking one file from each class

for c in classes:
  
                                          
  wav_file = df[df.label ==c].iloc[0,0]    #it will return df of particular class or instrument with all the files and in that only one example file from a class 
  print(wav_file)
  signal, rate = librosa.load(wav_file, sr=44100)   # we load our wave files




  




  fft[c] = calc_fft (signal, rate ) 

  bank = logfbank(signal[:rate], rate, nfilt = 26 , nfft = 1103).T     #T is transpose as the way it return the matrix
  fbank = bank

  mel =mfcc(signal[:rate], rate, numcep=13, nfilt = 26, nfft = 1103).T
  mfccs[c]=mel

mask = envelope(signal, rate, 0.0005)  
  signal = signal[mask]
  signals[c] = signal

print(os.getcwd())

print('inside cell')

if len(os.listdir('clean')) ==0:
  print("clean is zero")
  for f in tqdm(df.fname):
    signal, rate = librosa.load(f, sr = 16000)
    mask = envelope(signal,rate,0.0005)
    os.chdir("clean")
    wavfile.write(filename=f , rate =rate, data= signal[mask])
    os.chdir("..")
    print(f)
else:
  print("clean is not zero")

# this gonna look at ur pickle folder it will see there is an existing file is there and loading in pickle directory
def check_data():
  if os.path.isfile(config.p_path):    # checking pickle file path
    print('Loading existing data for {} model'.format(config.mode)) # if we found something in above mentioned path then we will load it here 
    with open(config.p_path, 'rb') as handle:   # rb = read bytes 
      print(handle)
      tmp =pickle.load(handle) #so pickle file will be loaded here
      return tmp    # tmp file will sent to the desired folder 
  else :
    return None

"""y = to_categorical(y, num_classes = 10)  # <--- it will decide no. of person audio we can take 
  #num_classes=10 means we can only have 10 person
  in below cod ewe are using that
"""

#function that gonna build all of our data so it will preproces to push through our model
def build_rand_feat():
  '''tmp = check_data()   #implemention newly created pickle folder here
  if tmp:             #data is found in tmp file  then it will take tuple 1 and tuple 2 mentioned in below line 
     print("Exiting build_rand_feat")
     return tmp.data[0],tmp.data[1]
     
     print (tmp)
  print(tmp) ''' 
  x=[]         
  y=[]
  _min, _max = float('inf'), -float('inf')                                      
  for _ in tqdm (range(n_samples)):            #generating min max value thru this "for loop"
    rand_class = np.random.choice(class_dist.index, p= prob_dist )  
    file = np.random.choice(df[df.label== rand_class].index)        
    rate, wav = wavfile.read('clean/'+file)  
    label  = df.at[file,'label']             
    rand_index = np.random.randint(0, wav.shape[0]-config.step)    
    sample = wav[rand_index:rand_index+config.step] 
    x_sample = mfcc(sample, rate, numcep = config.nfeat,nfilt=config.nfilt, nfft=config.nfft) #(in this step our matrix dimention is being framed i.e 12*9)
    _min = min(np.amin(x_sample), _min)    # amin choose out themin value from the matrix and "amin(x_sample), min" and this line will choose min value from matrix and compare with existing min value 
    _max = max(np.amax(x_sample), _max)
    '''x.append(x_sample if cong.mode == 'conv' else x_sample.T) "" #appending x_sample'''
    x.append(x_sample)   
    y.append(classes.index(label))            
  config.min = _min # we do nott need to recalculate min and max for validation set 
  config.max = _max
  x ,y = np.array(x) , np.array(y)          #converting list into matrix 
  x = (x- _min)/(_max-_min)            # we rescale or normalise the value  between 0 and 1
  if config.mode =='conv':       
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)  
  elif config.mode == 'time':                          #RNN
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2])
  y = to_categorical(y, num_classes = 10)  # <--- it will decide no. of person audio we can take 
  #num_classes=10 means we can only have 10 person
  # hot encoding the linear value of y above 
  config.data =(x,y) #return tupple for saving data & model from above mentioned in above code tmp.data[0],tmp.data[1] will be entered into x and y and we will store in config # once we store our data in config we will save entire object of config in a pickle
  print(config)



  with open(config.p_path,'wb') as handle :    #wb = write bytes
    pickle.dump(config, handle)
    

  return x,y

#CNN MODEL and layers using keras 

def get_conv_model():
  model = Sequential()  # we are using sequencial model as it is super simple we dont need functional API


  model.add(Conv2D(16,(3,3),strides=(1,1), activation ='relu', padding = 'same',input_shape =input_shape))  # here 16 = filters,(3,3 is convolution of 3*3),relu is activation fn,
  
  # layer 2
  model.add(Conv2D(32,(3,3), strides=(1,1),activation ='relu',  padding = 'same'))
  # layer 3
  model.add(Conv2D(64,(3,3),strides=(1,1), activation ='relu',  padding = 'same'))
  # layer 4
  model.add(Conv2D(128,(3,3),strides=(1,1), activation ='relu',  padding = 'same'))
  # more layer we add we get more new feature abt our data as well as our model will become more complex and slow
  
  #max pooling
  model.add(MaxPool2D(2,2))  
  model.add(Dropout(0.5)) 
  model.add(Flatten()) 
  #flattening  dense layers
  model.add(Dense(128, activation ='relu')) 
  model.add(Dense(64, activation ='relu')) 
  model.add(Dense(10, activation ='softmax')) 
  model.summary() #to check how model looks like
  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) #acc is accuracy 
  return model



from keras.callbacks import ModelCheckpoint
from cfg import config
import os

df = pd.read_csv('merge_splitAB.csv')
df.set_index('fname', inplace = True)
print(df)

#from google.colab import drive
#drive.mount('/content/drive')

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)       #if we do not use equal pie chart will look like ellipse
ax.axis('equal')
plt.show()

print(os.getcwd())
print(df)
print(df.columns)

for f in df.index:
    print(f)
    rate, signal = wavfile.read('clean/'+f)      #after EDA we save our file at clean directory
    print(signal.shape[0])
    df.at[f, 'length'] = signal.shape[0]/rate

print(df)

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

#in order to increase ur sample size we will take legth frm df, df.['length'].sum()/0.1 or 1/10sec meaning summing up the all the values in length col. and /0.1 sec gives no of smaple
n_samples = 2*int(df['length'].sum()/0.1)

# we will create probalility distribution to know waht is the probablility of getting audio from particular class 
prob_dist = class_dist /class_dist.sum()   # so it will convert evrything between 0 and 1
choices = np.random.choice(class_dist.index, p= prob_dist )  # class_dist.index this is all our musical instrument we have to choose from ,its just radomly sampling based the probability distribution



fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = config(mode='conv')

if config.mode == 'conv':
  x,y = build_rand_feat()       #building our feature  set from np.random.choice
  y_flat = np.argmax(y, axis=1)
     # we need to take the hot encoded  y matrix to their original class encoding and np.argmax will directly and fastly match up with original colom wrt to their hotencoded value
  input_shape = (x.shape[1],x.shape[2],1) #these are dimention input shape for CNN we will give to keras in neural network 
  model = get_conv_model()     # we are calling our model for that we will create a function "get_conv_model()" later on 


elif config.mode == 'time':
  x,y = build_rand_feat()
  y_flat = np.argmax(y, axis=1)     # repeating same for RNN as above in CNN
  input_shape = (x.shape[1],x.shape[2],1)
  model = get_recurrent_model()

#from google.colab import drive
#drive.mount('/content/drive')

class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)    # using utility " from sklearn.utils.class_weight import compute_class_weight" and "np.unique(y_flat),y_flat" reduce bias in our neuralnet



#creating model.fit(x,y) it will take x and y matrix and randomly create batches of our data
model.fit(x,y, epochs = 15, batch_size = 32, shuffle = True, class_weight = class_weight )  # batch_size = 32 , it is default #class_weight(parameter of model.fit) = class_weight(we have calculated above)

class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat) 

# creating model check point 
#below monitor = 'val_acc' monitoring validation accuracy out of 4 matrix we are recording here rest are training accuracy/accuracy , val_acc, loss, Validation loss. we r actually adding validation split to the data verbose = 1
#mode = 'max' for validation accuracy but if it is for loss we ake mode = 'min' since we want loss to b minimum
# "save_best_only=True" so it only save our model if its a improvement from our classification, 
#"save_weights_only =False" so when we save models in kers so it will b looking for a weight file and in old day it use to like i m lookin for model architecture 
#i m looking for supplyi m looking for weights to supply that architecture generally architecture going to b json or ml file we can store all of this in one model file

checkpoint = ModelCheckpoint(config.model_path, monitor = 'val_acc', verbose =1, mode = 'max', save_best_only=True, save_weights_only =False,period=1)

#creating model.fit(x,y) it will take x and y matrix and randomly create batches of our data
#model.fit(x,y, epochs = 5 , batch_size = 32, shuffle = True, class_weight = class_weight )  # batch_size = 32 , it is default #class_weight(parameter of model.fit) = class_weight(we have calculated above)
model.fit(x,y, epochs = 5 , batch_size = 32, shuffle = True, validation_split=0.1, callbacks = [checkpoint] ) #"class_weight = class_weight" we have removed from original 
# "validation_split=0.1" that means bottom 10% is taken from validation matrix so when u r using validation split suffle your data
#  "callbacks = [checkpoint]" so that it can take our checkpoint object that we have taken before

model.save(config.model_path) #path where we saved our model after its training



#back up code 
'''
def build_prediction(audio_dir):   # passing a audio directory and will make prediction in all the audio files in this directory it will return 3 things y_true, y_pred , fn_pred also metioned below
    y_true = []    #it will b true class,  2) in this build predict we are also creating accuracy matrix
    y_pred = []   #it will b prediction from thr neural network and 
    
    # out of softmax layer from our model it wil b a 1*10 array if will addup to all these values they will b total to 1 so idea behind softmax is that it is extention to logistic regression using the sigmoid activation to create individual class probability Ex. if want to know the probability that a file has been classified as accoustic guitar we will find at the zeroth index of 1*10 arrayand we will get its probability
    fn_prob ={} #fn_prob means file name probability
    print('Extracting features from audio')  #we approach each of our audio file very presizely
    for fn in tqdm(os.listdir(audio_dir)):   #fn is file name we are iterating list directory inside that audio directory
      print("\ninside filing")
      print(fn)
      rate, wav =wavfile.read(os.path.join(audio_dir, fn))   #joining audio dir and file name f and reading it
      label = fn2class[fn]      #in order to calculate accuracy matrix
      c= classes.index(label)   #so here classes is js a list with index 0 to 9 with all of our individual classes ## indetifying to whom this audio belongs
      y_prob =[]                # here we took y_prob equals to a list
      
      #for _ in tqdm (range(n_samples)):   #(from training part)
      for i in range(0,wav.shape[0]-config.step,config.step):     #now we we will go each specific index we gonna walk thru each audio file
        sample = wav[i:i+config.step]                         #sample is equal to wav from i to i+config.step and this sample will be 1/10th of sec of audio file
        #   below mention code in red is taken from training part 
        x = mfcc(sample, rate, numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)  #we have used it earliar as well in EDA
        print(x.shape)
        print("andriod")
        x = (x-config.min)/(config.max-config.min)   # scaling the data above line
        print(x.shape)
        if config.mode =='conv':    # for conv
          x =x.reshape(1,x.shape[0],x.shape[1],1)        #so we r reshaping our array to 1*1st feature*2nd feature size sine its a convolution model we need single grey scale channel for our array
        elif config.mode == 'time':    #recurrent
          x = np.expand_dims(x, axis =0)       #if we want to reshape the array we can add single demention on some thing we can use "np.expand_dim" and we can expand dimention to some (array x in the axis =0)
        print(x.shape)
        y_hat = model.predict(x)            #prediction one of the sample
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat,axis=1))       # from argmax we will get maximum value from y_hat 
        y_true.append(c)
      fn_prob[fn] =np.mean(y_prob, axis =0).flatten()  # so for each file name we want to store associated probability of the classification of that file $ we take mean (i.e np.mean)of every single thing Y_prob list and we have stored all prediction in it ".flatten" to flatten out the array
    print(fn_prob)
    return y_true, y_pred, fn_prob  
    '''

"""**SAVING AND PREDICTING MODEL**"""

import pickle
import os
import numpy as np
from tqdm  import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

print(os.getcwd())

def build_prediction(audio_dir):   # passing a audio directory and will make prediction in all the audio files in this directory it will return 3 things y_true, y_pred , fn_pred also metioned below
    y_true = []    #it will b true class,  2) in this build predict we are also creating accuracy matrix
    y_pred = []   #it will b prediction from thr neural network and 
    
    # out of softmax layer from our model it wil b a 1*10 array if will addup to all these values they will b total to 1 so idea behind softmax is that it is extention to logistic regression using the sigmoid activation to create individual class probability Ex. if want to know the probability that a file has been classified as accoustic guitar we will find at the zeroth index of 1*10 arrayand we will get its probability
    fn_prob ={} #fn_prob means file name probability
    print('Extracting features from audio')  #we approach each of our audio file very presizely
    for fn in tqdm(os.listdir(audio_dir)):   #fn is file name we are iterating list directory inside that audio directory
      print("\ninside filing")
      print(fn)
      rate, wav =wavfile.read(os.path.join(audio_dir, fn))   #joining audio dir and file name f and reading it
      label = fn2class[fn]      #in order to calculate accuracy matrix
      c= classes.index(label)   #so here classes is js a list with index 0 to 9 with all of our individual classes ## indetifying to whom this audio belongs
      y_prob =[]                # here we took y_prob equals to a list
      
      
      for i in range(0,wav.shape[0]-config.step,config.step):     #now we we will go each specific index we gonna walk thru each audio file
        sample = wav[i:i+config.step]                     #sample is equal to wav from i to i+config.step and this sample will be 1/10th of sec of audio file
        '''for _ in tqdm (range(n_samples)):   #(from training part)  
        rand_index = np.random.randint(0, wav.shape[0]-config.step)#(from training part)
        sample = wav[rand_index:rand_index+config.step]  #(from training part)'''
        print(len(sample))
        x = mfcc(sample, rate, numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)  #we have used it earliar as well in EDA
        print(x.shape)
        print("andriod")
        x = (x-config.min)/(config.max-config.min)   # scaling the data above line
        print(x.shape)
        if config.mode =='conv':    # for conv
          x =x.reshape(1,x.shape[0],x.shape[1],1)        #so we r reshaping our array to 1*1st feature*2nd feature size sine its a convolution model we need single grey scale channel for our array
        elif config.mode == 'time':    #recurrent
          x = np.expand_dims(x, axis =0)       #if we want to reshape the array we can add single demention on some thing we can use "np.expand_dim" and we can expand dimention to some (array x in the axis =0)
        print(x.shape)
        y_hat = model.predict(x)            #prediction one of the sample
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat,axis=1))       # from argmax we will get maximum value from y_hat 
        y_true.append(c)
      fn_prob[fn] =np.mean(y_prob, axis =0).flatten()  # so for each file name we want to store associated probability of the classification of that file $ we take mean (i.e np.mean)of every single thing Y_prob list and we have stored all prediction in it ".flatten" to flatten out the array
    print(fn_prob)
    return y_true, y_pred, fn_prob

#function that gonna build all of our data so it will preproces to push through our model
'''
def build_rand_feat():

  x=[]         
  y=[]
  _min, _max = float('inf'), -float('inf')                                      
  for _ in tqdm (range(n_samples)):            #generating min max value thru this "for loop"
    rand_class = np.random.choice(class_dist.index, p= prob_dist )  
    file = np.random.choice(df[df.label== rand_class].index)        
    rate, wav = wavfile.read('clean/'+file)  
    label  = df.at[file,'label']             
    rand_index = np.random.randint(0, wav.shape[0]-config.step)    
    sample = wav[rand_index:rand_index+config.step]   
    x_sample = mfcc(sample, rate, numcep = config.nfeat,nfilt=config.nfilt, nfft=config.nfft) 
    _min = min(np.amin(x_sample), _min)    # amin choose out themin value from the matrix and "amin(x_sample), min" and this line will choose min value from matrix and compare with existing min value 
    _max = max(np.amax(x_sample), _max)
    #x.append(x_sample if cong.mode == 'conv' else x_sample.T) "" #appending x_sample
    x.append(x_sample)   
    y.append(classes.index(label))            
  config.min = _min # we do nott need to recalculate min and max for validation set 
  config.max = _max
  x ,y = np.array(x) , np.array(y)          #converting list into matrix 
  x = (x- _min)/(_max-_min)            # we rescale or normalise the value  between 0 and 1
  if config.mode =='conv':       
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)  
  elif config.mode == 'time':                          #RNN
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2])
  y = to_categorical(y, num_classes = 10)      # hot encoding the linear value of y 
  config.data =(x,y) #return tupple for saving data & model from above mentioned in above code tmp.data[0],tmp.data[1] will be entered into x and y and we will store in config # once we store our data in config we will save entire object of config in a pickle
  print(config)



  with open(config.p_path,'wb') as handle :    #wb = write bytes
    pickle.dump(config, handle)
    

  return x,y  '''

df =pd.read_csv('merge_splitAB.csv')  #test.csv for uswhile testing 
classes = list(np.unique(df.label))  # we are creating which is unique to df.label
fn2class =dict(zip(df.fname,df.label))  # craeting dictionary fn2class means file name to classes so that for every file name we can immidietly find its true class label
print("\nfn2class Dictionary: ")
for fn2 in fn2class:
  print(fn2)
p_path = os.path.join('pickles','conv.p') # creating pickle path so the we can figure out the config for secific model


with open(p_path,'rb') as handle:   
  config =pickle.load(handle)       #storing config as we need min and max so that we can scale our data before we can go for prediction

model = load_model(config.model_path)   #that will load our model from keras

y_true, y_pred, fn_prob = build_prediction('clean')  # build prediction will b turned y_true, y_pred, fn_prob
#y_true, y_pred, fn_prob = build_prediction('clean')
acc_score = accuracy_score(y_true =y_true, y_pred=y_pred )  #from sklearn

print(fn_prob)

y_probs = []    # its gonna be list we are going to modify dataframe we are going to fill that data frame with associated class probability
# so each coloumn will b class like cello , guitar etc and give the probabilty from all of our aggregated results every 10th of second and store that probability in the dataframe 
for i, row in df.iterrows():
  y_prob = fn_prob[row.fname]    #when we iterrows then we have index  and evrything in the row 
  y_probs.append(y_prob)          #all the probability accumulating this individual output
  for c, p in zip(classes, y_prob):      # y_prob is 1/10 and classes here we chose 10
    df.at[i,c] = p                         #p is associated probability for that class


y_pred = [classes[np.argmax(y)]for y in y_probs]
df['y_pred'] = y_pred


df.to_csv('prediction.csv', index =False)

#trial
outcome=pd.read_csv("prediction.csv")
print(outcome)
outcome['A_on_Tens_scale'] =outcome['A']*10
outcome['B_on_Tens_scale'] =outcome['B']*10

'''if outcome[1,:]>0.1 :
    outcome[1,:]=1
else:
    outcome[1,:]=0'''

print(outcome)

#outcome.to_csv('outcome1.csv', index =False)