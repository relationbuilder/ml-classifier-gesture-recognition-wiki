# Stock Price Prediction ML Tutorial

**I provide [consulting services](http://BayesAnalytic.com/contact) to help adapt  [Quantized Classifier](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition) to meet your project needs.**

> > Summary:   During the period tested 36.8% of all SPY bars met our goal of rising by 1% in future bars before the price dropped by 1%.   The engine trained from 1,573 bars while making predictions for 171 bars ending in Jan-2017.     It predicted 33 bars  would meet the goal.  Of those bars it was correct 57.6% of the time.   This represents a  20.8% lift compared to trading random entry points.  The indicators used were primitive so results could improve with additional work. 

## Overview
>When first learning about stock trading I learned a general rule that if you can predict price movement correctly more than 50% of the time you can make a profit trading stocks provided:
>* Your losses per trade are equal same or smaller than your wins. 
>* Your trades execute at the price expected.
>* You can buy and sell the volume of stocks desired.
>* Your trading costs combined for winning and loosing trades are less than profits.
> ##### Goal
>>Our goal then is to use machine learning to help us identify when to Buy a given stock so more than 50% of the time price will rise by a predicted amount.    If we can be accurate a higher percentage of predictions then net profit will be higher provided we can identify a sufficient number of trades to make it worth our effort.

### Compute net profit

No system will make 100% accurate predictions  so your profits for any given amount of time will be equal to  (sum of profits from winning trades) - ((sum of losses from loosing trades) + (sum of trading costs).

### Downloading Data

For this example I downloaded SPY data from yahoo for the period from 2010 to 2016.   I chose this time frame because trading patterns have changed as automated trading has increased which means that data before 2010 is likely to have different patterns.  Since our Machine Learning depends on recognizing patterns using training data that has patterns similar to our current patterns is essential. 

> >   The script to download the SPY data is [yahoo-stock-download.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/yahoo-stock-download.py) but it can easily be changed to download other symbols.  The data file saved is  [SPY.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/SPY.csv) 

#### Sample of CSV File downloaded
      Date,Open,High,Low,Close,Volume,Adj Close
      2017-01-26,229.399994,229.710007,229.009995,229.330002,58208000,229.330002
      2017-01-25,228.699997,229.570007,228.509995,229.570007,80932500,229.570007
      2017-01-24,226.399994,228.080002,226.270004,227.600006,89144800,227.600006
      2017-01-23,226.740005,226.809998,225.270004,226.149994,72331900,226.149994
      2017-01-20,226.699997,227.309998,225.970001,226.740005,116431700,226.740005
      2017-01-19,226.839996,227.00,225.410004,225.910004,65357500,225.910004
      2017-01-18,226.539993,226.800003,225.899994,226.75,51146400,226.75

## Converting Bar Data Machine Learning Data

The raw numbers from stock bars provides relatively value free information.  Many indicators such as SMA, EMA and  RSI have been invented  to help humans extract patterns from how the stock prices change over time.   These indicators provide data that can be useful when trying to predict future stock prices.

Some of the things I found provides interesting and useful data are:

* Percentage of Change compared to a 30, 60, 90, etc day high.  

* Percentage of change compared to a 30, 60, 90, etc day  low

* Slope of change compared to some point in the past.

* Slope of change for a derived indicator for some point in the such as comparing the SMA(30) for the current bar to the SMA(30)  10 days ago.   

* > If we convert this amount of change between these to points and divide by the starting  value if gives us portion of change.  We can convert this to a slope by dividing by the number of days between the two bars.

The number and diversity of possible measurements is nearly infinite but our goal is not to teach  people how to implement new indicators but rather to generate some data using indicators that can be used as an example of how the Quantized Classifier can predict future stock prices.      Remember that better input data can improve the classifiers ability to accurately predict future prices. 

> > One value the Quantized classifier can provide is it can help identify which indicators deliver predictive value and which ones are just noise.    This form of guidance may be more valuable than the core classification capability. 

For this example I chose to use the slope of the SMA(30) comparing the current value against bars in the past.   I wanted to give the system some ability to detect a longer term down trend followed by a medium term counter trend followed by a short term turn around.    With this in mind I had it measure the slope of change for several points in in the past 3,6,12,20,30,60,90 bars.    This ended up producing a new data set with one row for each bar in the original file except I throw away the first 30 bars of data because the SMA are not valid until you have 30 days of data.   The Machine learning files much use integer class ID so there is a step required to map 

Since we needed to split the data to allow training on the early data while reserving more recent data for testing I went ahead and had the system create  the [spy.slp30.test.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/spy.slp30.test.csv) and [spy.slp30.train.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/spy.slp30.train.csv)    The script that reads the bar file [SPY.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/SPY.csv) is  [stock-prep-sma.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/stock-prep-sma.py)

We always allow machine learning engines to train on a part of the data then test how well they learned by running against part of the data they have never seen before. More accurate prediction of results on new data indicates either a good algorithm, a good set of input data or both.   The amount of data used for training for test can be adjusted in  [stock-prep-sma.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/stock-prep-sma.py)   This script is only intended as an example but it could easily be extended to include other indicators and to save a map file to make it easy to map row numbers back to bar dates.

## Sample Machine Learning Input Output generated

      class,sl3,sl6,sl12,sl20,sl30,sl60,sl90
      0,-0.004716421918704945,-0.0039838835243605365,-0.0023182429694179247,-0.0021709275598856426,-0.0019349373654685945,-0.001531443306210873,-0.0014528912280701756
      0,-0.005119112203832877,-0.0023261327346335765,-0.002443043586789742,-0.0021442751932373373,-0.0020344815550956113,-0.0016234031922049785,-0.0015956356311827575
      0,-0.0068310769443658765,-0.0038920815714912796,-0.003200659002496656,-0.002421785335014957,-0.0021096685077146658,-0.0016522511692599723,-0.0016232704787932433
      1,-0.005501746054551725,-0.005070161153104081,-0.002118519259259257,-0.002757670607052132,-0.0020246162990903587,-0.0015659866694324983,-0.0016211815437608026

> These rows can be mapped back to the source BAR data but there is also a command option but that will require another script that can map our row numbers back to Bar Dates. 
>
> **Known flaw:**   [stock-prep-sma.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/stock-prep-sma.py) currently considers bars that we started analyzing a failure if we run out of data before they rise or fall by 1%.   A better solution would be to omit those bars from the Test set because this can cause failures to be reported where the final state for that bar is not really known.  This could understand the sucess of the engine.

##Running the Classifer 

The classifier is invoked by a shell script [classifyTestStockspy.bat](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/classifyTestStockspy.bat)  The actual commands is shown here to allow me to explain some of what is done.
>```
>classifyFiles -train=data/spy.slp30.train.csv -test=data/spy.slp30.test.csv -numBuck=30 -testOut=tmpout/spy.slp30.out.csv -doOpt=true -optrandomize=false -optMaxTime=25 -OptClassId=1
>```
- -train - location it will read training data from
- -test - location is will read test data from.  This would be -class if using to predict against current data.
- -numBuck - is how we divide data elements into groups internally in the engine.  When -doOpt is true the engine is allowed to change this on a feature by feature basis.
- -doOpt - is set to true when you want the optimizer to run. 
- -optMaxTime is the maximum amount of time in seconds optimizer is allowed to run. 

#### Output from Classifier

      numRow=174  sucCnt=114 precis=0.6551724 failCnt=60 failPort=0.3448276
##### Summary By Class

      Train Probability of being in any class
      class=0,  cnt=2635  prob=0.6694215
      class=1,  cnt=1298  prob=0.3305785
      Num Train Row=1573 NumCol=8
    
      RESULTS FOR TEST DATA
      class=0 ClassCnt=107 classProb=0.61494255, Predicted=138 Correct=92  recall=0.8598131  Prec=0.6666667 Lift=0.051724136
      class=1 ClassCnt=64 classProb=0.3678161, Predicted=33 Correct=19  recall=0.296875  Prec=0.57575756 Lift=0.20794147
      Finished ClassifyTestFiles()

#####  Sample of Results by Row
This output is also saved in the file named -testOut paramter but is changed slightly because the system generates multiple files under some conditions. The file name actually generated this time is tmpout/spy.slp30.out.sum.csv.   This is the actual file you would read to when using the predicted values to make trades.     

      ndx,bestClass,bestProb,actClass,status
      0,0,4.0847554,0,ok
      1,0,4.0847554,0,ok
      2,0,4.0847554,0,ok
      3,0,4.0847554,1,fail
      4,0,4.590989,0,ok
      5,1,4.1749053,0,fail
      6,1,4.1749053,0,fail
      7,1,6.336764,1,ok
      8,1,5.7136507,1,ok
      9,1,5.7171845,0,fail
      10,1,5.82789,1,ok
      11,1,5.7298274,1,ok
      12,1,5.7298274,1,ok
      13,1,4.189946,1,ok



# Wrap up Summary

This is a super simple example. There are all kinds of enhancements some in the engine and others in the indicators. EG: A indicator showing the % over to 30 day low and % under 30 day max might give good predictive input. 

This is included with our [free open source classifier](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition)I recently released. It is still rough in a lot of places but is yielding good test results. 

My hope is that people will see what they get for free and then be willing to [pay me](http://BayesAnalytic.com/contact) to make enhancements to meet their needs and integrate it into their systems. 

Note: The optimizer currently does not save the settings found so the actual results can vary by run.   I am working on that feature. 

**I hope this helps feel free to contact me with questions**

Thanks [Joe Ellsworth](http://BayesAnalytic.com/contact)
Machine Learning Algorihtms Scientist & Consultant.



# Improving Predictions for important classes using the optimizer 

The Optimizer seeks to find relationships in the data that would allow it to improve prediction accuracy and to reduce the negative contribution from data that has negative impact on prediction accuracy.   

When predicting stocks for this usecase what we really care about is that when we think a stock will go up it really does go up by at least our 1% before it drops by more than 1%.    This means that predicting for class 1 the stock rising is more important than class 0 the stock falling simply because we will ignore class 0 predictions but if the system predicts a class 1 then we will buy the stock and will loose money when the system is wrong.   

As you can see from the two results below the version without the optimizer was accurate 63.8% of the time but it was only accurate 58% of the time when predicting for class 1.     The optimized version was accurate 65.4% of the time but class 1 received a much larger boost so it improved to 71.4% accurate.    Any level of accuracy above 50% can produce a profitable system. 

The randomizer is essentially a semi-random permutation based system so the results will vary from pass to pass depending on the data but  they can improve prediction accuracy  under specific conditions.   Part of the art of applying these systems is learning how to apply them to the current problem. 

#### Example Output with the Optimizer

```
numRow=174  sucCnt=112 precis=0.6436782 failCnt=62 failPort=0.3563218
Summary By Class
Train Probability of being in any class
class=0,  cnt=2635  prob=0.6694215
class=1,  cnt=1298  prob=0.3305785
Num Train Row=1573 NumCol=8

RESULTS FOR TEST DATA
  Num Train Rows=174
class=0 ClassCnt=109 classProb=0.62643677, Predicted=167 Correct=107  recall=0.98165137  Prec=0.6407186 Lift=0.014281809
class=1 ClassCnt=65 classProb=0.37356323, Predicted=7 Correct=5  recall=0.07692308  Prec=0.71428573 Lift=0.3407225
Finished ClassifyTestFiles()
```

#### Example Output without the Optimizer

```
numRow=174  sucCnt=111 precis=0.63793105 failCnt=63 failPort=0.36206895
Summary By Class
Train Probability of being in any class
class=1,  cnt=520  prob=0.3305785
class=0,  cnt=1053  prob=0.6694215
Num Train Row=1573 NumCol=8

RESULTS FOR TEST DATA
  Num Train Rows=174
class=1 ClassCnt=65 classProb=0.37356323, Predicted=12 Correct=7  recall=0.10769231  Prec=0.5833333 Lift=0.20977008
class=0 ClassCnt=109 classProb=0.62643677, Predicted=162 Correct=104  recall=0.95412844  Prec=0.6419753 Lift=0.015538514

```

