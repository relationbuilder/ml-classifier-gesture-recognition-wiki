# Stock Price Prediction ML Tutorial

**I provide [consulting services](http://BayesAnalytic.com/contact) to help adapt  [Quantized Classifier](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition) to meet your project needs.**

> > Summary:   During the period tested 32% of all SPY bars met our goal of the market price rising by at least 1% before it dropped by 1%.   This means that if you randomly purchased the stock only 1 time out of 3 would you exit with a 1% profit before you hit a stop loss at 1%.    The classifier was able to increase our win rate to 2 out of 3 purchases. 
> >
> > The Quantized classifier  trained from 1,399 SPY bars while making predictions for  350 bars  ending in Jan-2017.     It predicted 41 bars  would meet the goal.  Of those bars it was correct 65.8% of the time.   This represents a  33.8% lift compared to trading random entry points.  The indicators used were primitive so results could improve with additional work. 

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

### Getting Started

[Download the Quantized Classifier repository](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/downloads) and run the [make_go.bat](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/makeGO.bat?at=default&fileviewer=file-view-default) script with a command console open and with the current working directory set to where you unzipped the repository.   This is explained in more detail in the main [Quantized classifier readme](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition).   

Some of the bat files assume that you have installed [cygwin](https://cygwin.com/install.html) and added the cygwin\bin directory to your [PATH environment variable](http://www.computerhope.com/issues/ch000549.htm). 

You can also close the repository directly using Mercurial  using the following URI: https://joexdobs@bitbucket.org/joexdobs/ml-classifier-gesture-recognition  

### Downloading Data

For this example I downloaded SPY data from yahoo for the period from 2010 to 2016.   I chose this time frame because trading patterns have changed as automated trading has increased which means that data before 2010 is likely to have different patterns.  Since our Machine Learning depends on recognizing patterns using training data that has patterns similar to our current patterns is essential. 

> >   The script to download the SPY data is [yahoo-stock-download.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/yahoo-stock-download.py) but it can easily be changed to download other symbols.  The data file saved is  [SPY.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/SPY.csv)    The baseline data is included with the repository so you only need to run the download script if you want to test against more current data.

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

For this example I chose to use the slope of the Close current value against bars in the past.   I wanted to give the system some ability to detect a longer term down trend followed by a medium term counter trend followed by a short term turn around.    With this in mind I had it measure the slope of change for several points in in the past 3,6,12,20,30,60,90 bars.    

This ended up producing a new data set with one row for each bar in the original file except when using the SMA it throws away the first 30 bars of data because the SMA are not valid until you have N-days of data.   The Machine learning files much use integer class ID so there is a step required to map 

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
> > > The 30 day version of the converted data is included in the repository. You only need to run it again if you change the parameters in the stock-prep-sma.py or if you downloaded new data. 
>
> **Known flaw:**   [stock-prep-sma.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/stock-prep-sma.py) currently considers bars near the end of the input data set a failure under because we run out of data before they rise or fall by 1%.   A better solution would be to omit those bars from the Test set because this can cause failures to be reported where the final state for that bar is not really known.  This could understand the sucess of the engine.

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

##### Summary By Class

    numRow=350  sucCnt=251 precis=0.7171429 failCnt=99 failPort=0.28285712
    Summary By Class
    Train Probability of being in any class
    class=0,  cnt=927  prob=0.66261613
    class=1,  cnt=472  prob=0.33738384
    Num Train Row=1399 NumCol=8
    
    RESULTS FOR TEST DATA
      Num Test Rows=350
    class=1 ClassCnt=112 classProb=0.32, Predicted=41 Correct=27  recall=0.24107143  Prec=0.6585366 Lift=0.33853662
    class=0 ClassCnt=238 classProb=0.68, Predicted=309 Correct=224  recall=0.9411765  Prec=0.7249191 Lift=0.044919074
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



----------------------

## Predict future Silver (SLV) Prices

For silver I chose a harder goal where we wanted to find bars where the price would rise by at least 1.5% before it fell by 0.3%.   That means that the size of our wins would be at least 500% the size of our losses provided we set a stop loss at the 0.3% with a auto exit when stock rose by 1.5%.      

With this magnitude of difference for gains and losses we need a 20% win rate to make a break even.   Anything more than 20% increases profit.      

The Classifier was able to identify 10 out of 501 test bars that it thought would fit this criteria of which 5 turned out to be correct for a win rate of 50% over 2 times what we needed to break even.  

* Data download script [yahoo-stock-download.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/yahoo-stock-download.py) was extended to download daily silver SLV bars back to 2007.  To create [SLV.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/SLV.csv)
* Data conversion  [stock-prep-sma.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/stock-prep-sma.py) was extended to create silver machine learning files using the SMA30 on close.   Produces [https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/slv.slp30.train.csv](slv.slp30.train.csv)  and  [slv.slp30.test.csv](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/data/slv.slp30.test.csv)
* Classification script for [Classification](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/classifyTestStockslv.bat) script for silver added



#### Output from the classifier run SLV

```

Summary By Class
Train Probability of being in any class
class=1,  cnt=719  prob=0.35878244
class=0,  cnt=1285  prob=0.6412176
Num Train Row=2004 NumCol=8

RESULTS FOR TEST DATA
  numRow=501  sucCnt=314 precis=0.62674654 failCnt=187 failPort=0.37325346
  Num Test Rows=501
class=0 ClassCnt=314 classProb=0.62674654, Predicted=491 Correct=309  recall=0.98407644  Prec=0.6293279 Lift=0.002581358
class=1 ClassCnt=187 classProb=0.3732535, Predicted=10 Correct=5  recall=0.026737968  Prec=0.5 Lift=0.1267465
Finished ClassifyTestFiles()
```

# Adding Additional Data sources  

The analysis above was based on technical data derived from directly from BAR data.  There are other sources of data that could be used to add refinement to the predictive capacity of of the system.   They are roughly classed as followed.   

* Company Fundementals Data.

* Time Before and After the Next financials are released

* Important Market influencers such as Fed Announcements.   Market swings can be particularly volatile and so far outside the norm that it will confuse machine learning statisitics immediatly before and following those events.

* Wars and Rumors Wars.    Elections and Fears about Elections.

* News,  Commentary,  Blogging, Tweets, ect producing what is roughly classified as Sentiment data.


##### Fundamental data

Some of this data is easily added to the technical data simply as a extra few columns of data in the CSV files.  The Machine learning classifier doesn't care were the data comes from as long as it can be represented in a number in the CSV that has value for every row.    A perfect example of this could be the companies dept to equity ratio or their rate of increase in sales over the last 2 quarters both of which could influence future stock prices. 

##### Market moving Events 

Other aspects such as the fed announcements are harder to incorporate as a single number since not body knows what they are going to say.   This can be added to the model as a feature but it may be easier to simply lock out trading for the few days before and after these announcements unless you are building a trading system to try  on capitalize on those points of high volatility.

##### Sentiment Data

News and textual sentiment data can be mined and added to the engine both as a market wide sentiment and as sentiment about the company.  In general the [sentiment mining requires different approach](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/docs/text-classification/overview-classification.md) so the the best way to approach it is to use a separate classifier that is digesting the news data and producing numbers that can be added to the exisitng CSV as additional data columns.    

One of the greatest challenges with sentiment data is gaining access to the text containing necessary data in a timely fashion and a reasonable cost.  Many sites that contain valuable commentary that could be mined for sentiment but it is hidden behind pay walls.   Other text like twitter is free but may contain very limited value with lots of noise.          When I find a free source that is worth mining I will add it as an example to for the Quantized classifier. 

One of the more interesting challenges is that some authors have opinions that are more likely to be correct than others.   Any sentiment mining system needs to incorporate a notion of author credibility  and use it to rate sentiment from authors with greater credibility as higher influence than others.

Ultimately sentiment data can be reduced to numbers that are added to the original technical data to help boost accuracy of prediction or it can be used to adjust acceptable risk thresholds for portfolio management.      For example we could have a set of numbers such twitterMarketRise=0.5 meaning that the twitter feed seems to be neutral.     SeekingAlphaIBMBull=0.9  which means the seeking alpha analysts as a set are very bulling thinking IBM will rise.    Since we want to consolidate many sources into a small number of numbers being able to adjust for credibility of the source is critical.    The total number of valid columns is only limited by our imagination but more columns can actually hurt predictive accuracy if they contain only noise with no signal. 



## My Objective

My hope is that this article will inspire some of those working on trading systems to use [Quantized Classifier](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition)  as a component of their solution stack.  I  would love to provide [consulting services](http://BayesAnalytic.com/contact) to help them build a production grade system around the classifier.    I am also willing to provide consulting services to [add features they need](http://BayesAnalytic.com/contact) to Quantized classifier.   

---------------------











































































# Improving Predictions for important classes using the optimizer 

The Optimizer seeks to find relationships in the data that would allow it to improve prediction accuracy and to reduce the negative contribution from data that has negative impact on prediction accuracy.   

When predicting stocks for this usecase what we really care about is that when we think a stock will go up it really does go up by at least our 1% before it drops by more than 1%.    This means that predicting for class 1 the stock rising is more important than class 0 the stock falling simply because we will ignore class 0 predictions but if the system predicts a class 1 then we will buy the stock and will loose money when the system is wrong.   

As you can see from the two results below the version without the optimizer was accurate 63.8% of the time but it was only accurate 58% of the time when predicting for class 1.     The optimized version was accurate 65.4% of the time but class 1 received a much larger boost so it improved to 71.4% accurate.    Any level of accuracy above 50% can produce a profitable system. 

The randomizer is essentially a semi-random permutation based system so the results will vary from pass to pass depending on the data but  they can improve prediction accuracy  under specific conditions.   Part of the art of applying these systems is learning how to apply them to the current problem. 

-------------------

**NOTE:  As of 2017-01-30 I broke the analyzer when adding in another feature.     I want to implement  pre-analyzer before working on fixing the optimizer.  Please [let me know](http://BayesAnalytic.com) if you need the optimizer working and I will boost it's priority.** 

------------------------------



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

