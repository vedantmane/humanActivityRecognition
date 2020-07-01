---
title: "Human Activity Recognition"
author: "Vedant Mane"
date: "July 1, 2020"
output: 
      html_document: 
            keep_md: true
---


```r
if(!dir.exists("activityData")){
      trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
      testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
      dir.create("activityData")
      download.file(trainURL, destfile = "./activityData/training.csv")
      download.file(testURL, destfile = "./activityData/testing.csv")
}
training <- read.csv(file = "./activityData/training.csv")
testing <- read.csv(file = "./activityData/testing.csv")
```
