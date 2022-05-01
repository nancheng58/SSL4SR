# Dataset Usage

note: **All data sort by timestamp.**

Index of user and item start at 0

## Introduction

For each dataset, we filter out and delete users and items with less than five interactions.

Please download the dataset file in the code by yourself if you want to reprocess data.

For some models, because side information is required, there is independent data processing code file in the corresponding folder.

## Amazon Beauty

data type of **train.txt,test.txt,valid.txt**

| uid  | iid  | time stamp |
| ---- | ---- | ---------- |

## Yelp

data type of **train.txt,test.txt,valid.txt**

We fitter data by time which data_min <= timestamp <= data_max, date_max = '2019-12-31 00:00:00' and date_min = '2019-01-01 00:00:00'

| uid  | iid  | time stamp |
| ---- | ---- | ---------- |

## Movie-Len 1M

data type of **train.txt,test.txt,valid.txt**

| uid  | iid  | time stamp |
| ---- | ---- | ---------- |
