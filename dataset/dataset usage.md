# Dataset Usage



note: **All data sort by timestamp.**

Index of user and item start at 0

## Amazon Beauty

data type of **train.txt,test.txt,valid.txt**

| uid  | iid  | time stamp | score | review | review time |
| ---- | ---- | ---------- | ----- | ------ | ----------- |

umap.json and imap.json record original userid and itemid .

review time type : "%m %d,%Y", such as "01 30, 2014"

## Yelp

data type of **train.txt,test.txt,valid.txt**

| uid  | iid  | time stamp | score | review time |
| ---- | ---- | ---------- | ----- | ----------- |

umap.json and imap.json record original userid and itemid .

review time type : "%Y-%m-%d %H:%M:%S", such as "2014-10-11 03:34:02"

## Movie-Len 1M

data type of **train.txt,test.txt,valid.txt**

| uid  | iid  | score | time stamp |
| ---- | ---- | ----- | ---------- |

