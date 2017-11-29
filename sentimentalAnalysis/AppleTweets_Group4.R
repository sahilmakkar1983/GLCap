install.packages("SnowballC")
library(SnowballC)
install.packages("tm")
library(tm)
install.packages("ggplot2")
library(ggplot2)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("wordcloud")
library(wordcloud)
install.packages("topicmodels")
library(topicmodels)
install.packages("data.table")
library(data.table)
install.packages("stringi")
library(stringi)
install.packages("qdap")
library(qdap)
install.packages("dplyr")
library(dplyr)
install.packages("rJava")
library(rJava)

# Set directory and read data
setwd("C:/GL/Materials/Web_SocialMedia_Analytics")
tweets.df <- read.csv("tweets.csv")
names(tweets.df)

# Convert char date to correct date format
tweets.df$created <- as.Date(tweets.df$created, format= "%d-%m-%y")
str(tweets.df)
# Remove character string between < >
tweets.df$Tweet <- genX(tweets.df$Tweet, " <", ">")  #--- Needs rJava
head(tweets.df)

# Create document corpus with tweet text and clean up
myCorpus<- Corpus(VectorSource(tweets.df$Tweet)) 

myCorpus <- tm_map(myCorpus,tolower)
myStopWords<- c((stopwords('english')),c("apple", "lol", "dear", "hey", "freak","@","apples"))
myCorpus<- tm_map(myCorpus,removeWords , myStopWords) 
#myCorpus <- tm_map(myCorpus,removeNumbers)
myCorpus <- tm_map(myCorpus,removePunctuation)
myCorpus <- tm_map(myCorpus,stripWhitespace)

head(tweets.df)

dtm1 <- TermDocumentMatrix(myCorpus)

(freq.terms <- findFreqTerms(dtm1, lowfreq = 40))

term.freq <- rowSums(as.matrix(dtm1))
term.freq <- subset(term.freq, term.freq > 40)
df <- data.frame(term = names(term.freq), freq= term.freq)

ggplot(df, aes(reorder(term, freq),freq)) + theme_bw() + geom_bar(stat = "identity")  + coord_flip() +labs(list(title="Term Frequency Chart", x="Terms", y="Term Counts")) 

dtm <- DocumentTermMatrix(myCorpus)
#tdm<- TermDocumentMatrix(myCorpus, control= list(wordLengths= c(1, Inf)))

m <- as.matrix(dtm)

v <- sort(colSums(m),decreasing=TRUE)

head(v,14)

words <- names(v)

d <- data.frame(word=words, freq=v)

wordcloud(d$word,d$freq,min.freq=85)

# Identify and plot word correlations. For example - love, like, new. It depends on the business objective
WordCorr <- apply_as_df(myCorpus[1:500], word_cor, word = "iphone", r=.25)
plot(WordCorr)

qheat(vect2df(WordCorr[[1]], "word", "cor"), values=TRUE, high="red",
      digits=2, order.by ="cor", plot = FALSE) + coord_flip()

df <- data.frame(text=sapply(myCorpus, `[[`, "content"), stringsAsFactors=FALSE)
head(unique(df[grep("like", df$text), ]), n=10)

findAssocs(dtm, "ipad", 0.2)
findAssocs(dtm, "will", 0.2)

#dtm <- as.DocumentTermMatrix(tdm)

rowTotals <- apply(dtm , 1, sum)

NullDocs <- dtm[rowTotals==0, ]
dtm   <- dtm[rowTotals> 0, ]

if (length(NullDocs$dimnames$Docs) > 0) {
  tweets.df <- tweets.df[-as.numeric(NullDocs$dimnames$Docs),]
}

lda <- LDA(dtm, k = 5) # find 5 topic
term <- terms(lda, 5) # first 7 terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))


topics<- topics(lda)
topics<- data.frame(date=(tweets.df$created), topic = topics)
qplot (date, ..count.., data=topics, geom ="density", fill= term[topic], position="stack")

install.packages("devtools")
require(devtools)

install_github('sentiment140', 'okugami79')

library(sentiment)

# Use qdap polarity function to detect sentiment
sentiments <- polarity(tweets.df$text)

sentiments <- data.frame(sentiments$all$polarity)

sentiments[["polarity"]] <- cut(sentiments[[ "sentiments.all.polarity"]], c(-5,0.0,5), labels = c("negative","positive"))

table(sentiments$polarity)

#####Sentiment Plot by date {r Sentiment Plot}
sentiments$score<- 0
sentiments$score[sentiments$polarity == "positive"]<-1
sentiments$score[sentiments$polarity == "negative"]<- -1
sentiments$date <- as.IDate(tweets.df$created)
result <- aggregate(score ~ date, data = sentiments, sum)
plot(result, type = "l")


##### Stream Graph for sentiment by date {r Stream Graph plotting}

Data<-data.frame(sentiments$polarity)
colnames(Data)[1] <- "polarity"
Data$Date <- tweets.df$created
Data$text <- NULL
Data$Count <- 1


graphdata <- aggregate(Count ~ polarity + as.character.Date(Date),data=Data,FUN=length)
colnames(graphdata)[2] <- "Date"
str(graphdata)

##### StreamGraph {r Type III,warning=FALSE, echo=FALSE}

graphdata %>%
  streamgraph(polarity, Count, Date) %>%
  sg_axis_x(20) %>%
  sg_axis_x(1, "Date","%d-%b") %>%
  sg_legend(show=TRUE, label="Polarity: ")

