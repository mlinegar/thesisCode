library(tidyverse)
library(stringr)
library(devtools)
# install_github("mlinegar/litMagModelling")
library(litMagModelling)
library(tidytext)
# increase Java's available memory size
options(java.parameters = "-Xmx4g")

politicalAddresses <- read.csv("politicalAddresses.csv", stringsAsFactors = FALSE)
library(dplyr)

# some brief cleaning
library(lubridate)
politicalAddresses$date <- mdy(politicalAddresses$date)
modernPoliticalAddresses <- politicalAddresses[which(lubridate::year(politicalAddresses$date) > 2015),]
# clean up authors
modernPoliticalAddresses$author <- str_replace_all(modernPoliticalAddresses$author, "- I{1,2}","")
modernPoliticalAddresses$author <- str_trim(modernPoliticalAddresses$author)
modernPoliticalAddresses[which(modernPoliticalAddresses$author=="Donald J. Trump"),]$author <- "Donald Trump"

# create row ids and author-speech ids
modernPoliticalAddresses <- mutate(modernPoliticalAddresses, id = rownames(modernPoliticalAddresses))
modernPoliticalAddresses <- mutate(modernPoliticalAddresses, author_speech_id = paste(author, id, sep = "_"))
modernPoliticalAddresses$text <- tolower(modernPoliticalAddresses$text)
modernPoliticalAddresses$text <- str_replace_all(modernPoliticalAddresses$text, "\n", " ")
modernPoliticalAddresses$text <- str_replace_all(modernPoliticalAddresses$text, "<br />", " ")

# create data.frame of metadata so we can join on this later
modernPoliticalAddresses.metadata <- modernPoliticalAddresses %>% select(author_speech_id, author, date)
# may want to selectively replace punctuation: replace . as " " if word.length >5 (ex ".")
# else replace . ""
# code with help from stackoverflow user G. Grothendieck, here: 
# https://stackoverflow.com/questions/46394301/replacing-punctuation-in-string-in-different-ways-by-word-length-in-r/46395170#46395170
library(gsubfn)
# explanation:
# looks for a delimiter, up to 6 non-space characters and a delimiter 
# for any such match it runs the anonymous function specified in formula notation 
# in the second argument of gsubfn. That anonymous function removes any periods in the match. 
# In what is left the gsub replaces each period with a space.
pat <- "(?<=^| )(\\S{1,6})(?=$| )"
modernPoliticalAddresses$text <- gsub("[.]", " ", gsubfn(pat, ~ gsub("[.]", "", ..1), modernPoliticalAddresses$text, perl = TRUE))

# just doing this crudely for now
# doing this for apostrophes eventually gives a conjunction topic, which isn't ideal
# any better idea?
modernPoliticalAddresses$text <- str_replace_all(modernPoliticalAddresses$text, "[\'\"]", "")
modernPoliticalAddresses$text <- str_replace_all(modernPoliticalAddresses$text, "[[:punct:]]", " ")

# should probably do full name parsing first to prevent doubling of candidates when their
# first and last name appears
# trying to do this below with twotokenlist
# mapping usa/united states( of america)? to unitedstates
# isis, islamic state, and isil map to isis because they both appear relatively infrequently
twotokenlist <- read.csv("twotokenlist2.csv", stringsAsFactors = FALSE)
for(phrase in 1:nrow(twotokenlist)){
  modernPoliticalAddresses$text <- str_replace_all(modernPoliticalAddresses$text, twotokenlist$from[phrase], twotokenlist$to[phrase])
}

# big problem is in interviews, when candidates appear to talk about themselves a lot because their 
# name is mentioned on the transcript. Gotta fix that.
# because the twotokenlist (above) should have replaced all first/last/combo names with a single string
# we *should* be able to proceed in this simple manner

modernPoliticalAddresses <- modernPoliticalAddresses %>% mutate(text = str_replace_all(text, 
                                      removePunctuation(tolower(str_replace_all(author, " ", ""))), " "))

# need to stem/lemmatize the corpus
library(textstem)
modernPoliticalAddresses$text <- lemmatize_strings(modernPoliticalAddresses$text)


# and now tokenize!
custom_stopwords <- data.frame(word = c("applause", "mr", "will", "sen", "can", "go", "get", "say", "much",
                                        "cheer", "applause", "cd", "boo", "like", "think", "just", "applpause",
                                        "crosstalk", "matthews", "laughter", "jansing", "burnett", "tapper", 
                                        "cheering", "cheers", "bash", "camerota", "dickerson", "mitchell",
                                        "inaudible", "maddow", "anderson", "cooper", "um", "megan", "kelly",
                                        "chris", "hayes", "shepard", "smith", "moderator", "todd", "chuck", 
                                        "th", "youre", "hell", "hes", "shes", "ive", "uh", "weve", "theyre",
                                        "theres", "didn", "didnt", ", illl", "youre", "whats", "doesnt", 
                                        "doesn", "theyve", "cuomo", "youve", "joe", "michael", "wont", "week",
                                        "ago", "wasn"), 
                               stringsAsFactors = FALSE)

modernPoliticalAddresses.tidy <- modernPoliticalAddresses %>% 
  unnest_tokens(word, text) %>% 
  anti_join(stop_words) %>%
  anti_join(custom_stopwords)


# calculate tf-idf
modernPoliticalAddresses.tfidf <- modernPoliticalAddresses.tidy %>%
  count(author_speech_id, word, sort = TRUE) %>%
  ungroup() %>%
  bind_tf_idf(word, author_speech_id, n)

# fairly arbitrary cutoff point: idf less than 0.6, greater than 5
# removing less than 0.6 might be too harsh, but let's see how it does
# just make sure to lemmatize first
wordsToRemove.tfidf <- modernPoliticalAddresses.tfidf %>%
  filter(idf < 0.5 | idf > 5) %>% 
  transmute(word = word, idf = idf) %>%
  unique()

modernPoliticalAddresses.tidy <- modernPoliticalAddresses.tidy %>%
  anti_join(wordsToRemove.tfidf)

# and now stitch everything back together so we can run it with MALLET!
# much of this code was directly inspired/taken from Text Mining with R: A Tidy Approach by Julia Silge and David Robinson

modernPoliticalAddresses.collapsed <- modernPoliticalAddresses.tidy %>%
  mutate(word = str_replace(word, "'", ""), id = author_speech_id) %>%
  group_by(author_speech_id) %>%
  summarize(text = paste(word, collapse = " "), date = first(date), id = first(id), author = first(author))

# have to do this because at some point the text column became a factor apparently
modernPoliticalAddresses.collapsed <- data.frame(modernPoliticalAddresses.collapsed, stringsAsFactors = FALSE)
# create an empty file of "stopwords"
file.create(empty_file <- tempfile())

# we also need to make an ID column
ntopics <- 50
topic.model <- make_model(modernPoliticalAddresses.collapsed, n.topics = ntopics, textcolname = "text", stopListFile = empty_file, burnIn = 200, numRuns = 500, optFreq = 25)


# exploring the topic.model below
library(mallet)
doc.topics <- mallet.doc.topics(topic.model, normalized = TRUE, smoothed = TRUE)
topic.words <- mallet.topic.words(topic.model, normalized = TRUE, smoothed = TRUE)

# looks like topic.labels doesn't actually work - contact maintainer? always prints 3
topic.labels <- mallet.topic.labels(topic.model, topic.words, 5)
plot(mallet.topic.hclust(doc.topics, topic.words, 0.7), labels=topic.labels)


# document-topic pairs (which documents express which topics?)
doc.topics.tidy <- tidy(topic.model, matrix = "gamma")
# join with metadata
doc.topics.tidy <- doc.topics.tidy %>% left_join(modernPoliticalAddresses.metadata, by = c("document" = "author_speech_id"))
# per-topic-word probs (which words belong to which topic? What is the probability of each word by topic?)
topic.words.tidy <- tidy(topic.model, matrix = "beta")

topic.labels <- topic.words.tidy %>% 
  group_by(topic) %>%
  top_n(7, beta) %>%
  summarize(top_words = paste(term, collapse = " "))

# label all the topics
doc.topics.tidy <- doc.topics.tidy %>% left_join(topic.labels)
topic.words.tidy <- topic.words.tidy %>% left_join(topic.labels)


# logic checks: does summing over all betas (word-probs) by topic give 1? (yes)
topic.words.tidy %>% group_by(topic) %>% summarise(topic_sum = sum(beta))
# does summing over all gammas (topic-probs) by document give 1? (yes)
doc.topics.tidy %>% group_by(document) %>% summarise(doc_sum = sum(gamma))


