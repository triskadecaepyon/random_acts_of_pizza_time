library(plyr)
library(data.table)
library(ggplot2)
library(lubridate)
library(qdap)
library(gridExtra)

load(file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train.Rdata')
request <- paste(train_df_frame$request_title, train_df_frame$request_text_edit_aware)
request <- tolower(request)
request <- gsub('[^[:alpha:]]', ' ', request)

remove_space = function(x) {
  x = gsub('^ +', '', x)
  x = gsub(' +$', '', x)
  x = gsub(' +', ' ', x)  
}
request <- remove_space(request)
train_df_frame$request <- request

# Word frequency for each request 
request_word <- strsplit(train_df_frame$request_text_edit_aware, ' +')
#pad an extra white space in front of each word
request_word <- lapply(request_word, lapply, function(x) paste0(' ', x))
request_word <- lapply(request_word, unlist)

count_freq <- function(x, categories) {
  l = length(grep(categories, x))
  return(l / length(x))
}

# Based on the Author's paper, we split into the following topics
money <- ' money| check| now| broke| week| until| time| last|day| when| paid| next| first|night| after| tomorrow| month| while| account| before| long| rent| buy| bank| still| bill| ago| cash| due| soon| past| never|check| spent| year| poor| till| morning| dollar| financial| hour| evening| credit| budget| loan| buck| deposit| current| pay'
job <- ' work| job| check| employ| interview| fire| hire'
student <- ' college| student| school| roommate| study| university| final| semester| class| project| dorm| tuition'
family <- ' family| wife| parent| mother| husband| dad| son| daughter| father| mom| mum'
craving <- ' friend| boyfriend| girlfriend| crave| craving| birthday| boyfriend| celebrat| party| parties| game| movie| film| date| drunk| beer| invite| drink| waste'

freq_money <- unlist(lapply(request_word, count_freq, money))
freq_job <- unlist(lapply(request_word, count_freq, job))
freq_student <- unlist(lapply(request_word, count_freq, student))
freq_family <- unlist(lapply(request_word, count_freq, family))
freq_craving <- unlist(lapply(request_word, count_freq, craving))

# Calculate deciles (0 frequencies are represented as 0 decile)
quan_money <- quantile(freq_money[freq_money > 0], seq(0, .9, .1), na.rm = TRUE)
quan_job <- quantile(freq_job[freq_job > 0], seq(0, .9, .1), na.rm = TRUE)
quan_student <- quantile(freq_student[freq_student > 0], seq(0, .9, .1), na.rm = TRUE)
quan_family <- quantile(freq_family[freq_family > 0], seq(0, .9, .1), na.rm = TRUE)
quan_craving <- quantile(freq_craving[freq_craving > 0], seq(0, .9, .1), na.rm = TRUE)

train_df_frame$de_money <- findInterval(freq_money, quan_money)
train_df_frame$de_job <- findInterval(freq_job, quan_job)
train_df_frame$de_student <- findInterval(freq_student, quan_student)
train_df_frame$de_family <- findInterval(freq_family, quan_family)
train_df_frame$de_craving <- findInterval(freq_craving, quan_craving)

# Explore deciles and outcome
train_df_frame$requester_received_pizza <- as.factor(train_df_frame$requester_received_pizza)
#Split data frame for each variables, summarize, 
sr_money <- data.frame(cbind('money', ddply(train_df_frame, .(de_money), summarize, sr <- sum(requester_received_pizza == T) / length(requester_received_pizza))))
sr_job <- data.frame(cbind('job', ddply(train_df_frame, .(de_job), summarize, sr <- sum(requester_received_pizza == T) / length(requester_received_pizza))))
sr_student <- data.frame(cbind('student', ddply(train_df_frame, .(de_student), summarize, sr <- sum(requester_received_pizza == T) / length(requester_received_pizza))))
sr_family <- data.frame(cbind('family', ddply(train_df_frame, .(de_family), summarize, sr <- sum(requester_received_pizza == T) / length(requester_received_pizza))))
sr_craving <- data.frame(cbind('craving', ddply(train_df_frame, .(de_craving), summarize, sr <- sum(requester_received_pizza == T) / length(requester_received_pizza))))
sr_narrative <- rbindlist(list(sr_money, sr_job, sr_student, sr_family, sr_craving))

setnames(sr_narrative, c('narrative', 'decile', 'success_rate'))
pn = 
  ggplot(sr_narrative, aes(x = decile, y = success_rate, colour = narrative, group = narrative)) +
  geom_line() + ggtitle('Success rate vs. narrative') +
  scale_x_continuous(breaks = seq(0, 11, 1), name = 'Narrative Declie') + scale_y_continuous(name = 'Success Rate')
ggsave(pn, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/narrative.jpg')

#create temporal features
train_df_frame$request_time <- as.Date(as.numeric(train_df_frame$unix_timestamp_of_request_utc) / (3600*24), origin = '1970-01-01')
summary(train_df_frame$request_time) 

# Let see day of month
community_age <- train_df_frame$request_time - as.Date('2011-01-01')
quan_community_age <- quantile(community_age, seq(0, .9, .1))
train_df_frame$de_community_age <- findInterval(community_age, quan_community_age)
sr_community_age <- ddply(train_df_frame, .(de_community_age), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
#COMMUNITY AGE
pic_community_age <-
  ggplot(sr_community_age, aes(x = de_community_age, y = success_rate)) + geom_line(colour = '#f48e4a') + 
  ggtitle('Success Rate vs. Community Age') +
  scale_x_continuous(breaks = seq(0, 10, 1), name = 'Community Age Declie') +
  scale_y_continuous(name = 'Success Rate')
print(pic_community_age)
#DAY IN MONTH
train_df_frame$month_part <- as.factor(as.POSIXlt(train_df_frame$request_time)$mday <= 15)
sr_month <- ddply(train_df_frame, .(month_part), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_month <- ggplot(sr_month, aes(x = month_part, y = success_rate)) + geom_bar(stat = 'identity', fill = '#9d4af4') + ggtitle('Success Rate vs. Day of month') + scale_x_discrete(limits = c('TRUE', 'FALSE'), labels = c('First half of month', 'Second half of month'), name = 'Day of month') +
  scale_y_continuous(name = 'Success Rate')
print(pic_month)
# GRATITUDE
gratitude <- lapply(train_df_frame$request, function(x) as.factor(length(grep('thank|appreciate|advance', x)) > 0))
train_df_frame$gratitude <- unlist(gratitude)
sr_gratitude <- ddply(train_df_frame, .(gratitude), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_gratitude <- 
  ggplot(sr_gratitude, aes(x = gratitude, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#f44adc') + 
  ggtitle('Success Rate vs. Gratitude') +
  scale_x_discrete(limits = c('TRUE', 'FALSE'), labels = c('Gratitude expressed', 'Gratitude not expressed'), name = 'Gratitude') +
  scale_y_continuous(name = 'Success Rate')
print(pic_gratitude)
# HAS HYPERLINKS 
hyperlink <- lapply(train_df_frame$request, function(x) as.factor(length(grep('http', x)) > 0))
train_df_frame$hyperlink <- unlist(hyperlink)
sr_hyperlink <- ddply(train_df_frame, .(hyperlink), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_hyperlink <- 
  ggplot(sr_hyperlink, aes(x = hyperlink, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#29e7c5') + 
  ggtitle('Success Rate vs. Hyperlink Included') +
  scale_x_discrete(limits = c('TRUE', 'FALSE'), labels = c('With Hyperlink', 'Without Hyperlink'), name = 'Hyperlink') +
  scale_y_continuous(name = 'Success Rate')
print(pic_hyperlink)
#RECIPROCITY
reciprocity <- lapply(train_df_frame$request, function(x) as.factor(length(grep('pay.+forward|pay.+back|return.+favor|repay', x)) > 0))
train_df_frame$reciprocity = unlist(reciprocity)
sr_reciprocity <- ddply(train_df_frame, .(reciprocity), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_reciprocity <- 
  ggplot(sr_reciprocity, aes(x = reciprocity, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#3929e7') + 
  ggtitle('Success Rate vs. Reciprocity') +
  scale_x_discrete(limits <- c('TRUE', 'FALSE'), labels = c('With Reciprocity', 'Without Reciprocity'), name = 'Reciprocity') +
  scale_y_continuous(name = 'Success Rate')
print(pic_reciprocity)
#SENTIMENT
sentiment <- lapply(train_df_frame$request, function(x) polarity(x)[[1]]$polarity)
sentiment_vector <- unlist(sentiment)
sentiment_positive_median <- median(sentiment_vector[sentiment_vector > 0])
sentiment_negative_median <- median(sentiment_vector[sentiment_vector < 0])
train_df_frame$sentiment_positive <- as.factor(sentiment_vector > sentiment_positive_median)
train_df_frame$sentiment_negative <- as.factor(sentiment_vector < sentiment_negative_median)
sr_sentiment_positive <- ddply(train_df_frame, .(sentiment_positive), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
sr_sentiment_negative <- ddply(train_df_frame, .(sentiment_negative), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_sentiment_positive <- 
  ggplot(sr_sentiment_positive, aes(x = sentiment_positive, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#29e73f') + 
  ggtitle('Success Rate vs. Positive sentiment') +
  scale_x_discrete(limits = c('TRUE', 'FALSE'), labels = c('Positive Expressed', 'Not Expressed'), name = 'Positive sentiment') +
  scale_y_continuous(name = 'Success Rate')
print(pic_sentiment_positive)
pic_sentiment_negative <- 
  ggplot(sr_sentiment_negative, aes(x = sentiment_negative, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#295ce7') + 
  ggtitle('Success Rate vs. Negative Sentiment') +
  scale_x_discrete(limits = c('TRUE', 'FALSE'), labels = c('Negative Expressed', 'Not Expressed'), name = 'Negative sentiment') +
  scale_y_continuous(name = 'Success Rate')
print(pic_sentiment_negative)
#REQUEST BODY LENGTH - EFFORT
request_length <- lapply(train_df_frame$request, function(x) nchar(x) / 100)
train_df_frame$request_length <- unlist(request_length)
pic_request_length <-
  ggplot(train_df_frame, aes(x = requester_received_pizza, y = request_length)) +
  geom_boxplot(fill = '#d0f44a') +
  ggtitle('Success Rate vs. Request Length') +
  scale_x_discrete(limits <- c('TRUE', 'FALSE'), labels = c('Success', 'Fail'), name = 'Request outcome') +
  scale_y_continuous(name = 'Requent length (in 100 words)')
print(pic_request_length)
#KARMA DECILE
karma <- as.numeric(train_df_frame$requester_upvotes_minus_downvotes_at_request)
quan_karma <- quantile(karma, seq(0, .9, .1))
train_df_frame$de_karma <- findInterval(karma, quan_karma)
sr_karma <- ddply(train_df_frame, .(de_karma), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_karma <- 
  ggplot(sr_karma, aes(x = de_karma, y = success_rate)) + geom_line(colour = '#694af4') + 
  ggtitle('Success Rate vs. Karma') +
  scale_x_continuous(breaks = seq(0, 11, 1), name = 'Karma Declie') +
  scale_y_continuous(name = 'Success Rate')
print(pic_karma)
# POST BEFORE SINCE LAST POST
train_df_frame$posted_before = as.factor(as.numeric(train_df_frame$requester_number_of_posts_on_raop_at_request) > 0)
sr_post_before <- ddply(train_df_frame, .(posted_before), summarize, success_rate = sum(requester_received_pizza == T) / length(requester_received_pizza))
pic_post_before <- 
  ggplot(sr_post_before, aes(x = posted_before, y = success_rate)) + 
  geom_bar(stat = 'identity', fill = '#e2f44a') + 
  ggtitle('Success Rate vs. Have Posted Before on ROAP') +
  scale_x_discrete(limits <- c('TRUE', 'FALSE'), labels = c('Posted before', 'Never posted before'), name = 'Whether the requester has posted on ROAP before') +
  scale_y_continuous(name = 'Success Rate')
print(pic_post_before)

#COMBINE RESULTs
#combine all plots
grid.arrange(pic_community_age, pic_month, pic_gratitude, pic_hyperlink, pic_reciprocity, pic_sentiment_positive, pic_sentiment_negative, pic_request_length, pic_karma, pic_post_before, ncol = 2)

train_select <- subset(train_df_frame, select = c(de_community_age, month_part, gratitude, hyperlink, reciprocity, sentiment_positive, sentiment_negative, request_length, de_karma, posted_before,
                            de_craving, de_family, de_job, de_money, de_student, requester_received_pizza))

save(train_df_frame, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_new.Rdata')
save(train_select, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_select.Rdata')

#QUANTILES
quantiles <- data.frame(do.call(cbind, list(quan_community_age, quan_craving, quan_family, quan_job, quan_money, quan_student, quan_karma)))
names(quantiles) <- c('quan_community_age', 'quan_craving', 'quan_family', 'quan_job', 'quan_money', 'quan_student', 'quan_karma')
save(quantiles, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/quantiles.Rdata')


#############################  TRAIN MODELS ############################# 


