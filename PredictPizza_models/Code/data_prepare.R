library(tm)
library(rjson)
library(KernSmooth)
library(SnowballC)
library(wordcloud)
library(plyr)
options(java.parameters = "-Xmx4g")
library(topicmodels)
library(slam)
library(caret)
library(skmeans)
library(NLP)
#library(openNLP)
options(stringsAsFactors = F)


# Read data from json file
train_df <- fromJSON(file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train.json') # Convert JSON object to R object

#Initialises necessary vectors.
giver_username_if_known <- c()
number_of_downvotes_of_request_at_retrieval <- c()
number_of_upvotes_of_request_at_retrieval <- c()
post_was_edited <- c()
request_id <- c()
request_number_of_comments_at_retrieval <- c()
request_text <- c()
request_text_edit_aware <- c()
request_title <- c()
requester_account_age_in_days_at_request <- c()
requester_account_age_in_days_at_retrieval <- c()
requester_days_since_first_post_on_raop_at_request <- c()
requester_days_since_first_post_on_raop_at_retrieval <- c()
requester_number_of_comments_at_request <- c()
requester_number_of_comments_at_retrieval <- c()
requester_number_of_comments_in_raop_at_request <- c()
requester_number_of_comments_in_raop_at_retrieval <- c()
requester_number_of_posts_at_request <- c()
requester_number_of_posts_at_retrieval <- c()
requester_number_of_posts_on_raop_at_request <- c()
requester_number_of_posts_on_raop_at_retrieval <- c()
requester_number_of_subreddits_at_request <- c()
requester_received_pizza <- c()
requester_subreddits_at_request <- c()
requester_upvotes_minus_downvotes_at_request <- c()
requester_upvotes_minus_downvotes_at_retrieval <- c()
requester_upvotes_plus_downvotes_at_request <- c()
requester_upvotes_plus_downvotes_at_retrieval <- c()
requester_user_flair <- c()
requester_username <- c()
unix_timestamp_of_request <- c()
unix_timestamp_of_request_utc <- c()

for (i in 1:length(train_df)){
  giver_username_if_known[i]= train_df[[i]]$giver_username_if_known
  number_of_downvotes_of_request_at_retrieval[i] = train_df[[i]]$number_of_downvotes_of_request_at_retrieval
  number_of_upvotes_of_request_at_retrieval[i] = train_df[[i]]$number_of_upvotes_of_request_at_retrieval
  post_was_edited[i] = train_df[[i]]$post_was_edited
  request_id[i] = train_df[[i]]$request_id
  request_number_of_comments_at_retrieval[i] = train_df[[i]]$request_number_of_comments_at_retrieval
  request_text[i] = train_df[[i]]$request_text
  request_text_edit_aware[i] = train_df[[i]]$request_text_edit_aware
  request_title[i] = train_df[[i]]$request_title
  requester_account_age_in_days_at_request[i] = train_df[[i]]$requester_account_age_in_days_at_request
  requester_account_age_in_days_at_retrieval[i] = train_df[[i]]$requester_account_age_in_days_at_retrieval
  requester_days_since_first_post_on_raop_at_request[i] = train_df[[i]]$requester_days_since_first_post_on_raop_at_request
  requester_days_since_first_post_on_raop_at_retrieval[i] = train_df[[i]]$requester_days_since_first_post_on_raop_at_retrieval
  requester_number_of_comments_at_request[i] = train_df[[i]]$requester_number_of_comments_at_request
  requester_number_of_comments_at_retrieval[i] = train_df[[i]]$requester_number_of_comments_at_retrieval
  requester_number_of_comments_in_raop_at_request[i] = train_df[[i]]$requester_number_of_comments_in_raop_at_request
  requester_number_of_comments_in_raop_at_retrieval[i] = train_df[[i]]$requester_number_of_comments_in_raop_at_retrieval
  requester_number_of_posts_at_request[i] = train_df[[i]]$requester_number_of_posts_at_request
  requester_number_of_posts_at_retrieval[i] = train_df[[i]]$requester_number_of_posts_at_retrieval
  requester_number_of_posts_on_raop_at_request[i] = train_df[[i]]$requester_number_of_posts_on_raop_at_request
  requester_number_of_posts_on_raop_at_retrieval[i] = train_df[[i]]$requester_number_of_posts_on_raop_at_retrieval
  requester_number_of_subreddits_at_request[i] = train_df[[i]]$requester_number_of_subreddits_at_request
  requester_received_pizza[i] = train_df[[i]]$requester_received_pizza
  requester_subreddits_at_request[i] = length(train_df[[i]]$requester_subreddits_at_request)
  requester_upvotes_minus_downvotes_at_request[i] = train_df[[i]]$requester_upvotes_minus_downvotes_at_request
  requester_upvotes_minus_downvotes_at_retrieval[i] = train_df[[i]]$requester_upvotes_minus_downvotes_at_retrieval
  requester_upvotes_plus_downvotes_at_request[i] = train_df[[i]]$requester_upvotes_plus_downvotes_at_request
  requester_upvotes_plus_downvotes_at_retrieval[i] = train_df[[i]]$requester_upvotes_plus_downvotes_at_retrieval
  requester_user_flair[i] = length(train_df[[i]]$requester_user_flair)
  requester_username[i] = train_df[[i]]$requester_username
  unix_timestamp_of_request[i] = train_df[[i]]$unix_timestamp_of_request
  unix_timestamp_of_request_utc[i] = train_df[[i]]$unix_timestamp_of_request_utc
}

train_df_frame <- data.frame("giver_username_if_known" = unlist(giver_username_if_known),                             
                            "number_of_downvotes_of_request_at_retrieval" = unlist(number_of_downvotes_of_request_at_retrieval),         
                            "number_of_upvotes_of_request_at_retrieval" = unlist(number_of_upvotes_of_request_at_retrieval),           
                            "post_was_edited" = unlist(post_was_edited),                                     
                            "request_id" = unlist(request_id),                                          
                            "request_number_of_comments_at_retrieval" = unlist(request_number_of_comments_at_retrieval),             
                            "request_text" = unlist(request_text),                                        
                            "request_text_edit_aware" = unlist(request_text_edit_aware),                             
                            "request_title" = unlist(request_title),                                       
                            "requester_account_age_in_days_at_request" = unlist(requester_account_age_in_days_at_request),            
                            "requester_account_age_in_days_at_retrieval" = unlist(requester_account_age_in_days_at_retrieval),          
                            "requester_days_since_first_post_on_raop_at_request" = unlist(requester_days_since_first_post_on_raop_at_request),  
                            "requester_days_since_first_post_on_raop_at_retrieval" = unlist(requester_days_since_first_post_on_raop_at_retrieval),
                            "requester_number_of_comments_at_request" = unlist(requester_number_of_comments_at_request),             
                            "requester_number_of_comments_at_retrieval" = unlist(requester_number_of_comments_at_retrieval),           
                            "requester_number_of_comments_in_raop_at_request" = unlist(requester_number_of_comments_in_raop_at_request),     
                            "requester_number_of_comments_in_raop_at_retrieval" = unlist(requester_number_of_comments_in_raop_at_retrieval),   
                            "requester_number_of_posts_at_request" = unlist(requester_number_of_posts_at_request),                
                            "requester_number_of_posts_at_retrieval" = unlist(requester_number_of_posts_at_retrieval),              
                            "requester_number_of_posts_on_raop_at_request" = unlist(requester_number_of_posts_on_raop_at_request),        
                            "requester_number_of_posts_on_raop_at_retrieval" = unlist(requester_number_of_posts_on_raop_at_retrieval),      
                            "requester_number_of_subreddits_at_request" = unlist(requester_number_of_subreddits_at_request),           
                            "requester_received_pizza" = unlist(requester_received_pizza),                            
                            "requester_subreddits_at_request" = unlist(requester_subreddits_at_request),                     
                            "requester_upvotes_minus_downvotes_at_request" = unlist(requester_upvotes_minus_downvotes_at_request),        
                            "requester_upvotes_minus_downvotes_at_retrieval" = unlist(requester_upvotes_minus_downvotes_at_retrieval),      
                            "requester_upvotes_plus_downvotes_at_request" = unlist(requester_upvotes_plus_downvotes_at_request),         
                            "requester_upvotes_plus_downvotes_at_retrieval" = unlist(requester_upvotes_plus_downvotes_at_retrieval),       
                            "requester_user_flair" = unlist(requester_user_flair),                                
                            "requester_username" = unlist(requester_username),                                  
                            "unix_timestamp_of_request" = unlist(unix_timestamp_of_request),                           
                            "unix_timestamp_of_request_utc" = unlist(unix_timestamp_of_request_utc))


#Visualization: <  requester_received_pizza  VS  number_of_downvotes_of_request_at_retrieval >
downvotes <- data.frame(train_df_frame$requester_received_pizza, 
                        train_df_frame$number_of_downvotes_of_request_at_retrieval)
downvote_table <- table(downvotes)
x <- unique(downvotes$train_df_frame.number_of_downvotes_of_request_at_retrieval)
y <- downvote_table[2,]/downvote_table[1,]
plot(x, y, col = "dodgerblue", xlab = "# of Downvotes",
     ylab = "Proportion of Successful Pizza Requests", main = "# of Downvotes VS Proportion of Successful Pizza Requests", pch = 15)

#Fits a linear model to the data whilst ignoring all values that jet off to infinity.
lm_fit_df <- data.frame(x, y)
lm_fit_df <- lm_fit_df[which(lm_fit_df$y != Inf),]
fit <- lm(lm_fit_df$y ~ lm_fit_df$x)
abline(0.438249, -0.009727, col = "red", lwd = 1.5)
hhat <- dpik(lm_fit_df$x)
kern <- ksmooth(lm_fit_df$x, lm_fit_df$y, bandwidth = hhat)
lines(kern, lwd=2, col= "cadetblue")

###################################################################

upvotes <- data.frame(train_df_frame$requester_received_pizza, 
                      train_df_frame$number_of_upvotes_of_request_at_retrieval)
upvote_table <- table(upvotes)

x <- unique(upvotes$train_df_frame.number_of_upvotes_of_request_at_retrieval)
y <- upvote_table[2,]/upvote_table[1,]
plot(x, y, col = "dodgerblue", xlab = "# of upvotes",
     ylab = "Proportion of Successful Pizza Requests", main = "# upvotes VS Proportion of Successful Pizza Requests", pch = 15)

lm_fit_df <- data.frame(xvals, yvals)
lm_fit_df <- lm_fit_df[which(lm_fit_df$yvals != Inf),]
fit <- lm(lm_fit_df$yvals ~ lm_fit_df$xvals)
abline(0.653607, -0.001204, col = "green", lwd = 1.5)
hhat <- 20
kern <- ksmooth(lm_fit_df$xvals, lm_fit_df$yvals, bandwidth = hhat)
lines(kern, lwd=2, col= "cadetblue")

#######################################################################
# # Totoal Attention
total_attention <- train_df_frame$requester_upvotes_plus_downvotes_at_retrieval + train_df_frame$request_number_of_comments_at_retrieval
total_attention_df <- data.frame(total_attention, train_df_frame$requester_received_pizza)
plot(total_attention_df$total_attention, total_attention_df$train_df_frame.requester_received_pizza)

#Titles text mining.
# Get corpus from request title
successful_requests <- train_df_frame[which(train_df_frame$requester_received_pizza == TRUE),]
successful_titles <- successful_requests$request_title
n_successful_titles <- list()

for (i in 1:length(successful_titles)){
  n_successful_titles[[i]] <- unlist(strsplit(as.vector(successful_titles[i]), " "))
}

total_successful_title_words = unlist(n_successful_titles)
total_successful_title_words = gsub('[^[:alpha:]]', ' ', total_successful_title_words)
total_successful_title_words = tolower(total_successful_title_words)
total_successful_title_words_table_df <- data.frame(table(total_successful_title_words))
total_successful_title_words_table_df <- total_successful_title_words_table_df[order(total_successful_title_words_table_df$Freq),]
# OK ABOVE

# Too much stop words, "noise" words in the dataset
words_page <- readLines("/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/commonwords.txt")
common_words = words_page[1:500]

for (i in 1:length(common_words)){
  common_words[i] <- strsplit(common_words[i], "<")[[1]][1]
}
common_words <- c(common_words, "is", "a", "on", "to", "in", "of","the", "and", "Pizza", "pizza", "Pizza:","pizza.", "[Request]", "[REQUEST]", "[request]", "-", "edit:")

total_successful_title_words_table_df <- total_successful_title_words_table_df[-which(total_successful_title_words_table_df$total_successful_title_words %in% common_words),]
total_successful_title_words_table_df <- total_successful_title_words_table_df[order(total_successful_title_words_table_df$Freq),]
title_wordcloud <- Corpus(VectorSource(total_successful_title_words_table_df$total_successful_title_words))

##############################################################


# Get corpus from request body
successful_requests <- train_df_frame[which(train_df_frame$requester_received_pizza == TRUE),]
successful_body <- successful_requests$request_text_edit_aware
n_successful_body <- list()

for (i in 1:length(successful_body)){
  n_successful_body[[i]] <- unlist(strsplit(as.vector(successful_body[i]), " "))
}

total_successful_body_words <- unlist(n_successful_body)
total_successful_body_words <- gsub('[^[:alpha:]]', ' ', total_successful_body_words)
total_successful_body_words <- tolower(total_successful_body_words)
total_successful_body_words_table_df <- data.frame(table(total_successful_body_words))
total_successful_body_words_table_df <- total_successful_body_words_table_df[order(total_successful_body_words_table_df$Freq),]

# Too much stop words, "noise" words in the dataset
words_page <- readLines("/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/commonwords.txt")
common_words <- words_page[1:500]

for (i in 1:length(common_words)){
  common_words[i] = strsplit(common_words[i], "<")[[1]][1]
}
common_words = c(common_words, "i","is", "a", "on", "to", "in", "of","the", "and", "Pizza", "pizza", "Pizza:","pizza.", "[Request]", "[REQUEST]", "[request]", "-", "edit:")
total_successful_body_words_table_df <- total_successful_body_words_table_df[-which(total_successful_body_words_table_df$total_successful_body_words %in% common_words),]
total_successful_body_words_table_df <- total_successful_body_words_table_df[order(total_successful_body_words_table_df$Freq),]
body_wordcloud <- Corpus(VectorSource(total_successful_body_words_table_df$total_successful_body_words))

save(body_wordcloud, file = "/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/request_body_words.Rdata")

# CLEANING DATA
#Check length
total_successful_body_words_len <- unlist(lapply(total_successful_body_words, length))
# Above show all item is "word"

successful_body_words_noun <- list()

for (i in 1:length(total_successful_body_words)) {
  successful_body_words_noun[[i]] = total_successful_body_words[[i]]
}

# paste back to corpus
successful_body_sentence <- lapply(successful_body_words_noun, paste, collapse = ' ')
successful_body_sentence <- Corpus(VectorSource(successful_body_sentence))
save(successful_body_sentence, file = 'corpusRequestBody.RData')

dtm = DocumentTermMatrix(successful_body_sentence)
save(dtm, file = 'dtm.RData')

# tf-idf
tf <- tapply(dtm$v / row_sums(dtm)[dtm$i], dtm$j, mean)
idf <- log2(nDocs(dtm) / col_sums(dtm > 0))
tf_idf <- tf * idf

#keep  tf-idf >= 25%
dtm_trim <- dtm[, tf_idf >= quantile(tf_idf, .25)]
trim_ind <- which(row_sums(dtm_trim) > 0)
dtm_trim <- dtm_trim[trim_ind, ]
save(dtm_trim, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/dtm_trim.Rdata')

train_trim <- successful_requests[trim_ind, ]
save(train_trim, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_cleaned.Rdata')

############################################################
#lda
#determine the optimal number of topics via cv
set.seed(2014)
folder <- createFolds(train_trim$requester_received_pizza)
lda_eval <- data.frame(fold = integer(), topic = integer(), perplex = numeric())

for (i in 1:length(folder)) {
  for (k in 2:10) {
    cat(i, k, '\n')
    dtm_train <- dtm_trim[-folder[[i]], ]
    dtm_test <- dtm_trim[folder[[i]], ]
    
    lda_train <- LDA(dtm_train, k, control = list(seed = 2014))
    lda_test <- LDA(dtm_test, model = lda_train)
    
    lda_eval <- rbind(lda_eval, c(i, k, perplexity(lda_test)))
  }
}
names(lda_eval) <- c('fold', 'topic', 'perplex')
pp <- ggplot(lda_eval, aes(x = topic, y = perplex, colour = as.factor(fold), group = as.factor(fold))) + geom_line()
ggsave(pp, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/perplex.jpg')
sink('/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/lda.txt')
ddply(lda_eval, .(fold), summarize, min_p = min(perplex), min_t = topic[which.min(perplex)])
sink()
# length(folder) <- 11, split into 11 folders is better.
lda_m <- LDA(dtm_trim, 11, control = list(seed = 2014))
lda_topics <- posterior(lda_m)$topics
lda_terms <- posterior(lda_m)$terms

#word cloud
words <- names(lda_terms[1, ])
for (i in 1:11) {
  png(paste0('lda', i, '.png'), width = 400, height = 400)
  wordcloud(words, lda_terms[i, ], max.words = 50, random.order = F, col = brewer.pal(8, "Dark2"))
  dev.off()
}

#try sk-means
set.seed(2014)
sk_m <- skmeans(dtm_trim, 5)
sk_pt <- sk_m$prototypes

#word cloud
words <- names(sk_pt[1, ])
wordcloud(words, sk_pt[1, ], max.words = 500, random.order = F, col = brewer.pal(8, "Dark2"))
wordcloud(words, sk_pt[2, ], max.words = 500, random.order = F, col = brewer.pal(8, "Dark2"))
wordcloud(words, sk_pt[3, ], max.words = 500, random.order = F, col = brewer.pal(8, "Dark2"))
wordcloud(words, sk_pt[4, ], max.words = 500, random.order = F, col = brewer.pal(8, "Dark2"))
wordcloud(words, sk_pt[5, ], max.words = 500, random.order = F, col = brewer.pal(8, "Dark2"))
