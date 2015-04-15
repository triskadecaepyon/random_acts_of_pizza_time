"""
Python functions written for Random Acts of Pizza that
do data transformations and cleanups in a programmatic way.

Strategy is a facade-like function that gives standard processing an easy call.
Optionally, one may use the supplied functions for custom pre-processing.

"""
import pandas as pd
import time


def reddit_pre_process_strategy(training_data_frame):
    return_frame = reddit_raop_train_preprocessor(training_data_frame)
    # Process textual data with NLP
    # Binarize some of the continuous data
    return_frame = reddit_roap_convert_unix_time(return_frame)
    return_frame = reddit_roap_convert_hours_to_categories(return_frame)
    # return_frame = reddit_roap_convert_region(return_frame)
    return_frame = return_frame.drop('unix_timestamp_of_request', 1)
    return_frame = return_frame.drop('unix_timestamp_of_request_utc', 1)
    return return_frame


def reddit_raop_train_preprocessor(training_data_frame):
    """
    Used to remove the non-test set features for testing ease.  Note that the target is still included
    for use in classification and regression usage.

    Returns a dataframe back to the user with the dropped features.
    """
    # other method: reduced_training_set = 
    # training_data_frame.drop(training_data_frame.columns[[1,2,3,5,6,10,12,14,16,18,22,23,25,27,29]], 1)
    drop_set = ['number_of_downvotes_of_request_at_retrieval', 'number_of_upvotes_of_request_at_retrieval',
                'post_was_edited', 'request_number_of_comments_at_retrieval', 'request_text',
                'requester_account_age_in_days_at_retrieval',
                'requester_days_since_first_post_on_raop_at_retrieval', 'requester_number_of_comments_at_retrieval', 
                'requester_number_of_comments_in_raop_at_retrieval', 'requester_number_of_posts_at_retrieval', 
                'requester_number_of_posts_on_raop_at_retrieval', 'requester_upvotes_minus_downvotes_at_retrieval', 
                'requester_upvotes_plus_downvotes_at_retrieval', 'requester_user_flair']
    
    for drop_v in drop_set:
        training_data_frame = training_data_frame.drop(drop_v, 1)

    return pd.DataFrame(training_data_frame)


def reddit_roap_convert_unix_time(training_data_frame):
    """
    Used to convert the unix time stamps into hours, and replace the dataframe with hours instead.
    """
    converted_frame = []
    for val in xrange(0, len(training_data_frame.unix_timestamp_of_request)):
        converted_frame.append(reddit_unix_time_convert_hour(training_data_frame.unix_timestamp_of_request[val]))
    training_data_frame.unix_timestamp_of_request = pd.DataFrame(converted_frame)
    
    converted_frame = []
    for val in xrange(0, len(training_data_frame.unix_timestamp_of_request_utc)):
        converted_frame.append(reddit_unix_time_convert_hour(training_data_frame.unix_timestamp_of_request_utc[val]))
    training_data_frame.unix_timestamp_of_request_utc = pd.DataFrame(converted_frame)

    return training_data_frame


def reddit_roap_convert_hours_to_categories(training_data_frame):
    """
    Converts the hours into categorical values of time of day.
    Note that the time_of_day is a categorical series in the dataframe
    """
    converted_frame = []
    for val in xrange(0, len(training_data_frame.unix_timestamp_of_request)):
        # converted_frame.append(reddit_unix_time_convert_hour(training_data_frame.unix_timestamp_of_request[val]))
        hour = training_data_frame.unix_timestamp_of_request[val]
        if hour < 9:
            time_of_day = 'morning'
        elif hour < 15:
            time_of_day = 'midday'
        elif hour < 18:
            time_of_day = 'afternoon'
        else:
            time_of_day = 'night'
        converted_frame.append(time_of_day)
        
    training_data_frame['time_of_day'] = pd.Series(converted_frame, dtype="category")
    return training_data_frame


def reddit_unix_time_convert_hour(time_value):
    time_val = time.gmtime(float(time_value))
    return time_val.tm_hour


def reddit_roap_convert_region(training_data_frame):
    """
    A method to convert the UTC offset of the region.
    Assumes an error offset of -6.
    """
    converted_frame = []

    for val in xrange(0, len(training_data_frame.unix_timestamp_of_request)):
        # Warning: shift of 6 is applied to attempt to correct the UTC time zone
        converted_frame.append(training_data_frame.unix_timestamp_of_request_utc[val] -
                               training_data_frame.unix_timestamp_of_request[val]-6)
    training_data_frame['utcoffset'] = pd.DataFrame(converted_frame)
    #print training_data_frame.utcoffset.unique()
    return training_data_frame
