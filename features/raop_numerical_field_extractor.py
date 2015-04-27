from preprocessing.preprocesser import reddit_roap_convert_unix_time
from features.raop_field_extractor import RAOPFieldExtractor

class RAOPNumericalFieldExtractor(RAOPFieldExtractor):

    def __init__(self):
        numericalFields = [
            'requester_account_age_in_days_at_request',
            'requester_days_since_first_post_on_raop_at_request',
            'requester_number_of_comments_at_request',
            'requester_number_of_comments_in_raop_at_request',
            'requester_number_of_posts_at_request',
            'requester_number_of_posts_on_raop_at_request',
            'requester_number_of_subreddits_at_request',
            'requester_upvotes_minus_downvotes_at_request',
            'requester_upvotes_plus_downvotes_at_request'
        ]

        RAOPFieldExtractor.__init__(self, numericalFields)

    def transform(self, X):
        return RAOPFieldExtractor.transform(self, X)
