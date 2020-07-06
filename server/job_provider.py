import requests
import time
from datetime import date
from itertools import chain, dropwhile, takewhile

from server.config import config

class JobProvider:
    hiring_string = config['whos_hiring_search_string']
    hiring_user = config['whos_hiring_user']
    min_post_length = 50

    def __init__(self, jobs_api_url, throttle_group_size = None, throttle_duration = 1):
        self.jobs_api_url = jobs_api_url
        self.throttle_group_size = throttle_group_size
        self.throttle_duration = throttle_duration

    # This class is db agnostic, pass in lowest_saved_id if you have anything saved or you'll get the same result each time 
    def get_next_post(self, historical_limit = 1, lowest_saved_id = None, highest_saved_id = None):
        whois_post_ids = self.get_whois_post_ids()
        post_ids_iter = iter(whois_post_ids) if lowest_saved_id is None or highest_saved_id is None else (
            # Overoptimization perhaps? assumption is that there's going to be more than 2x posts on the right tail than we'll iterate through here
            chain(takewhile(lambda x: x > highest_saved_id, whois_post_ids), dropwhile(lambda x: x >= lowest_saved_id, whois_post_ids))
        )
        month_posts = None

        # TODO: upgrade this to walrus operator once you can bump to python 3.8
        next_id = next(post_ids_iter, None)
        while month_posts is None and next_id is not None:
            month_posts = self.lookup_hiring_posts(next_id, historical_limit)
            next_id = next(post_ids_iter, None)

        return month_posts

    # These are returned in date desc order by default
    def get_whois_post_ids(self):
        return self.get_json(f'/user/{self.hiring_user}/submitted.json')

    def lookup_hiring_posts(self, parent_id, historical_limit):
        hiring_post = self.get_json(f'/item/{parent_id}.json')
        post_date = date.fromtimestamp(hiring_post['time'])
        if 'deleted' in hiring_post or not self.is_valid_post(post_date, hiring_post['title'], historical_limit):
            return None
        else:
            print(f'Fetching posts for {post_date}.')
            return self.get_child_posts(parent_id, hiring_post['time'], hiring_post['kids'], self.throttle_group_size, self.throttle_duration)

    # largely differs from lookup_hiring_posts by assuming that the parent id is a valid hiring post
    def get_hiring_posts(self, parent_id):
        hiring_post = self.get_json(f'/item/{parent_id}.json')
        return self.get_child_posts(parent_id, hiring_post['time'], hiring_post['kids'])

    def get_child_posts(self, parent_id, parent_time, child_ids, throttle_group_size = None, throttle_duration = None):
        posts = []
        for i, child_id in enumerate(child_ids):
            if throttle_group_size is not None:
                if i % throttle_group_size == 0:
                    time.sleep(throttle_duration)

            child_post = self.get_json(f'/item/{child_id}.json')
            # TODO: injestion validation if there's too much junk
            if child_post is not None and child_post.get('parent') == parent_id and len(child_post.get('text', '')) > self.min_post_length:
                posts.append(self.parse_post(child_post, parent_time))

        return posts

    def get_json(self, path):
        url = f'{self.jobs_api_url}{path}'
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f'Call to "{url}" failed with {response.text}')

        return response.json()

    def get_months(self, num_months):
        whois_ids = self.get_whois_post_ids()
        month_posts = []
        for id in whois_ids:
            post = self.get_json(f'/item/{id}.json')
            post_date = date.fromtimestamp(post['time'])
            if self.is_valid_title(post.get('title', '')):
                if self.is_valid_date(post_date, num_months, search_within_limit= True):
                    month_posts.append(post)
                else:
                    # posts are by desc date, once we reach a post too old then we're done searching
                    return month_posts

        return month_posts # just incase limit was too big to reach

    @staticmethod
    def is_valid_post(post_date, post_title, month_limit, search_string, search_within_limit = False):
        return JobProvider.is_valid_title(post_title) and JobProvider.search_by_limit(post_date, month_limit, search_within_limit)

    @staticmethod
    def is_valid_title(title):
        return JobProvider.hiring_string in title

    @staticmethod
    def is_valid_date(post_date, limit, search_within_limit):
        today = date.today()

        # a month of 0 doesn't exactly work. Need to offset by one when determining if limit pushes us to prior year once we hit month 0 (which would be december)
        year_limit = today.year + min((today.month - limit - 1) // 12, 0)
        month_limit = ((today.month - limit -1) % 12) + 1

        if search_within_limit:
            return (post_date.year == year_limit and post_date.month > month_limit) or post_date.year > year_limit
        else:
            return (post_date.year == year_limit and post_date.month <= month_limit) or post_date.year < year_limit


    @staticmethod
    def parse_post(post, parent_date):
        return {
            'by': post.get('by'),
            'text': post.get('text'),
            'id': post.get('id'),
            'parent': post.get('parent'),
            'date': parent_date,
            'preferred': None
        }

    @staticmethod
    def format_post(post):
        return {
            'by': post.get('by'),
            'text': post.get('text'),
            'id': post.get('id'),
            'parent': post.get('parent'),
            'date': post.get('date'),
            'preferred': post.get('preferred'),
            'title': post.get('title')
        }