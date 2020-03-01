import requests
import time
from datetime import date
from itertools import chain, dropwhile, takewhile

class JobProvider:
    hiring_string = 'ho is hiring?'
    hiring_user = 'whoishiring'

    def __init__(self, jobs_api_url, throttle_group_size = None, throttle_duration = 1):
        self.jobs_api_url = jobs_api_url
        self.throttle_group_size = throttle_group_size
        self.throttle_duration = throttle_duration

    # This class is db agnostic, pass in lowest_saved_id if you have anything saved or you'll get the same result each time 
    def get_next_post(self, historical_limit = 1, lowest_saved_id = None, highest_saved_id = None):
        whois_post_ids = self.get_json(f'/user/{self.hiring_user}/submitted.json')
        post_ids_iter = iter(whois_post_ids) if lowest_saved_id is None or highest_saved_id is None else (
            # Overoptimization perhaps? assumption is that there's going to be more than 2x posts on the right tail than we'll iterate through here
            chain(takewhile(lambda x: x > highest_saved_id, whois_post_ids), dropwhile(lambda x: x >= lowest_saved_id, whois_post_ids))
        )
        month_posts = None

        # TODO: upgrade this to walrus operator once you can bump to python 3.8
        next_id = next(post_ids_iter, None)
        while month_posts is None and next_id is not None:
            month_posts = self.get_hiring_posts(next_id, historical_limit)
            next_id = next(post_ids_iter, None)

        return month_posts

    def get_hiring_posts(self, parent_id, historical_limit):
        posts = []
        hiring_post = self.get_json(f'/item/{parent_id}.json')
        if not self.is_valid_post(date.fromtimestamp(hiring_post['time']), hiring_post['title'], historical_limit, self.hiring_string):
            return None
        else:
            parent_date = hiring_post['time']
            for i, child_id in enumerate(hiring_post['kids']):
                # TODO: update this if you intend for concurrent users
                if self.throttle_group_size is not None:
                    if i % self.throttle_group_size == 0:
                        time.sleep(self.throttle_duration)

                child_post = self.get_json(f'/item/{child_id}.json')
                # TODO: injestion validation if there's too much junk
                if child_post is not None and child_post.get('parent') == parent_id:
                    posts.append(self.parse_post(child_post, parent_date))

            return posts

    def get_json(self, path):
        url = f'{self.jobs_api_url}{path}'
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f'Call to "{url}" failed with {response.text}')

        return response.json()

    @staticmethod
    def is_valid_post(post_date, post_title, month_limit, search_string):
        today = date.today()

        # a month of 0 doesn't exactly work. Need to offset by one when determining if limit pushes us to prior year once we hit month 0 (which would be december)
        year_limit = today.year + min((today.month - month_limit - 1) // 12, 0)
        month_limit = ((today.month - month_limit -1) % 12) + 1
        return search_string in post_title and (
            (post_date.year == year_limit and post_date.month <= month_limit) or post_date.year < year_limit
        )

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
            'preferred': post.get('preferred')
        }