import requests
from datetime import date
from itertools import dropwhile

class JobProvider:
    hiring_string = 'ho is hiring?'
    hiring_user = 'whoishiring'

    def __init__(self, jobs_api_url, historical_limit = 2):
        self.jobs_api_url = jobs_api_url
        self.historical_limit = historical_limit

    # This class is db agnostic, pass in lowest_saved_id if you have anything saved or you'll get the same result each time 
    def get_next_historical(self, lowest_saved_id = None):
        whois_post_ids = self.get_json(f'/user/{self.hiring_user}/submitted.json')
        post_ids_iter = iter(whois_post_ids) if lowest_saved_id is None else dropwhile(lambda x: x >= lowest_saved_id)
        month_posts = None

        # TODO: upgrade this to walrus operator once you can bump to python 3.8
        next_id = next(post_ids_iter, None)
        while month_posts is None and next_id is not None:
            month_posts = self.get_hiring_posts(next_id)
            next_id = next(post_ids_iter, None)

        return month_posts

    def get_hiring_posts(self, parent_id):
        posts = []
        hiring_post = self.get_json(f'/item/{parent_id}.json')
        if not self.is_valid_post(date.fromtimestamp(hiring_post['time']), hiring_post['title'], self.historical_limit, self.hiring_string):
            return None
        else:
            parent_date = hiring_post['time']
            for child_id in hiring_post['kids'][:5]:
                child_post = self.get_json(f'/item/{child_id}.json')
                posts.append(self.format_post(child_post, parent_date))

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
    def format_post(post, parent_date):
        return {
            'by': post.get('by'),
            'text': post.get('text'),
            'id': post.get('id'),
            'date': parent_date,
            'preferred': None
        }