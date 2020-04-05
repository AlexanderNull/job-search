def get_jobs_range(jobs_table):
    oldest_saved_post = jobs_table.find_one(sort=[('parent', 1)])
    newest_saved_post = jobs_table.find_one(sort=[('parent', -1)])
    
    oldest_id, newest_id = (None, None) if oldest_saved_post is None or newest_saved_post is None else (
        oldest_saved_post['parent'], newest_saved_post['parent']
    )

    return (oldest_id, newest_id)