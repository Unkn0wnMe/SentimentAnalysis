from twitter.account import Account
from twitter.scraper import Scraper
from twitter.search import Search

email, username, password = 'email', 'username', 'pwd'
#account = Account(email, username, password)

search = Search(email=email, username=username, password=password, save=True, debug=True)

res = search.run(
    retries=5000,
    limit=50000,
    queries=[
        {
            'category': 'Latest',
            'query': 'Donald Trump'
        }
    ]
)

with open('output.txt', 'w') as file:
    file.write(str(res))