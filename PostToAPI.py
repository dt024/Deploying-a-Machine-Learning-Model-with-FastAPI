"""Script to post to FastAPI instance for model inference"""

import requests
import json

#url = "enter heroku web app url here"
url = "https://udacity-sec4.herokuapp.com/inference"


# explicit the sample to perform inference on
sample =  {  'age':30,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"HS-grad",
            'education_num':1,
            'marital_status':"Separated",
            'occupation':"Handlers-cleaners",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Male",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':35,
            'native_country':"United-States"
            }

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())

