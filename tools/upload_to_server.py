# -*- coding: utf-8 -*-
"""
created on Fri May 11 2018
Make a json document with the appropriate class labels and pass it to server

"""

import json
import os
import glob
import datetime
from lxml import html
import urllib2
import cookielib

# create a date
date = datetime.datetime.now()
date = date.strftime('%s')
date1 = str(int(date)*1000)
date2 = str((int(date)*1000)-500)

# log into the server
cj = cookielib.CookieJar()
opener = urllib2.build_opener(
    urllib2.HTTPCookieProcessor(cj),
    urllib2.HTTPHandler(debuglevel=1)
)

login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
login_form = opener.open(login_url).read()

csrf_token = html.fromstring(login_form).xpath(
    '//input[@name="csrfmiddlewaretoken"]/@value'
)[0]

values = {
    'username': 'kevin',
    'password': 'plankt0n',
}

params = json.dumps(values)

req = urllib2.Request('http://planktivore.ucsd.edu/caymans_data/rois/login_user',
                      params, headers={'X-CSRFToken': str(csrf_token),
                                       'X-Requested-With': 'XMLHttpRequest',
                                       'User-agent': 'Mozilla/5.0',
                                       'Content-type': 'application/json'
                                       }
                      )

resp = opener.open(req)
print('login ' + resp.read())

# get classifier output and put on server
def parse_csv_file(csv_file, selected_columns, label):
    import pandas as pd
    import os
    label_dict = {0:'non-fish_egg', 1:'fish_egg'}

    assert isinstance(selected_columns, list)
    assert len(selected_columns) == 2
    assert selected_columns[1] == 'predictions' or selected_columns[1] == 'label'
    assert label in label_dict.values()

    image_str = selected_columns[0]
    label_str = selected_columns[1]

    df = pd.read_csv(csv_file)
    parsed_df = df[selected_columns].copy()
    parsed_df['class'] = df[label_str].map({0:'non-fish_egg', 1:'fish_egg'})
    parsed_df = parsed_df[parsed_df['class'] == label]

    images = parsed_df['image'].apply(lambda x: os.path.basename(x)).tolist()
    labels = parsed_df['class'].tolist()

    return images, labels

phase = 'train'
label = 'non-fish_egg'
version = 4
if phase == 'train' or phase == 'val':
    root = '/data6/lekevin/cayman'
    img_dir = '/data6/lekevin/cayman/rawdata'
    csv_filename = os.path.join (root, 'data', str (version), 'data_{}.csv'.format(phase))
    selected_columns = ['image', 'label']
else:
    csv_filename = '/data6/lekevin/cayman/records/model_d3/version_1/test_predictions.csv'
    selected_columns = ['image', 'predictions']

mach = 'model_d{}_{}'.format(version, phase)
images, labels = parse_csv_file(csv_filename, selected_columns, label=label)

# build the dictionary
submit = {"label": label, "tag": "", "images": images, "name": 'kevin',
          "machine_name": mach, "started": date2, "submitted": date1,
          "is_machine": True, "query_path": "", "notes": ""}

# dump to json
submit_json = json.dumps(submit)
print(submit_json)

req1 = urllib2.Request('http://planktivore.ucsd.edu/caymans_data/rois/label_images',
                       submit_json, headers={'X-CSRFToken': str(csrf_token),
                                             'X-Requested-With': 'XMLHttpRequest',
                                             'User-agent': 'Mozilla/5.0',
                                             'Content-type': 'application/json'
                                             }
                       )

resp1 = opener.open(req1)

print("Submitted for {}".format(csv_filename))
