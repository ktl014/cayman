import json
import os
import glob
import datetime
from lxml import html
import urllib2
import cookielib
import pandas as pd

def gather_data():
    import pandas as pd
    df = pd.DataFrame()
    ROOT = '/data6/lekevin/cayman'
    rawdata_path = [os.path.join(ROOT,'rawdata','{}_SPC_Images_3-COLOR'.format(i)) for i in ['EC1', 'EC2', 'EC3']]
    for i in range(4):
        parse_param = {'sep':' ', 'names':['image', 'label', 'cool_example']}
        if i == 2:
            filename = os.path.join(ROOT, 'rawdata/yolksac.txt')
        elif i == 3:
            filename = '/data6/lekevin/cayman/rawdata/d3_predictions1.txt'
            parse_param = {'sep':',', 'names':['image', 'day', 'label']}
        else:
            filename = os.path.join(ROOT,'rawdata/classes_EC{}_1516combined.txt'.format(i+1))
        temp = pd.read_csv(filename, sep=parse_param['sep'], names=parse_param['names'], header=None)
        temp['day'] = ['EC{}'.format(i+1)] * temp.shape[0] if i != 3 else temp['day'].map ({'Thu Feb 16': 'EC3', 'Wed Feb 15': 'EC2', 'Fri Feb 17': 'EC3'})
        df = df.append(temp, ignore_index=True)

    # Map class labels to numeric labels
    with open (os.path.join (ROOT, 'rawdata/labels.txt')) as f:
        labels = {int (k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}
    df['class'] = df['label'].map(labels)
    return df

class SPCServer(object):
    def __init__(self):
        # Dates initialized for uploading purposes
        date = datetime.datetime.now().strftime('%s')
        self.date_submitted = str(int(date)*1000)
        self.date_started = str((int(date)*1000)-500)

        # Data for uploading images
        self.submit_dict = {"label": "",
                            "tag": "",
                            "images": [],
                            "name": "",
                            "machine_name":"",
                            "started":self.date_started,
                            "submitted": self.date_submitted,
                            "is_machine": False,
                            "query_path": "",
                            "notes": ""
        }

    def prep_for_upload(self, login_url, account_info):
        '''
        Authorizes access to the server

        Usage:
            ==> account_info = {'username':'kevin', 'password': 'plankt0n'}
            ==> login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
            ==> prep_for_upload(login_url=login_url)

        :return:
        '''
        assert isinstance(account_info, dict)
        assert isinstance(login_url, str)

        cj = cookielib.CookieJar()
        self.opener = urllib2.build_opener(
            urllib2.HTTPCookieProcessor(cj),
            urllib2.HTTPHandler(debuglevel=1)
        )

        if login_url == None:
            login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
        login_form = self.opener.open (login_url).read ()

        self.csrf_token = html.fromstring(login_form).xpath(
            '//input[@name="csrfmiddlewaretoken"]/@value')[0]

        params = json.dumps(account_info)
        req = urllib2.Request ('http://planktivore.ucsd.edu/caymans_data/rois/login_user',
                               params, headers={'X-CSRFToken': str (self.csrf_token),
                                                'X-Requested-With': 'XMLHttpRequest',
                                                'User-agent': 'Mozilla/5.0',
                                                'Content-type': 'application/json'
                                                }
                               )
        self.resp = self.opener.open(req)
        print('Successfully logged in {}'.format(self.resp.read()))

    def upload(self):
        '''
        Uploads submit dictionary to initialized url from prep_for_upload()

        Usage:
            ==> spc.submit_dict['name'] = 'brian'
            ==> spc.submit_dict['label'] = label
            ==> spc.submit_dict['image'] = images
            ==> spc.submit_dict['is_machine'] = True
            ==> spc.submit_dict['machine_name'] = mach
            ==> spc.upload()
        :return:
        '''
        assert isinstance(self.submit_dict['images'], list)
        assert isinstance(self.submit_dict['label'], str)
        assert self.submit_dict['name'] != ""

        self.submit_json = json.dumps(self.submit_dict)
        self.req1 = urllib2.Request('http://planktivore.ucsd.edu/caymans_data/rois/label_images',
                       self.submit_json, headers={'X-CSRFToken': str(self.csrf_token),
                                             'X-Requested-With': 'XMLHttpRequest',
                                             'User-agent': 'Mozilla/5.0',
                                             'Content-type': 'application/json'
                                             }
                       )
        self.resp1 = self.opener.open(self.req1)

    def prep_for_retrieval(self):
        # Find meta file

        pass

    def retrieve(self):
        pass

    def download(self):
        pass

if __name__ == '__main__':
    account_info = {'username': 'kevin', 'password': 'plankt0n'}
    login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
    spc = SPCServer()
    spc.prep_for_upload(account_info=account_info, login_url=login_url)

    df = pd.read_csv('/data6/lekevin/cayman/records/caffenet/version_1/test_predictions.csv')
    df['image'] = df['image'].apply(lambda x: os.path.basename(x))

    # for each label, grab the list of images and submit them to the server
    labels = ['copepod_nauplius', 'hard_to_id', 'trichodesmium']
    # labels = sorted(df['class'].unique())
    grouped = df.groupby(df['class'])
    redo_f = open('redo.txt', 'w')
    maximum_imgs = 15000
    for label in labels:
        grouped_images = grouped.get_group(label)['image'].tolist()
        if len(grouped_images) > maximum_imgs:
            for i in range(0, len(grouped_images), 15000):
                batch = grouped_images[i:i+15000]
                spc.submit_dict['name'] = 'model_d5_test'
                spc.submit_dict['label'] = label
                spc.submit_dict['images'] = batch
                spc.submit_dict['is_machine'] = True
                spc.submit_dict['machine_name'] = "model_d5_test"
                spc.upload()
        else:
            spc.submit_dict['name'] = 'model_d5_test'
            spc.submit_dict['label'] = label
            spc.submit_dict['images'] = grouped_images
            spc.submit_dict['is_machine'] = True
            spc.submit_dict['machine_name'] = "model_d5_test"
            try:
                spc.upload()
            except:
                redo_f.write('{}\n'.format(label))
        print('Uploaded {} {} images'.format(label, len(grouped_images)))
    redo_f.close()


