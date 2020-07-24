import multiprocessing
import os
import os.path as osp
import uuid

import numpy as np
import random

from math import sqrt
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.video import write_segment_to_video, upload_to_gcs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

class SyntheticComparisonCollector(object):
    def __init__(self, run_name, human_label):
        self._comparisons = []
        self._reward_list = []
        self._X = []
        self._y = []
        self._y_predict = []
        self._human_label = human_label
        #self._predictor = LinearRegression()
        #self._predictor = SVR(kernel='linear')
        self._predictor = SVR(kernel='rbf')
        #self._predictor = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
        #self._predictor = MLPRegressor(random_state=1, max_iter=500)
        self._run_name = run_name

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] >= 0.0 and comp['label'] <= 1.0 ]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    def predict_pref(self, left_reward, right_reward):
        pref = None
        val = self._predictor.predict( [ [left_reward, right_reward] ])
        pref = max(0,val[0])
        pref = min(1,pref)
        return pref
    
    def synthetic_pref(self, left_reward, right_reward):
        min_reward = np.percentile(self._reward_list,10)
        max_reward = np.percentile(self._reward_list,90)
        normalise_left = ( ( left_reward - min_reward ) / ( max_reward - min_reward ) )
        normalise_right = ( ( right_reward - min_reward ) / ( max_reward - min_reward ) )

        if normalise_left > 1.0: 
            normalise_left = 1.0
        if normalise_right > 1.0:
            normalise_right = 1.0
        if left_reward > right_reward :
            final_label = 0.5 + normalise_left * 0.5
        elif left_reward < right_reward:
            final_label = 0.5 - normalise_right * 0.5
        else:
            final_label = 0.5
        
        return final_label

    def output_file(self,data_list,log_type=None,append=False):
        filename = "./log/{}-{}.txt".format(self._run_name,log_type)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        mode = "w+"
        if append:
            mode = "a+"
        f = open(filename,mode)
        f.write("{}\n".format(data_list))
        f.close()

    #@staticmethod
    def _add_synthetic_label(self,comparison, syth=True):
        left_seg = comparison['left']
        right_seg = comparison['right']

        left_reward = np.sum(left_seg["original_rewards"])
        right_reward = np.sum(right_seg["original_rewards"])
        self._reward_list.append(left_reward)
        self._reward_list.append(right_reward)
        self.output_file(self._reward_list,'reward')
        print("Reward List: {}".format(self._reward_list))
        if len(self._y) >= self._human_label:
            if self._human_label == len(self._y):
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._X, self._y, test_size=0.3, random_state=1)
                self._predictor.fit(self.X_train, self.y_train)
            method = 'Agent'
            comparison['label'] = self.predict_pref(left_reward, right_reward)
            self._y_predict.append(comparison['label'])
            ground_true = self.synthetic_pref(left_reward, right_reward)
            mse = 0
            mse = (comparison['label'] - ground_true) ** 2
            self.output_file([left_reward, right_reward,comparison['label'],ground_true,"A", mse],'pref',True)
        else:
            if syth:
                method = 'Synthetic'
                comparison['label'] =  self.synthetic_pref(left_reward, right_reward)
            else:
                method = 'Human'
                while True:
                    try:
                        human_input = input("{} - L: {}, R: {} - Please rate your preference (R: 0.0- L: 1.0): ".format(len(self._X), left_reward, right_reward))
                        human_float = float(human_input)
                        if human_float < 0 or human_float > 1:
                            print("Input Error! Please rate between 0.0-1.0")
                        else:
                            comparison['label'] = human_float
                            break
                    except ValueError:
                        print("Input Format Error! Please rate between 0.0-1.0")
            self._X.append( [left_reward,right_reward] )
            self._y.append(comparison['label'])
            self.output_file([left_reward, right_reward,comparison['label'],"H"],'pref',True)
        print("{} - L: {}, R: {}, {} Pref: {} ".format(len(self._X), left_reward, right_reward, method, comparison['label']) )


def _write_and_upload_video(env_id, gcs_path, local_path, segment):
    env = make_with_torque_removed(env_id)
    write_segment_to_video(segment, fname=local_path, env=env)
    upload_to_gcs(local_path, gcs_path)

class HumanComparisonCollector():
    def __init__(self, env_id, experiment_name):
        from human_feedback_api import Comparison

        self._comparisons = []
        self.env_id = env_id
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(4)

        if Comparison.objects.filter(experiment_name=experiment_name).count() > 0:
            raise EnvironmentError("Existing experiment named %s! Pick a new experiment name." % experiment_name)

    def convert_segment_to_media_url(self, comparison_uuid, side, segment):
        tmp_media_dir = '/tmp/rl_teacher_media'
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = osp.join(tmp_media_dir, media_id)
        gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        gcs_path = osp.join(gcs_bucket, media_id)
        self._upload_workers.apply_async(_write_and_upload_video, (self.env_id, gcs_path, local_path, segment))

        media_url = "https://storage.googleapis.com/%s/%s" % (gcs_bucket.lstrip("gs://"), media_id)
        return media_url

    def _create_comparison_in_webapp(self, left_seg, right_seg):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(comparison_uuid, 'left', left_seg),
            media_url_2=self.convert_segment_to_media_url(comparison_uuid, 'right', right_seg),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""

        comparison_id = self._create_comparison_in_webapp(left_seg, right_seg)
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "id": comparison_id,
            "label": None
        }

        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in range(0,2) ]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        from human_feedback_api import Comparison

        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is no response yet, so we just wait
