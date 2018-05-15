'''
model

Created on May 11 2018 16:41 
#@author: Kevin Le 
'''
import caffe
import numpy as np

class ClassModel(object):
    def __init__(self):
        pass

    def prep_for_training(self, solver_proto, weights, LMDBs, gpu_id):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        # self.append_proto(proto=train_proto, LMDB=LMDBs)

        self.solver = caffe.SGDSolver(solver_proto)
        self.solver.net.copy_from(weights)

    def load_solver_proto(self):
        pass

    def train(self, n=1):
        self.solver.step(n)

    def load(self):
        pass

    def save(self, model_fn):
        self.solver.net.save(model_fn)

    def append_proto(self, proto, LMDB):
        pass

    def prep_for_deploy(self, deploy_proto, weights, gpu_id):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.deploy = caffe.Net(deploy_proto, caffe.TEST, weights=weights)

    def forward(self, batch, batch_size):
        self.deploy.blobs['data'].data[:batch_size] = batch
        self.deploy.forward()
        return np.copy(self.deploy.blobs['prob'].data[:batch_size,:])

def main():
    pass

if __name__ == '__main__':
    pass