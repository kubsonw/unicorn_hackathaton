import sagemaker
from sagemaker.pytorch import PyTorch


sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/pytorch-alerts'

role = 'arn:aws:iam::753227428434:role/service-role/AmazonSageMaker-ExecutionRole-20180725T102686'

#inputs = sagemaker_session.upload_data(path='data_short', bucket=bucket, key_prefix=prefix)
#print('input spec (in this case, just an S3 path): {}'.format(inputs))

estimator = PyTorch(entry_point='model.py',
	role=role,
	framework_version='1.0.0',
	train_instance_count=1,
	train_instance_type='ml.p3.2xlarge',
	hyperparameters={
		'lr': 1e-3,
		'epochs': 100,
		'log-interval': 1,
		'backend': 'gloo'
	})
estimator.fit({'training': 's3://sagemaker-us-east-1-753227428434/sagemaker/pytorch-alerts'})
