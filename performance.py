"""Store time and communication data volume for each worker"""

from typing import Callable, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceStore:
    """Implement store for performance data"""

    def __init__(self):
        self._local_train_times = []
        self._forward_pass_times = []
        self._local_val_times = []
        self._grad_reduce_times = []
        self._val_metric_reduce_times = []
        self._emb_broadcast_time = 0.0
        self._compute_local_embs_times = []
        self._load_gpu_times = []
        self._cv_message_passing_t = 0
        self._cv_grad_reduce_t = 0
        self._local_grad_encryption = []
        self._compute_loss_times = []
        self._back_pass_times = []
        
        
    def get_compute_loss_times(self):
        """Get compute loss times"""
        return self._compute_loss_times
    
    def add_compute_loss_time(self, time_t: float):
        """Add compute loss time"""
        self._compute_loss_times.append(time_t)
        
    def get_backward_pass_times(self):
        """Get backward pass times"""
        return self._back_pass_times
    
    def add_backward_pass_time(self, time_t: float):
        """Add backward pass time"""
        self._back_pass_times.append(time_t)
        
    def get_mean_backward_pass_time(self):
        """Get mean backward pass time"""
        return np.mean(self._back_pass_times)
    
    def get_mean_compute_loss_time(self):
        """Get mean compute loss time"""
        return np.mean(self._compute_loss_times)

    def get_forward_pass_times(self):
        """Get forward pass times"""
        return self._forward_pass_times
    
    def get_mean_forward_pass_time(self):
        """Get mean forward pass time"""
        return np.mean(self._forward_pass_times)
    
    def get_mean_local_train_time(self):
        """Get mean local train time"""
        return np.mean(self._local_train_times)
    
    def add_forward_pass_time(self, time_t: float):
        """Add forward pass time"""
        self._forward_pass_times.append(time_t)

    def get_mean_grad_encryption(self):  
        """Get mean grad encryption time"""  
        return np.mean(self._local_grad_encryption)
    
    def get_std_grad_reduce_times(self):
        """Get std grad reduce times"""
        return np.std(self._grad_reduce_times)
    
    def get_std_forward_pass_time(self):
        """Get std forward pass time"""
        return np.std(self._forward_pass_times)
    
    def get_std_grad_encryption(self):
        """Get std grad encryption time"""
        return np.std(self._local_grad_encryption)
    
    def add_local_grad_encryption(self, time_t: float):
        """Add local grad encryption time"""
        self._local_grad_encryption.append(time_t)
        
    def get_std_local_train_time(self):
        """Get std local train times"""
        return np.std(self._local_train_times)
        
    def get_std_compute_loss_time(self):
        """Get std compute loss times"""
        return np.std(self._compute_loss_times)
    
    def get_std_backward_pass_time(self):
        """Get std backward pass times"""
        return np.std(self._back_pass_times)
    
    def get_communication_volume(self):
        """Get communication volume"""
        return int(self._cv_grad_reduce_t + self._cv_message_passing_t)

    def get_cv_message_passing_t(self):
        """Get simulated communication volume for message-passing"""
        return int(self._cv_message_passing_t)

    def get_cv_grad_reduce_t(self):
        """Get simulated communication volume for grad-reduce by central server"""
        return int(self._cv_grad_reduce_t)

    def get_local_train_times(self):
        """Get local train times"""
        return self._local_train_times

    def get_local_val_times(self):
        """Get local val times"""
        return self._local_val_times

    def get_grad_reduce_times(self):
        """Get grad reduce times"""
        return self._grad_reduce_times
    
    def get_mean_grad_reduce_times(self):
        """Get grad reduce times"""
        return np.mean(self._grad_reduce_times)

    def get_val_metric_reduce_times(self):
        """Get val metric reduce times"""
        return self._val_metric_reduce_times

    def get_emb_broadcast_time(self):
        """Get emb broadcast time"""
        return self._emb_broadcast_time

    def get_compute_local_embs_times(self):
        """Get compute local embs times"""
        return self._compute_local_embs_times

    def get_load_gpu_times(self):
        """Get load gpu times"""
        return self._load_gpu_times

    def get_aggregate_metric(
        self, metric_get_f: Callable, metric_aggregate_f: Optional[Callable] = np.mean
    ):
        """Get any aggregate metric"""
        if not metric_aggregate_f:
            raise ValueError("metric_aggregate_f cannot be None")
        return metric_aggregate_f(metric_get_f())

    def add_cv_message_passing_t(self, t: float):
        """Add simulated communication volume for message-passing"""
        self._cv_message_passing_t += t

    def add_cv_grad_reduce_t(self, t: float):
        """Add simulated communication volume for grad-reduce by central server"""
        self._cv_grad_reduce_t += t

    def add_local_train_time(self, time_t: float):
        """Add local train time"""
        self._local_train_times.append(time_t)

    def add_local_val_time(self, time_t: float):
        """Add local val time"""
        self._local_val_times.append(time_t)

    def add_grad_reduce_time(self, time_t: float):
        """Add grad reduce time"""
        self._grad_reduce_times.append(time_t)

    def add_val_metric_reduce_time(self, time_t: float):
        """Add val metric reduce time"""
        self._val_metric_reduce_times.append(time_t)

    def set_emb_broadcast_time(self, time_t: float):
        """Set emb broadcast time"""
        self._emb_broadcast_time = time_t

    def add_compute_local_embs_time(self, time_t: float):
        """Add compute local embs time"""
        self._compute_local_embs_times.append(time_t)

    def add_load_gpu_time(self, time_t: float):
        """Add load gpu time"""
        self._load_gpu_times.append(time_t)

    def get_all_metrics(self):
        """Get all metrics"""
        return {
            "communication_volume": self.get_communication_volume(),
            "cv_message_passing_t": self.get_cv_message_passing_t(),
            "cv_grad_reduce_t": self.get_cv_grad_reduce_t(),
            "local_train_time": self.get_local_train_times(),
            "local_val_time": self.get_local_val_times(),
            "grad_reduce_time": self.get_grad_reduce_times(),
            "val_metric_reduce_time": self.get_val_metric_reduce_times(),
            "emb_broadcast_time": self.get_emb_broadcast_time(),
        }

    def get_necessary_time_metrics(self):
        """Get a selection of useful metrics for time"""
        return {
            "total_local_train_time": sum(self.get_local_train_times()),
            "total_grad_reduce_time": sum(self.get_grad_reduce_times()),
            "total_emb_broadcast_time": self.get_emb_broadcast_time(),
            "total_compute_local_embs_time": sum(self.get_compute_local_embs_times()),
            "avg_local_train_time": np.mean(self.get_local_train_times()),
            "avg_grad_reduce_time": np.mean(self.get_grad_reduce_times()),
        }

    def get_necessary_cv_metrics(self):
        """Get a selection of useful metrics for communication volume"""
        return {
            "total_cv_message_passing_t": self.get_cv_message_passing_t(),
            "total_cv_grad_reduce_t": self.get_cv_grad_reduce_t(),
            "total_communication_volume": self.get_communication_volume(),
        }