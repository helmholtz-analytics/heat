import heat as ht
import torch
import unittest
import heat.optim as optim
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.mnist import MNISTDataset



class TestBatchNormalization1D(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm1d(16)

            def forward(self, x):
                return self.batch(x)



        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_1d(self):
       
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()

            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)
                data_list.append(torch.randn(32, 16, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.), 1e-4)
                loss.backward
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()


class TestBatchNormalization2D(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm2d(16)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_2d(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.), 1e-4)
                loss.backward()
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()

class TestBatchNormalization3D(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm3d(16)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_3d(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28, 28, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.*28.), 1e-4)
                loss.backward()
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()



class TestBatchNormalization1DNoAffine(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = ht.nn.Conv2d(1, 1, 3, 1)
                self.batch = ht.nn.SyncBatchNorm(16, affine=False)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 3, 1)
                self.batch = torch.nn.BatchNorm1d(16, affine=False)

            def forward(self, x):
                return self.batch(x)



        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )




    def test_batchnorm_float32_1d_noaffine(self):
       
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()

            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)
                data_list.append(torch.randn(32, 16, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank


            output = self.dp_model(data_list[ht.MPI_WORLD.rank])
            output_torch = self.model_torch(data)
            output_torch = output_torch[32*rank:32*(rank+1)]
            self.assertNotAlmostEqual(output.std().item(), 1.0)
            self.assertEqual(output.shape, (32, 16, 28))
            self.assertEqual(data.dtype, output.dtype)
            self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.), 1e-4)

class TestBatchNormalization2DNoAffine(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = ht.nn.Conv2d(1, 1, 3, 1)
                self.batch = ht.nn.SyncBatchNorm(16, affine=False)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 3, 1)
                self.batch = torch.nn.BatchNorm2d(16, affine=False)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_2d_noaffine(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            output = self.dp_model(data_list[ht.MPI_WORLD.rank])
            output_torch = self.model_torch(data)
            output_torch = output_torch[32*rank:32*(rank+1)]
            loss = F.mse_loss(output.mean(), target)
            loss_torch = F.mse_loss(output_torch.mean(), target)
            self.assertNotAlmostEqual(output.std().item(), 1.0)
            self.assertEqual(output.shape, (32, 16, 28, 28))
            self.assertEqual(data.dtype, output.dtype)
            self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.), 1e-4)

class TestBatchNormalization3DNoAffine(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = ht.nn.Conv2d(1, 1, 3, 1)
                self.batch = ht.nn.SyncBatchNorm(16, affine=False)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 3, 1)
                self.batch = torch.nn.BatchNorm3d(16, affine=False)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_3d_noaffine(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank


            output = self.dp_model(data_list[ht.MPI_WORLD.rank])
            output_torch = self.model_torch(data)
            output_torch = output_torch[32*rank:32*(rank+1)]
            loss = F.mse_loss(output.mean(), target)
            loss_torch = F.mse_loss(output_torch.mean(), target)
            self.assertNotAlmostEqual(output.std().item(), 1.0)
            self.assertEqual(output.shape, (32, 16, 28, 28, 28))
            self.assertEqual(data.dtype, output.dtype)
            self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.*28.), 1e-4)



class TestBatchNormalization1DNoMoment(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16, momentum=None)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm1d(16, momentum=None)

            def forward(self, x):
                return self.batch(x)



        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_1d_nomoment(self):
       
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()

            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)
                data_list.append(torch.randn(32, 16, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.), 1e-4)
                loss.backward
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()


class TestBatchNormalization2DNoMoment(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16, momentum=None)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm2d(16, momentum=None)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_2d(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.), 1e-4)
                loss.backward()
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()

class TestBatchNormalization3DNoMoment(unittest.TestCase):

    def setUp(self):

        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.batch = ht.nn.SyncBatchNorm(16, momentum=None)
            def forward(self, x):
                return self.batch(x)

        class NetTorch(torch.nn.Module):
            def __init__(self):
                super(NetTorch, self).__init__()
                self.batch = torch.nn.BatchNorm3d(16, momentum=None)

            def forward(self, x):
                return self.batch(x)
                


        self.no_cuda = False
        self.lr = 2.0
        self.iterations = 10
        self.use_cuda = True #not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = Net().to(self.device)
        self.model_torch = NetTorch().to(self.device)

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer_torch = optim.Adadelta(self.model_torch.parameters(), lr=self.lr)

        self.dp_optim = ht.optim.DataParallelOptimizer(self.optimizer, blocking=False)
        self.dp_model = ht.nn.DataParallel(
            self.model, comm=ht.MPICommunication(), optimizer=self.dp_optim, blocking_parameter_updates=False
        )


    def test_batchnorm_float32_3d(self):
    
        if torch.cuda.is_available():

            self.model.train()
            data_list = list()
            
            for tt1 in range(ht.MPI_WORLD.size):
                torch.manual_seed(tt1)   
                data_list.append(torch.randn(32, 16, 28, 28, 28).to(self.device))

            target = data_list[ht.MPI_WORLD.rank].mean()
            data = torch.cat(data_list, 0)
            rank = ht.MPI_WORLD.rank

            for tt1 in range(self.iterations):

                self.dp_optim.zero_grad()
                self.optimizer_torch.zero_grad()
                output = self.dp_model(data_list[ht.MPI_WORLD.rank])
                output_torch = self.model_torch(data)
                output_torch = output_torch[32*rank:32*(rank+1)]
                loss = F.mse_loss(output.mean(), target)
                loss_torch = F.mse_loss(output_torch.mean(), target)
                self.assertNotAlmostEqual(output.std().item(), 1.0)
                self.assertEqual(output.shape, (32, 16, 28, 28, 28))
                self.assertEqual(data.dtype, output.dtype)
                self.assertLess(torch.norm(output-output_torch).item()/(32.*16.*28.*28.*28.), 1e-4)
                loss.backward()
                loss_torch.backward()

                self.dp_optim.step()
                self.optimizer_torch.step()


