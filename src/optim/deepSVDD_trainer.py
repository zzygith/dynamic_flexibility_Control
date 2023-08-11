from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
from .heaterExchangerState import HENState
from .reactorCooler2dState import RC2DState
from .reactorCooler5dState import RC5DState
from .buffer1dState_NoControl import Buffer
from .buffer1dState_Control import BufferControl


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, dataForConstraints: str = 'mine'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        self.dataForConstraints=dataForConstraints

        #self.eps=1e-6 #to avoid inf
        self.eps=1e-10
        self.eta=100 #weighting for unsatisfied constraints #1000 #10 #good eta10 sa100
        #self.satisfiedP = 1000 #relu, with counting penalty
        #self.satisfiedP = 10 #tanh, all parameter 100:10, center(2,2)
        self.satisfiedP = 10
        self.penalty = torch.tensor(-1.0, device=self.device)

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.lossHistory=[]

    def train(self, dataset: BaseADDataset, net: BaseNet):

        constraintsFunc=self.conditionFunctionList(self.dataForConstraints)
        stateModel=self.stateModelFunction(self.dataForConstraints)
        if self.dataForConstraints=='mine_heater_1d':
            nU=50
            uRangeLow=0
            uRangeHigh=250
            uLength=1
            ######added
            thetaNumber=1
            predictionStateNumber=4

        elif self.dataForConstraints=='mine_reactorCooler_2d':
            #nU=100
            nU=400
            uRangeLow = [0,3.00]
            uRangeHigh = [6.804,3.56]
            uLength=2
            ######added
            thetaNumber=2
            predictionStateNumber=5

        elif  self.dataForConstraints=='mine_reactorCooler_5d':
            #nU=100
            nU=400
            uRangeLow = [0,3.00]
            uRangeHigh = [6.804,3.56]
            uLength=2
            ######added
            thetaNumber=5
            predictionStateNumber=5

        elif self.dataForConstraints=='mine_dynamic_1dtheta_noControl':
            nU=1600
            uRangeLow=0
            uRangeHigh=800
            uLength=1
            ######added
            thetaNumber=1
            predictionStateNumber=1

        elif self.dataForConstraints=='mine_dynamic_1dtheta_Control':
            nU=50 # 把每个theta重复的次数,即每个theta对应多少个U
            uRangeLow=0
            uRangeHigh=5**0.5/10
            uLength=1
            ######added
            thetaNumber=1
            predictionStateNumber=1
            tRangeLow=0
            tRangeHigh=800
            nTime=1600 # 把每个thetaAndU组合重复的次数,即每个thetaAndU组合对应多少个T

        logger = logging.getLogger()
        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                expanedInput=inputs.repeat(1,nU).reshape(-1,thetaNumber)#2即输入theta的维度数量,100即随机U的数量
                #uRandom=np.random.uniform(uRangeLow,uRangeHigh,size=(nU,uLength))
                uRandom=torch.tensor(np.random.uniform(uRangeLow,uRangeHigh,size=(nU*inputs.shape[0],uLength)),dtype=torch.float32).to(self.device)
                expandedInputAndU=torch.cat((expanedInput,uRandom),1)

                expandedInputAndUWaitingForT=expandedInputAndU.repeat(1,nTime).reshape(-1,nTime,uLength+thetaNumber)
                tRandom=torch.tensor(np.random.uniform(tRangeLow,tRangeHigh,size=(expandedInputAndUWaitingForT.shape[0],nTime,1)),dtype=torch.float32).to(self.device)
                expandedInputAndUAndT=torch.cat((expandedInputAndUWaitingForT,tRandom),2)
                ######
                expandedInputAndUAndTReForNN=expandedInputAndUAndT.reshape(-1,uLength+thetaNumber+1) # timeT 只有1维
                exStates=stateModel(expandedInputAndUAndTReForNN).reshape(-1,nTime,predictionStateNumber)
                exStatesInputAndUAndT=torch.cat((expandedInputAndUAndT,exStates),2).reshape(-1,nU,nTime,uLength+thetaNumber+1+predictionStateNumber)
                distConstrainFlagTensor=constraintsFunc(exStatesInputAndUAndT)

                # exStates=stateModel(expandedInputAndU)
                # exStatesInputAndU=torch.cat((expandedInputAndU,exStates),1)
                # exStatesInputAndU=exStatesInputAndU.reshape(-1,nU,uLength+thetaNumber+predictionStateNumber)#5是状态的数量,2即输入状态的维度数量,2是u的维度数量                
                # distConstrainFlagTensor=constraintsFunc(exStatesInputAndU,nU)

                ####check the satisfied theta
                #losses=torch.where(distConstrainFlagTensor == 0, self.satisfiedP*dist, self.eta * ((dist + self.eps)**self.penalty))
                #logger.info(exStatesInputAndU)

                inputsTheta=inputs.cpu().detach().numpy()
                #inputsTheta=inputsTheta.flatten()[0]

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                distArray = dist.cpu().detach().numpy()
##################################################################################               
                # distConstrainFlag=np.zeros_like(distArray)
                # for i in range(0,len(distConstrainFlag)):
                #     satisfiedNum=0
                #     # nU=300
                #     # uRangeLow=0
                #     # uRangeHigh=3

                #     # if dataForConstraintsChoice=='mine':
                #     # nU=50
                #     # uRangeLow=0
                #     # uRangeHigh=250
                #     # uRandom=np.random.uniform(uRangeLow,uRangeHigh,nU)

                #     # nU=50
                #     # U_min = [0,3.00]
                #     # U_max = [6.804,3.56]
                #     # uRandom = np.random.uniform(low=U_min, high=U_max, size=(nU,2))

                #     uRandom=np.random.uniform(uRangeLow,uRangeHigh,size=(nU,uLength))

                #     for k in uRandom:
                #         #if self.condition(inputsTheta[i],k):
                #         if constraintsFunc(inputsTheta[i],k,stateModel):
                #             #satisfiedNum=satisfiedNum+1
                #             satisfiedNum=1
                #             break
                #     distConstrainFlag[i]=satisfiedNum

                # distConstrainFlagTensor=torch.tensor(distConstrainFlag).to(self.device)
                # ####check the satisfied theta
                # # logger.info(distConstrainFlagTensor)

                # # satisfiedTheta = torch.where(distConstrainFlagTensor > 0, torch.flatten(inputs[:,0]), distConstrainFlagTensor)

                # #losses=torch.where(distConstrainFlagTensor > 0, self.satisfiedP*dist*distConstrainFlagTensor, self.eta * ((dist + self.eps)**self.penalty))
                # losses=torch.where(distConstrainFlagTensor > 0, self.satisfiedP*dist, self.eta * ((dist + self.eps)**self.penalty))
##############################################################################################################################################
                # logger.info(satisfiedTheta)
                # logger.info(losses)
                
                # if epoch%10==0:
                #     logger.info(satisfiedTheta)  
                #     logger.info(dist)
                #     logger.info(losses)     
                losses=torch.where(distConstrainFlagTensor > 0, self.eta * ((dist + self.eps)**self.penalty),self.satisfiedP*dist)                
                loss = torch.mean(losses)

                # nU=3
                # uRangeLow=0
                # uRangeHigh=3
                # uRandom=np.random.uniform(uRangeLow,uRangeHigh,nU)

                # allUnsatisfiedFlag=True
                # lossTRY=0.0

                # for i in uRandom:
                #     logger.info('inputsTheta %f' %inputsTheta)
                #     logger.info('uRandom %f' %i)
                #     logger.info(inputs)
                #     if self.condition(inputsTheta,i):
                #         lossTRY=lossTRY+torch.mean(dist)
                #         allUnsatisfiedFlag=False

                # if allUnsatisfiedFlag:
                #     lossTRY=torch.mean(dist)**(-1)

                # if self.objective == 'soft-boundary':
                #     scores = dist - self.R ** 2
                #     loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                ##################################
                #else:
                #    scores,indices = torch.sort(dist)
                #    loss = 0
                    
                #    #sp: 0.95, random data: 0.9
                #    for i in range(1,6):
                #        loss = loss + 5*i*scores[int(0.9*len(scores))-i]
                #        loss = loss - i*scores[int(0.9*len(scores))+i]
                    
                #loss.backward()
                #optimizer.step()
                ###################################
                # else:
                #     loss = torch.mean(dist)




                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            if epoch == self.n_epochs - 1:
                print("LOSS", loss_epoch / n_batches)
            
            self.lossHistory.append(loss_epoch)
        #self.lossHistory=torch.cat(self.lossHistory, dim=1)
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')
#####################################################################        

        
########################################################################
        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        pass



    # def conditionFunctionList(self,dataForConstraintsChoice):
    #     if dataForConstraintsChoice=='mine':
    #         def constraint(theta,z):
    #             flag=False
    #             if z-theta<=0 and -z-theta/3+4/3<=0 and z+theta-4<=0:
    #                 flag=True
    #             return flag
    #         return constraint

    #     elif dataForConstraintsChoice=='mine_heater_1d':
    #         def constraint(theta,z):
    #             flag=False
    #             if -25*theta+z-0.5*theta*z+10<=0 and -190*theta+z+10<=0 and -270*theta+z+250<=0 and 260*theta-z-250<=0:
    #                 flag=True
    #             return flag
    #         return constraint
 

#look for the reason why it's so slow
    # def conditionFunctionList(self,dataForConstraintsChoice):
    #     if dataForConstraintsChoice=='mine':
    #         def constraint(theta,z,stateModel):
    #             flag=False
    #             if z-theta<=0 and -z-theta/3+4/3<=0 and z+theta-4<=0:
    #                 flag=True
    #             return flag
    #         return constraint

    #     elif dataForConstraintsChoice=='mine_heater_1d':
    #         def constraint(theta,z,stateModel):
    #             flag=False
    #             mlk=np.array([[1.5,22.0]])
    #             mlkk=stateModel(torch.tensor(mlk,dtype=torch.float32).to(self.device))
    #             if -25*theta+z-0.5*theta*z+10<=0 and -190*theta+z+10<=0 and -270*theta+z+250<=0 and 260*theta-z-250<=0:
    #                 flag=True
    #             return flag
    #         return constraint

##############################################################################
########speed up 后
    def conditionFunctionList(self,dataForConstraintsChoice):
        # if dataForConstraintsChoice=='mine':
        #     def constraint(theta,z,stateModel):
        #         flag=False
        #         if z-theta<=0 and -z-theta/3+4/3<=0 and z+theta-4<=0:
        #             flag=True
        #         return flag
        #     return constraint

        if dataForConstraintsChoice=='mine_dynamic_1dtheta_Control':
            def constraint(exStatesInputAndUAndT):
                constResults=torch.zeros([exStatesInputAndUAndT.shape[0],exStatesInputAndUAndT.shape[1],exStatesInputAndUAndT.shape[2],2], dtype=torch.float32) #2是const的数量,即predictionStateNumber.to(self.device)#1是const的数量
                h=exStatesInputAndUAndT[:,:,:,3:4]
                constraint1=1-h
                constraint2=h-10
                constResults[:,:,:,0:1]=constraint1
                constResults[:,:,:,1:2]=constraint2
                # constResults[:,:,2:3]=constraint3
                # constResults[:,:,3:4]=constraint4                
                constResultsRelu=torch.relu(constResults)
                # constResultFlag=constResultsRelu.clone()
                # constResultFlag[constResultFlag==0]=1
                # constResultFlag[constResultFlag!=0]=0

                constResultsReluSum1=torch.sum(constResultsRelu,dim=2)
                #constResultsReluSum2=torch.prod(constResultsReluSum1,dim=1)
                constResultsReluSum2=torch.sum(constResultsReluSum1,dim=2)
                constResultsReluSum3=torch.prod(constResultsReluSum2,dim=1)
                return constResultsReluSum3
            return constraint   

        # elif dataForConstraintsChoice=='mine_reactorCooler_5d':
        #     def constraint(theta,z,stateModel):
        #         flag=False
        #         Ca0=32.04
        #         #stateInput=torch.tensor(np.array([theta.flatten(),z])).to(self.device)
        #         #stateInput=torch.tensor(np.append(theta.flatten()/10.0,z.flatten()),dtype=torch.float32).to(self.device)
        #         #stateInput = torch.tensor(np.append( [a/b for a,b in zip(theta.flatten(),[10.0,10.0,100.0,100.0,1.0])], z.flatten()), dtype=torch.float32).to(self.device)
        #         stateInput = torch.tensor(np.append(theta.flatten(), z.flatten()), dtype=torch.float32).to(self.device)
        #         #states=stateModel(stateInput).cpu().detach().numpy().flatten()
        #         states=stateModel(stateInput)
        #         states=torch.flatten(states)
        #         Tw1 = stateInput[2] * 100.0
        #         Tw2=stateInput[6]*100.0
        #         Ca1=states[0]
        #         T1=states[1]*10+390.
        #         T2=states[2]*10+300.
        #         constraint1=(Ca0-Ca1)/Ca0
        #         constraint3=T1-T2
        #         constraint4=T1-Tw2-11.1
        #         constraint5=T2-Tw1-11.1
        #         if constraint1>=0.9 and T1>=311 and T1<=389 and constraint3>=0 and constraint4>=0 and constraint5>=0:
        #              flag=True
        #         return flag
        #     return constraint



###############################################################################

    def stateModelFunction(self,dataForConstraintsChoice):
        # if dataForConstraintsChoice == 'mine_dynamic_1dtheta_noControl':
        #     dataRoot = './optim/buffer_noControl.pt'
        #     BF1DStateModel = Buffer().to(self.device)
        #     BF1DStateModel.load_state_dict(torch.load(dataRoot))
        #     return BF1DStateModel
        
        if dataForConstraintsChoice == 'mine_dynamic_1dtheta_Control':
            dataRoot = './optim/buffer_Control.pt'
            BF1DStateModel = BufferControl().to(self.device)
            BF1DStateModel.load_state_dict(torch.load(dataRoot))
            return BF1DStateModel
        

            # def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
    #     """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    #     n_samples = 0
    #     c = torch.zeros(net.rep_dim, device=self.device)
    
    #     net.eval()
    #     with torch.no_grad():
    #         for data in train_loader:
    #             # get the inputs of the batch
    #             inputs, _, _ = data
    #             inputs = inputs.to(self.device)
    #             outputs = net(inputs)
    #             n_samples += outputs.shape[0]
    #             c += torch.sum(outputs, dim=0)
    #     c /= n_samples
    
    #     c[(abs(c) < 0.01) & (c < 0)] = -0.01
    #     c[(abs(c) < 0.01) & (c >= 0)] = 0.01
    
    
    #     return c


    # def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
    #     """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    #     n_samples = 0
    #     c = torch.zeros(net.rep_dim, device=self.device)
    
    #     net.eval()
    #     with torch.no_grad():
    #         for data in train_loader:
    #             # get the inputs of the batch
    #             inputs, _, _ = data
    #             inputs = inputs.to(self.device)
    #             outputs = net(inputs)
    #             n_samples += outputs.shape[0]
    #             c += torch.sum(outputs, dim=0)
    #     c /= n_samples
    
    #     c[(abs(c) < eps) & (c < 0)] = -eps
    #     c[(abs(c) < eps) & (c > 0)] = eps
    
    
    #     return c


    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.ones(net.rep_dim, device=self.device)*(2)
        return c

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
