require 'nn'
require 'cudnn'
require 'cunn'

local Sequential = nn.Sequential
local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Avg = cudnn.SpatialAveragePooling
local Identity = nn.Identity
local ConcatTable = nn.ConcatTable
local Parallel = nn.ParallelTable
local Add = nn.CAddTable
local Dropout = nn.Dropout
local Prod = nn.CMulTable
local BN = cudnn.SpatialBatchNormalization
local SelectTable = nn.SelectTable


local function ShareGradInput(module, key)
   assert(key)
   module.__shareGradInputKey = key
   return module
end

function GRCL(nIn, nOut, nIter, fSiz, rSiz, pad, opt)
        nIter = nIter or 3
        fSiz = fSiz or 3
        rSiz = rSiz or 3

        function getBlock_(nIn, nOut)
          local net = nn.Sequential()
          net:add(ShareGradInput(BN(nIn), 'first'))
          net:add(cudnn.ReLU(true))
          conv = cudnn.SpatialConvolution(nIn, nOut, 1, 1, 1, 1, 0, 0)
          net:add(conv)
          return net
        end

        function getBlock(nIn,nOut)
          local net = nn.Sequential()
          net:add(ShareGradInput(BN(nIn), 'first'))
          net:add(cudnn.ReLU(true))
          net:add(cudnn.SpatialConvolution(nIn, nOut, 3, 3, 1, 1, 1, 1):noBias())
          if opt.dropRate > 0 then net:add(nn.Dropout(opt.dropRate)) end
          net:add(BN(nOut))
          net:add(cudnn.ReLU(true))
          net:add(cudnn.SpatialConvolution(nOut, nOut, 3, 3, 1, 1, 1, 1):noBias())
          return net
        end

        function getBlock_feed(nIn,nOut)
          local net = nn.Sequential()
          net:add(ShareGradInput(BN(nIn), 'first'))
          net:add(cudnn.ReLU(true))
          net:add(cudnn.SpatialConvolution(nIn, nOut, 3, 3, 1, 1, 1, 1):noBias())
          return net
        end
       
        function getBlock_feed_(nIn, nOut)
          local net = nn.Sequential()
          net:add(ShareGradInput(BN(nIn), 'first'))
          net:add(cudnn.ReLU(true))
          conv = cudnn.SpatialConvolution(nIn, nOut, 1, 1, 1, 1, 0, 0):noBias()
          net:add(conv)
          return net
        end


        local nets = {}
        for i = 1, nIter do
          local net = Sequential()
          local rec = Sequential()
          local rec_res = Sequential()
          local concat_rec0 = ConcatTable()
          local concat_rec1 = ConcatTable()
          local rBlock = getBlock(nOut, nOut)
          local rBlock_res = getBlock_(nOut, nOut)
          local seq00 = Sequential()
          local concat0 = ConcatTable()
          local parall0 = Parallel()
          local concat1 = ConcatTable()
          local parall1 = Parallel()
          local gate_  = Sequential()
          local gate_1 = Sequential()
          local gate_2 = Sequential()

          if i == 1 then
            rec:add(Identity())
          else
            rec:add(nets[i - 1])
          end

          concat_rec0:add(Identity()):add(Identity())
          if i==1 then
            seq00:add(SelectTable(1)):add(rec)
          else
            seq00:add(rec)
          end

          local paraz = Parallel()
          local concat00 = ConcatTable()
          local seq_1 = Sequential() 

          concat0:add(seq00):add(Identity())

          paraz:add(concat_rec0):add(SelectTable(2))
          concat00:add(SelectTable(1)):add(paraz)

          parall1:add(Sequential():add(SelectTable(2)):add(rBlock_res)):add(Identity())
          concat1:add(Sequential():add(SelectTable(1)):add(SelectTable(1)):add(rBlock)):add(parall1)
          parall0:add(Identity()):add(concat1)

          gate_:add(Parallel():add(Identity()):add(Identity())):add(Add()):add(nn.Sigmoid())
          gate_1:add(Parallel():add(Identity()):add(gate_)):add(Prod())
          gate_2:add(Parallel():add(Identity()):add(gate_1)):add(Add())

          net:add(concat0):add(concat00):add(parall0):add(gate_2)

          table.insert(nets, net)
     end
     
     local fInit = getBlock_feed(nIn, nOut)
     local fInit_res = getBlock_feed_(nIn, nOut)
     local concat = ConcatTable()
     concat:add(fInit):add(fInit_res)

     return Sequential():add(concat):add(nets[#nets])

     end


