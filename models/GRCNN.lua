require 'nn'
require 'cunn'
require 'cudnn'
require 'models/GRCL'

local function createModel(opt)

   --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
   local dropRate = opt.dropRate

   --# channels before entering the first GRCL
   local nChannels = 64
   
   local ave_T = (opt.depth * 1.0 - 7)/6
   local T1 = math.ceil((ave_T * 1 )/2)
   local T2 = math.ceil((ave_T * 2 )/2)
   local T3 = math.ceil((ave_T * 3 )/2)

  -- local T = N - 1 

   function addGRCL(model, nIn, nOut, T, opt)
         model:add(GRCL(nIn, nOut, T, 3, 3, 1, opt))    
         channels = nOut
         return channels
   end


   function addTransition(model, nChannels, nOutChannels, opt, last, pool_size)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))      
      if last then
         model:add(cudnn.SpatialAveragePooling(pool_size, pool_size))
         model:add(nn.Reshape(nChannels))      
      else
         model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0):noBias())
         if opt.dropRate > 0 then model:add(nn.Dropout(opt.dropRate)) end
         model:add(cudnn.SpatialAveragePooling(2, 2))
      end      
   end

   -- Build GRCNN
   local model = nn.Sequential()

   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' or opt.dataset == 'svhn' then

      --Initial convolution layer
      model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1):noBias())      

      --1st GRCL and transition
      nChannels = addGRCL(model, 64, 40, T1, opt)
      addTransition(model, nChannels, nChannels, opt)
    
      --2nd GRCL and transition
      nChannels = addGRCL(model, nChannels, 80, T2, opt)
      addTransition(model, nChannels, nChannels, opt)

      --3rd GRCL and transition
      nChannels = addGRCL(model,nChannels,  120, T3, opt)
      addTransition(model, nChannels, nChannels, opt, true, 8)

   elseif opt.dataset == 'imagenet' then
      local ave_T = (opt.depth * 1.0 - 14)/6
      local T1 =  3
      local T2 =  math.ceil((ave_T * 1)/2) 
      local T3 =  math.ceil((ave_T * 2)/2) 
      local T4 =  math.ceil((ave_T * 3)/2)

      --Initial transforms (224x224)
      model:add(cudnn.SpatialConvolution(3, 64, 7,7, 2,2, 3,3):noBias())
      model:add(cudnn.SpatialBatchNormalization(64))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

      --Dense-Block 1 and transition (56x56)
      nChannels = addGRCL(model, 64, 64, T1, opt)
      addTransition(model, nChannels, nChannels, opt)

      --Dense-Block 2 and transition (28x28)
      nChannels = addGRCL(model, nChannels, 144, T2, opt)
      addTransition(model, nChannels, nChannels, opt)

      --Dense-Block 3 and transition (14x14)
      nChannels = addGRCL(model, nChannels, 256, T4, opt)
      addTransition(model, nChannels, nChannels, opt)

      --Dense-Block 4 and transition (7x7)
      nChannels = addGRCL(model,nChannels,  512, T3, opt)
      addTransition(model, nChannels, nChannels, opt, true, 7)
   end


   if opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
   elseif opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
   elseif opt.dataset == 'imagenet' then
      model:add(nn.Linear(nChannels, 1000))
   elseif opt.dataset == 'svhn' then
      model:add(nn.Linear(nChannels, 10))
   end

   --Initialization following ResNet
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   print(model)
   return model
end

return createModel
