export CUDA_VISIBLE_DEVICES=0,1,2,3
th main.lua -netType GRCNN -dataset imagenet -data /home_wjf/imagenet -batchSize 96 -LR 0.05 -nEpochs 100 -depth 66 -nGPU 4 -nThreads 16
