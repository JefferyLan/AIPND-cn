import sys
import argparse
from utils import *
from build_network import *


# python validate.py valid_dir checkpoint
# 选项：
# 使用 GPU 进行训练：python predict.py valid_dir checkpoint --gpu
# 结束

# TODO: Do validation on the test set
def validate(net, cuda_enabled, valid_loader):
    correct = 0
    total = 0
    #confusion_matrix = meter.ConfusionMeter(2)
    net.eval()
    for i, (images, labels) in enumerate(valid_loader):   
        val_input = Variable(images, True)
        val_label = Variable(labels.long(), True)
        if cuda_enabled:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        outputs = net(val_input)
        #confusion_matrix.add(score.data.squeeze(), labels.long())
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += val_label.size(0)
        correct += (predicted == val_label.cpu().long()).sum()
        
    accuracy = float(100 * correct) / float(total)
    print("Accuracy: [%d / %d] = %.4f \n" 
          % (correct, total, accuracy))
    net.train()
    
    return accuracy

def main(argv): 
    parser = argparse.ArgumentParser()
    parser.add_argument('valid_dir', metavar='valid_dir', nargs='+', help='image to validate')
    parser.add_argument('checkpoint', metavar='checkpoint', nargs='+', help='full path to load checkpoint')
    parser.add_argument('--gpu', help='enable gpu or not, 1 for enabled, otherwise disabled', type = int)
    args = parser.parse_args()  
    
    batch_size = 64 
    if args.valid_dir:
        #2.1 load valid data
        valid_loader = load_valid_data(args.valid_dir[0], batch_size)
    else:
        print('valid_dir is missing')
        return 
    
    #4. load checkpoints    
    net, class_to_idx, num_epochs = load_checkpoint(args.checkpoint[0])
    print("num_epochs=%d" % (num_epochs))
    # 5. valid
    cuda_enabled = False
    if args.gpu:
        if(args.gpu == 1 and torch.cuda.is_available()):
            cuda_enabled = True
    print("cuda_enabled=\n", cuda_enabled)
    if cuda_enabled:
        net.cuda()
    #2.2 valid network 
    accuracy = validate(net, cuda_enabled, valid_loader)
    
if __name__ == '__main__':
    main(sys.argv)

