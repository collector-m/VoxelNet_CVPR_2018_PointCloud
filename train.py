import os
import sys
import time
import shutil
import argparse

import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from torch.nn.utils import clip_grad_norm_

from tensorboardX import SummaryWriter

from config import cfg
from utils.utils import box3d_to_label
from model.model import RPN3D
from loader.kitti import KITTI as Dataset
from loader.kitti import collate_fn

import pdb


parser = argparse.ArgumentParser(description = 'training')

parser.add_argument('--lr', type = float, default = 0.01, help = 'initial learning rate')
parser.add_argument('--alpha', type = float, default = 1.5, help = 'alpha in loss function')
parser.add_argument('--beta', type = float, default = 1, help = 'beta in loss function')

parser.add_argument('--max_epoch', type = int, default = 1, help = 'max epoch')
parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size')
parser.add_argument('--workers', type = int, default = 4)

parser.add_argument('--summary_interval', type = int, default = 100, help = 'iter interval for training summary')
parser.add_argument('--summary_val_interval', type = int, default = 200, help = 'iter interval for val summary')
parser.add_argument('--val_epoch', type = int, default = 10, help = 'epoch interval for dump val data')

parser.add_argument('--log_root', type = str, default = 'log')
parser.add_argument('--log_name', type = str, default = 'train.txt')
parser.add_argument('--tag', type = str, default = 'default', help = 'log tag')

parser.add_argument('--print_freq', default = 20, type = int, help = 'print frequency')

parser.add_argument('--resumed_model', type = str, default = '', help = 'if specified, load the specified model')
parser.add_argument('--saved_model', type = str, default = 'kitti_{}.pth.tar')

# For test data
parser.add_argument('--output_path', type = str, default = './preds', help = 'results output dir')
parser.add_argument('--vis', type = bool, default = True, help = 'set to True if dumping visualization')

args = parser.parse_args()


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in cfg.GPU_AVAILABLE)

    start_epoch = 0
    global_counter = 0
    min_loss = sys.float_info.max

    # Build data loader
    train_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'training'), shuffle = True, aug = True, is_testset = False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn,
                                  num_workers = args.workers, pin_memory = False)

    val_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'validation'), shuffle = False, aug = False, is_testset = False)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn,
                                num_workers = args.workers, pin_memory = False)
    val_dataloader_iter = iter(val_dataloader)

    # Build model
    model = RPN3D(cfg.DETECT_OBJ, args.alpha, args.beta)

    # Resume model if necessary
    if args.resumed_model:
        model_file = os.path.join(save_model_dir, args.resumed_model)
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file)
            start_epoch = checkpoint['epoch']
            global_counter = checkpoint['global_counter']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> Loaded checkpoint '{}' (epoch {}, global_counter {})".format(
                args.resumed_model, checkpoint['epoch'], checkpoint['global_counter'])))
        else:
            print(("=> No checkpoint found at '{}'".format(args.resumed_model)))

    model = nn.DataParallel(model).cuda()

    # Optimization scheme
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [150])

    # Init file log
    log = open(os.path.join(args.log_root, args.log_name), 'a')

    # Init TensorBoardX writer
    summary_writer = SummaryWriter(log_dir)

    # train and validate
    tot_epoch = start_epoch
    for epoch in range(start_epoch, args.max_epoch):
        # Learning rate scheme
        lr_sched.step()

        counter = 0
        batch_time = time.time()

        tot_val_loss = 0
        tot_val_times = 0

        for (i, data) in enumerate(train_dataloader):

            model.train(True)   # Training mode

            counter += 1
            global_counter += 1

            start_time = time.time()

            # Forward pass for training
            _, _, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(data)

            forward_time = time.time() - start_time

            loss.backward()

            # Clip gradient
            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()

            batch_time = time.time() - batch_time

            if counter % args.print_freq == 0:
                # Print training info
                info = 'Train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} ' \
                       'cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f}'.format(
                    counter, epoch + 1, args.max_epoch, loss.item(), reg_loss.item(), cls_loss.item(), cls_pos_loss_rec.item(),
                    cls_neg_loss_rec.item(), forward_time, batch_time)
                info = '{}\t'.format(time.asctime(time.localtime())) + info
                print(info)

                # Write training info to log
                log.write(info + '\n')
                log.flush()

            # Summarize training info
            if counter % args.summary_interval == 0:
                print("summary_interval now")
                summary_writer.add_scalars(str(epoch + 1), {'train/loss' : loss.item(),
                                                            'train/reg_loss' : reg_loss.item(),
                                                            'train/cls_loss' : cls_loss.item(),
                                                            'train/cls_pos_loss' : cls_pos_loss_rec.item(),
                                                            'train/cls_neg_loss' : cls_neg_loss_rec.item()}, global_counter)

            # Summarize validation info
            if counter % args.summary_val_interval == 0:
                print('summary_val_interval now')

                with torch.no_grad():
                    model.train(False)  # Validation mode

                    val_data = next(val_dataloader_iter)    # Sample one batch

                    # Forward pass for validation and prediction
                    probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data)

                    summary_writer.add_scalars(str(epoch + 1), {'validate/loss': loss.item(),
                                                                'validate/reg_loss': reg_loss.item(),
                                                                'validate/cls_loss': cls_loss.item(),
                                                                'validate/cls_pos_loss': cls_pos_loss_rec.item(),
                                                                'validate/cls_neg_loss': cls_neg_loss_rec.item()}, global_counter)

                    try:
                        # Prediction
                        tags, ret_box3d_scores, ret_summary = model.module.predict(val_data, probs, deltas, summary = True)

                        for (tag, img) in ret_summary:
                            img = img[0].transpose(2, 0, 1)
                            summary_writer.add_image(tag, img, global_counter)
                    except:
                        raise Exception('Prediction skipped due to an error!')

                    # Add sampled data loss
                    tot_val_loss += val_loss.item()
                    tot_val_times += 1

            batch_time = time.time()

        # Save the best model
        avg_val_loss = tot_val_loss / float(tot_val_times)
        is_best = avg_val_loss < min_loss
        min_loss = min(avg_val_loss, min_loss)
        save_checkpoint({'epoch': epoch + 1, 'global_counter': global_counter, 'state_dict': model.module.state_dict(), 'min_loss': min_loss},
                        is_best, args.saved_model.format(cfg.DETECT_OBJ))

        # Dump test data every 10 epochs
        if (epoch + 1) % args.val_epoch == 0:   # Time consuming
            # Create output folder
            os.makedirs(os.path.join(args.output_path, str(epoch + 1)), exist_ok = True)
            os.makedirs(os.path.join(args.output_path, str(epoch + 1), 'data'), exist_ok = True)
            os.makedirs(os.path.join(args.output_path, str(epoch + 1), 'log'), exist_ok=True)

            if args.vis:
                os.makedirs(os.path.join(args.output_path, str(epoch + 1), 'vis'), exist_ok = True)

            model.train(False)  # Validation mode

            with torch.no_grad():
                for (i, val_data) in enumerate(val_dataloader):

                    # Forward pass for validation and prediction
                    probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data)

                    front_images, bird_views, heatmaps = None, None, None
                    if args.vis:
                        tags, ret_box3d_scores, front_images, bird_views, heatmaps = \
                            model.module.predict(val_data, probs, deltas, summary = False, vis = True)
                    else:
                        tags, ret_box3d_scores = model.module.predict(val_data, probs, deltas, summary = False, vis = False)

                    # tags: (N)
                    # ret_box3d_scores: (N, N'); (class, x, y, z, h, w, l, rz, score)
                    for tag, score in zip(tags, ret_box3d_scores):
                        output_path = os.path.join(args.output_path, str(epoch + 1), 'data', tag + '.txt')
                        with open(output_path, 'w+') as f:
                            labels = box3d_to_label([score[:, 1:8]], [score[:, 0]], [score[:, -1]], coordinate = 'lidar')[0]
                            for line in labels:
                                f.write(line)
                            print('Write out {} objects to {}'.format(len(labels), tag))

                    # Dump visualizations
                    if args.vis:
                        for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                            front_img_path = os.path.join(args.output_path, str(epoch + 1), 'vis', tag + '_front.jpg')
                            bird_view_path = os.path.join(args.output_path, str(epoch + 1), 'vis', tag + '_bv.jpg')
                            heatmap_path = os.path.join(args.output_path, str(epoch + 1), 'vis', tag + '_heatmap.jpg')
                            cv2.imwrite(front_img_path, front_image)
                            cv2.imwrite(bird_view_path, bird_view)
                            cv2.imwrite(heatmap_path, heatmap)

            # Run evaluation code
            cmd_1 = './eval/KITTI/launch_test.sh'
            cmd_2 = os.path.join(args.output_path, str(epoch + 1))
            cmd_3 = os.path.join(args.output_path, str(epoch + 1), 'log')
            os.system(' '.join([cmd_1, cmd_2, cmd_3]))

        tot_epoch = epoch + 1

    print('Train done with total epoch:{}, iter:{}'.format(tot_epoch, global_counter))

    # Close TensorBoardX writer
    summary_writer.close()


def save_checkpoint(state, is_best, filename = 'to_be_determined.pth.tar'):
    torch.save(state, '%s/%s' % (save_model_dir, filename))
    if is_best:
        best_filename = filename.replace('.pth.tar', '_best.pth.tar')
        shutil.copyfile('%s/%s' % (save_model_dir, filename), '%s/%s' % (save_model_dir, best_filename))


if __name__ == '__main__':
    dataset_dir = cfg.DATA_DIR
    train_dir = os.path.join(cfg.DATA_DIR, 'training')
    val_dir = os.path.join(cfg.DATA_DIR, 'validation')
    log_dir = os.path.join('./log', args.tag)
    save_model_dir = os.path.join('./saved', args.tag)
    os.makedirs(log_dir, exist_ok = True)
    os.makedirs(save_model_dir, exist_ok = True)

    run()



