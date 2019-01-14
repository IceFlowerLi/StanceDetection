import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# Save the information of the best accuracy.
class BestResult:
    def __init__(self):
        self.best_dev_accuracy = -1
        self.best_accuracy = -1
        self.best_epoch = 1
        self.best_test = False
        self.accuracy = -1


def train(train_iter, dev_iter, test_iter,
          model, args):

    if args.device != -1:
        model.cuda()

    optimizer = None
    if args.Adam:
        print("Use Adam optimizer to train model...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    best_accuracy = BestResult()
    for epoch in range(1, args.epochs + 1):
        print("\n@@ %d epoch, all %d epochs @@" % (epoch, args.epochs))

        for steps, batch in enumerate(train_iter):
            target, tweet, attitude = batch.Target, batch.Tweet, batch.Attitude

            # Transport the batch data to GPU.
            if args.device != -1:
                target = target.cuda()
                tweet = tweet.cuda()
                attitude = attitude.cuda()

            optimizer.zero_grad()
            logit = model(target, tweet)
            loss = F.cross_entropy(logit, attitude)
            loss.backward()

            # utils.clip_grad_norm(model.parameters(), max_norm=10)

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(attitude.size()).data == attitude.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                print(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             train_size,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            '''
            if steps % args.test_interval == 0:
                print("\nDev  Accuracy: ", end="")
                eval(dev_iter, model, args, best_accuracy, epoch, test=False)
                print("Test Accuracy: ", end="")
                eval(test_iter, model, args, best_accuracy, epoch, test=True)

            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model.state_dict(), save_path)
                if os.path.isfile(save_path) and args.rm_model is True:
                    os.remove(save_path)
            '''

    return epoch


def eval(data_iter, model, args, best_accuracy, epoch, test=False):

    model.eval()
    corrects, avg_loss = 0, 0

    for batch in data_iter:
        target, tweet, attitude = batch.Target, batch.Tweet, batch.Attitude

        # Transport the batch data to GPU.
        if args.device != -1:
            target = target.cuda()
            tweet = tweet.cuda()
            attitude = attitude.cuda()

        logit = model(target, tweet)
        loss = F.cross_entropy(logit, attitude, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss, accuracy, corrects, size))

    if test is False:
        if accuracy >= best_accuracy.best_dev_accuracy:
            best_accuracy.best_dev_accuracy = accuracy
            best_accuracy.best_epoch = epoch
            best_accuracy.best_test = True
    if test is True and best_accuracy.best_test is True:
        best_accuracy.accuracy = accuracy

    if test is True:
        print("The Current Best Dev Accuracy: {:.4f}, and Test Accuracy is :{:.4f}, locate on {} epoch.\n".format(
            best_accuracy.best_dev_accuracy, best_accuracy.accuracy, best_accuracy.best_epoch))
        best_accuracy.best_test = False