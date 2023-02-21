import os
import torch
import argparse
from mobilenet import *
from tqdm.auto import tqdm
import datetime
from torch.utils.data import DataLoader
from dataset import *
from loss import *
from torchvision import transforms


def generate_predictions(args):

    model = MobileSaliency()
    state_dict = torch.load(args.file_weight, map_location='cpu')
    model.load_state_dict(state_dict)  #

    model = model.cuda()
    model.eval()

    list_video_name = [str('%03d' % i) for i in range(1, 1008)]
    list_video_name = list_video_name[args.test_video_start-1:args.test_video_end-1]
    kl_all, cc_all, nss_all, sim_all, time_all = [], [], [], [], []
    for video in list_video_name:
        one_video_kl, one_video_cc, one_video_nss, one_video_sim = [], [], [], []
        os.makedirs(args.save_path+'/'+str(video))
        one_video_frmaes = len(os.listdir(os.path.join(args.path_data, video, 'voice_frames')))

        for img_idx in range(1, one_video_frmaes+1):
            img, _ = torch_transform(os.path.join(args.path_data, video, 'voice_frames', str('%03d.jpg'%img_idx)))
            img = img.unsqueeze(0)
            kl, cc, nss, sim, time = process(model, img, video, img_idx, args)
            one_video_kl.append(kl), one_video_cc.append(cc), one_video_nss.append(nss), one_video_sim.append(sim)
            time_all.append(time)

        ave_one_video_kl = avg_list(one_video_kl)
        ave_one_video_cc = avg_list(one_video_cc)
        ave_one_video_nss = avg_list(one_video_nss)
        ave_one_video_sim = avg_list(one_video_sim)
        print('Evaluate video {}: kl:{:.3f}, cc:{:.3f}, nss:{:.3f}, sim:{:.3f}'.format(video, ave_one_video_kl,
                                                                                       ave_one_video_cc, ave_one_video_nss,
                                                                                       ave_one_video_sim))
        kl_all.append(ave_one_video_kl)
        cc_all.append(ave_one_video_cc)
        nss_all.append(ave_one_video_nss)
        sim_all.append(ave_one_video_sim)

        if len(kl_all) % 10 == 0:
            print('[Average by now]: KL:{:.3f}, CC:{:.3f}, NSS:{:.3f}, SIM:{:.3f}, Time:{:.3f}'
                  .format(avg_list(kl_all), avg_list(cc_all), avg_list(nss_all), avg_list(sim_all), avg_list(time_all)))

    KL = avg_list(kl_all)
    CC = avg_list(cc_all)
    NSS = avg_list(nss_all)
    SIM = avg_list(sim_all)
    print('Average performance: kl:{:.3f}, cc:{:.3f}, nss:{:.3f}, sim:{:.3f}, average time per frame:{:.5f}'.format(KL, CC, NSS, SIM, avg_list(time_all)))


def torch_transform(path):
    img_transform = transforms.Compose([
            # transforms.Resize((768, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (76, 76), interpolation=cv2.INTER_AREA)
    sz = img.shape
    img = img_transform(img)
    return img, sz


def avg_list(list_a):
    sum_ = 0
    for i in list_a:
        sum_ += i
    return sum_/len(list_a)


def process(model, img, video, img_idx, args):
    with torch.no_grad():
        img = img.cuda()
        # start = datetime.datetime.now()
        starter.record()
        pre = model(img)
        # end = datetime.datetime.now()
        ender.record()
        torch.cuda.synchronize()
        time = starter.elapsed_time(ender)
        # time = (end - start).total_seconds()
        pre = pre.squeeze(0).squeeze(0).cpu().numpy()
        pre = cv2.resize(pre, (720, 1280))
        pre = torch.FloatTensor(pre).unsqueeze(0).cuda()
        # cal metrics
        gt_sal = cv2.imread(os.path.join(args.path_data, video, 'Saliency_maps', str('%03d.png' % img_idx)), 0)
        gt_sal = cv2.resize(gt_sal, (720, 1280))
        if np.max(gt_sal) > 1.0:
            gt_sal = gt_sal / 255.0

        gt_fix = cv2.imread(os.path.join(args.path_data, video, 'Fixation_maps', str('%03d.png' % img_idx)), 0)
        gt_fix = cv2.resize(gt_fix, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
        gt_fix = torch.FloatTensor(gt_fix)
        gt_fix = gt_fix.masked_fill(gt_fix != 0, float(1.0))

        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()  # torch.FloatTensor:to float32
        gt_fix = torch.FloatTensor(gt_fix).unsqueeze(0).cuda()

        kl, cc, nss = loss_func(pre, gt_sal, gt_fix, args)
        sim = similarity(pre, gt_sal)

        # generate predictions
        if 711 < int(video) < 712:
            pre = pre.cpu().data[0].numpy()  #
            pre = cv2.resize(pre, (720, 1280))
            pre = cv2.normalize(pre, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(os.path.join(args.save_path, video, str('%03d.png' % img_idx)), pre, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    return kl, cc, nss, sim, time


if __name__ == '__main__':
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    os.makedirs(os.path.join('test_and_predict', str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))))
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight', default="runs/2022-07-11-17:48:30/ali_saliency.pt", type=str)
    parser.add_argument('--test_video_start', default=701, type=int)
    parser.add_argument('--test_video_end', default=711, type=int)
    parser.add_argument('--path_data', default="/root/Ali_saliency/Data/Test/", type=str)
    parser.add_argument('--clip_size', default=16, type=int)
    parser.add_argument('--save_path', default='test_and_predict/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')), type=str)

    args = parser.parse_args()

    generate_predictions(args)
