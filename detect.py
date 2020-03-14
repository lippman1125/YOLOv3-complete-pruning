import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from copy import deepcopy
from vgg import vgg
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

CLASSES = {0: "gun", 1: "thumbup", 2: "victory", 3: "negative", 4: "ok"}
def detect(save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    # hand_dir = [s for s in source.split("/") if s != ''][-1]
    hand_dir = Path(source).name
    if not os.path.exists(hand_dir):
        os.makedirs(hand_dir)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    if opt.gesture:
        # modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        # modelc.to(device).eval()
        checkpoint = torch.load(opt.gesture)
        modelc = vgg(dataset="handdata")
        modelc.load_state_dict(checkpoint['state_dict'])
        modelc.to(device).eval()
        trans = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])


    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det_exp = det.clone()
                w, h = det_exp[:, 2] - det_exp[:, 0], det_exp[:, 3] - det_exp[:, 1]
                max_side = torch.max(torch.stack((w, h), dim=1))
                x_c, y_c = (det_exp[:, 2] + det_exp[:, 0])/2, (det_exp[:, 3] + det_exp[:, 1])/2
                # det_exp[:, 0] = det_exp[:, 0] - w*0.1
                # det_exp[:, 2] = det_exp[:, 2] + w*0.1
                # det_exp[:, 1] = det_exp[:, 1] - h*0.1
                # det_exp[:, 3] = det_exp[:, 3] + h*0.1
                # rectangle expand to square (1.1times)
                if "thumbup_gesture" in p:
                    det_exp[:, 0] = x_c - max_side*1.4/2
                    det_exp[:, 2] = x_c + max_side*1.4/2
                    det_exp[:, 1] = y_c - max_side*1.8/2
                    det_exp[:, 3] = y_c + max_side*1.0/2
                else:
                    det_exp[:, 0] = x_c - max_side*1.2/2
                    det_exp[:, 2] = x_c + max_side*1.2/2
                    det_exp[:, 1] = y_c - max_side*1.2/2
                    det_exp[:, 3] = y_c + max_side*1.2/2
                clip_coords(det_exp[:,:4], im0.shape)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                im_hand = deepcopy(im0)
                if save_img:
                    idx = 0
                    for *xyxy, conf, _, cls in det_exp:
                        x1,y1,x2,y2 = [int(c) for c in xyxy]
                        hand = im_hand[y1:y2, x1:x2, :]
                        hand_path = os.path.join(hand_dir, os.path.splitext(Path(p).name)[0] + "_{}.jpg".format(idx))
                        cv2.imwrite(hand_path, hand)
                        idx += 1
                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        plot_one_box(xyxy, im0, label=None, color=colors[int(cls)])

                for *xyxy, conf, _, cls in det_exp:
                    # plot_one_box(xyxy, im0, label=None, color=(0,0,255))
                    x1, y1, x2, y2 = [int(c) for c in xyxy]
                    hand = deepcopy(im_hand[y1:y2, x1:x2, :])
                    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
                    hand = cv2.resize(hand, (32, 32))
                    hand_pil = Image.fromarray(hand)
                    data = torch.unsqueeze(trans(hand_pil), 0).to(device)
                    # print(data.shape)
                    logits = modelc(data)
                    # print(logits)
                    output = F.softmax(logits, dim=1).squeeze()
                    # print(output)
                    prob = output.argmax(dim=0)
                    print("gesture = {}".format(CLASSES[prob.cpu().item()]))
                    title = "gesture = {}".format(CLASSES[prob.cpu().item()])
                    plot_one_box(xyxy, im0, label=title, color=(0, 0, 255))


            # else:
            #     # if no hand detected, we will mark manually
            #     hand_path = os.path.join(hand_dir, Path(p).name)
            #     cv2.imwrite(hand_path, im0)

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                cv2.waitKey(30)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--gesture', default='', help='gesture recognition model')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
