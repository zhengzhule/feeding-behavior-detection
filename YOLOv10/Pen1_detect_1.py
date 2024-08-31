import cv2
import torch
from ultralytics import YOLOv10
import os
import argparse
from strong_sort.strong_sort import StrongSORT
from collections import deque
import numpy as np
from pathlib import Path
total_processed_frames = {"三分之一帧数": 0}
import glob
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
# 检测参数
parser.add_argument('--show-vid', action='store_true', default=False, help='display tracking video results')
parser.add_argument('--weights', default="4Unewbest.pt", type=str, help='weights path')
parser.add_argument('--source', default='/media/pengyingqi/6564-79C5/zanshi/yolov10-strongsort/4am/crop', type=str, help='video(.mp4)path')
parser.add_argument('--project', default=ROOT / 'runs' / 'track4am', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--vis', default=True, action='store_true', help='visualize image')
parser.add_argument('--conf_thre', type=float, default=0.5, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.5, help='iou_thre')
parser.add_argument('--vid-stride', type=int, default=3, help='video frame-rate stride')
parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--save-vid', action='store_true', default=True, help='save video tracking results')
# 跟踪参数
parser.add_argument('--track_model', default=r"./track_models/osnet_x0_25_msmt17.pt", type=str, help='track model')
parser.add_argument('--max_dist', type=float, default=0.2, help='max dist')
parser.add_argument('--max_iou_distance', type=float, default=0.7, help='max_iou_distance')
parser.add_argument('--max_age', type=int, default=3000000, help='max_age')
parser.add_argument('--n_init', type=int, default=5, help='n_init')
parser.add_argument('--nn_budget', type=int, default=100, help='nn_budget')
# 解析参数
opt = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.2, iou_threshold=0.5, line_thickness=2):
        self.device = device
        self.model = YOLOv10(weight_path).to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.line_thickness = line_thickness  # Save the line_thickness parameter

        self.tracker = StrongSORT(
            opt.track_model,
            self.device,
            max_dist=opt.max_dist,
            max_iou_distance=opt.max_iou_distance,
            max_age=opt.max_age,
            n_init=opt.n_init,
            nn_budget=opt.nn_budget,
        )

        self.trajectories = {}
        self.max_trajectory_length = 5

    @staticmethod
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def write_results(self, frame_idx, txt_path, online_targets):
        with open(txt_path, 'a') as f:
            for t in online_targets:
                bbox = t[:4]
                tid = t[4]
                cls = t[5]
                conf = t[6]
                bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                bbox_w = bbox_right - bbox_left
                bbox_h = bbox_bottom - bbox_top
                x1 = (bbox_left + bbox_right) / 2
                y1 = bbox_top
                # 4号圈早
                if x1 < 1700 and y1 > 222:
                    h = 1.97605748e-10 * x1 ** 4 - 7.59015002e-07 * x1 ** 3 + 9.05807475e-04 * x1 ** 2 - 3.37221616e-01*x1 ** 1 + 4.79487679e+02
                    # 4号圈晚
                    # if 370 < x1 < 2100 and y1 > 1000:
                    #     h =  5.70176957e-08 * x1 ** 3 -2.52976561e-04 * x1 ** 2  +3.40239992e-01 * x1 + 1.07687571e+03

                    # 6号圈早
                    #    if  < 2142 and y1 > 1015:
                    #     h =  6.43097509e-08  * x1 ** 3 -2.69096291e-04 * x1 ** 2 +3.34666588e-01  * x1 + 7.69840281e+02

                    # 6号圈晚
                    #    if  < 2142 and y1 > 1015:
                    #     h =  4.84819014e-08  * x1 ** 3 -2.25541915e-04  * x1 ** 2  +2.86953569e-01  * x1 + 1.09168688e+03

                    # 格式: frame_idx, id, bbox_left, bbox_top, bbox_w, bbox_h, y1, h, conf, cls, -1
                    f.write(
                        f"{total_processed_frames['三分之一帧数']} {frame_idx + 1} {int(tid)} {bbox_left:.2f} {bbox_top:.2f} {bbox_w:.2f} {bbox_h:.2f} {y1:.2f} {h:.2f} {conf:.2f} {int(cls)} -1\n")

    def detect_image(self, img_bgr, frame_idx, txt_path):
        image = img_bgr.copy()
        results = self.model(img_bgr, conf=self.conf_threshold, iou=self.iou_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
        confidences = results[0].boxes.conf
        class_preds = results[0].boxes.cls

        confidences_expanded = confidences.unsqueeze(1)
        class_preds_expanded = class_preds.unsqueeze(1)
        boxes_tensor = torch.from_numpy(boxes).to(class_preds_expanded.device)
        xywhs = xyxy2xywh(boxes_tensor)
        online_targets = self.tracker.update(xywhs.cpu(), confidences_expanded.cpu(), class_preds_expanded.cpu(), image)

        # 将结果写入文件
        self.write_results(frame_idx, txt_path, online_targets)

        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = t[4]
            cls = t[5]
            xmin, ymin, xmax, ymax = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
            class_pred = int(cls)
            color = self.get_color(class_pred + 4)
            center = (int(xmin + xmax) // 2, int(ymin + ymax) // 2)
            bbox_label = results[0].names[class_pred]

            x1 = (xmin + xmax) / 2
            y1 = ymin

            # 4号圈早
            if x1 < 1700 and y1 > 222:
                h = 1.97605748e-10 * x1 ** 4 - 7.59015002e-07 * x1 ** 3 + 9.05807475e-04 * x1 ** 2 - 3.37221616e-01 * x1 ** 1 + 4.79487679e+02

                # 4号圈晚
                # if 370 < x1 < 2100 and y1 > 1000:
                #     h =  5.70176957e-08 * x1 ** 3 -2.52976561e-04 * x1 ** 2  +3.40239992e-01 * x1 + 1.07687571e+03

                # 6号圈早
                #    if  < 2142 and y1 > 1015:
                #     h =  6.43097509e-08  * x1 ** 3 -2.69096291e-04 * x1 ** 2 +3.34666588e-01  * x1 + 7.69840281e+02

                # 6号圈晚
                #    if  < 2142 and y1 > 1015:
                #     h =  4.84819014e-08  * x1 ** 3 -2.25541915e-04  * x1 ** 2  +2.86953569e-01  * x1 + 1.09168688e+03

                if ymax > h:
                    act = 'Feeding'
                elif ymax < h:
                    act = 'Picking'
                else:
                    act = 'ing'

                bbox_label = f'{bbox_label} {int(tid)} {act}'

            cv2.rectangle(img_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, self.line_thickness)


            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=self.max_trajectory_length)
            self.trajectories[tid].appendleft(center)

            # 截断轨迹长度
            if len(self.trajectories[tid]) > self.max_trajectory_length:
                self.trajectories[tid] = self.trajectories[tid][:self.max_trajectory_length]

            for i in range(1, len(self.trajectories[tid])):
                if self.trajectories[tid][i - 1] is None or self.trajectories[tid][i] is None:
                    continue

                thickness = int(np.sqrt(64 / float(i + 1)))
                cv2.line(img_bgr, self.trajectories[tid][i - 1],
                         self.trajectories[tid][i], color,
                         thickness)

            # 显示类名和跟踪ID
            cv2.putText(img_bgr, bbox_label,
                        (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

        return img_bgr


# Example usage
if __name__ == '__main__':
    model = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre,
                     line_thickness=opt.line_thickness)


def process_video(video_file, opt):
    opt.source = video_file
    # Initialize the Detector with the line thickness argument
    model = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre,
                     line_thickness=opt.line_thickness)

    # 创建新的项目文件夹
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 寻找下一个可用的exp文件夹
    exp_number = 1
    while (save_dir / f'exp{exp_number}').exists():
        exp_number += 1
    save_dir = save_dir / f'exp{exp_number}'
    save_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(opt.source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # 创建输出视频文件路径
    video_path = save_dir / f"{Path(opt.source).stem}_out.mp4"
    outVideo = cv2.VideoWriter(str(video_path), fourcc, fps, size)

    # 创建txt文件路径
    txt_path = save_dir / f"{Path(opt.source).stem}_results.txt"

    # 创建或清空txt文件
    open(txt_path, 'w').close()

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # 使用 vid-stride 参数
        if frame_idx % opt.vid_stride == 0:
            img_vis = model.detect_image(frame, frame_idx, txt_path)
            outVideo.write(img_vis)
            # 在这里增加计数器
            total_processed_frames["三分之一帧数"] += 1

            # Check if --show-vid is set
            if opt.show_vid:
                # 创建可调整大小的窗口
                cv2.namedWindow('track', cv2.WINDOW_NORMAL)

                # 重新调整显示图像的大小
                img_vis = cv2.resize(img_vis, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

                # 显示图像
                cv2.imshow('track', img_vis)

                # 你可以使用 `cv2.resizeWindow` 来手动设置窗口大小
                cv2.resizeWindow('track', 1800, 580)  # 你可以将800x600替换为任何你想要的窗口大小

                # 等待用户按键（这里的 30 表示 30ms 后继续下一帧，按 'q' 退出）
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        frame_idx += 1

    capture.release()
    outVideo.release()

    # Destroy windows only if --show-vid is set
    if opt.show_vid:
        cv2.destroyAllWindows()

    print(f"Results saved to {save_dir}")

def main(opt):
    # 查找文件夹中的所有视频文件
    video_files = sorted(glob.glob(os.path.join(opt.source, '*.mp4')))

    # 处理每个视频文件
    for video_file in video_files:
        process_video(video_file, opt)

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)

