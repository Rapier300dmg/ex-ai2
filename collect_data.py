import cv2, os, argparse
from utils.video_utils import extract_mouth

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect(label, out_dir, max_samples):
    save_dir = os.path.join(out_dir, label)
    mkdir(save_dir)
    cap = cv2.VideoCapture(0)
    count = len(os.listdir(save_dir))
    while True:
        ret, frame = cap.read()
        if not ret: break
        roi, vis = extract_mouth(frame)
        cv2.putText(vis, f"{label}: {count}/{max_samples}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Collect", vis)
        if roi is not None and count < max_samples:
            cv2.imwrite(os.path.join(save_dir, f"{count:04d}.jpg"), roi)
            count += 1
        if cv2.waitKey(1)&0xFF==27 or count>=max_samples:
            break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--out_dir", default="data/raw")
    p.add_argument("--max_samples", type=int, default=200)
    args=p.parse_args()
    collect(args.label, args.out_dir, args.max_samples)
