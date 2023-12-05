# polygon falan ai da dursun

import ultralytics
import cv2
import supervision as sv
import numpy as np


class Model:
    def __init__(self, path):
        self.model_1 = self.model_yukle(path)
        # initiate polygon zone
        self.polygon_1 = np.array(
            [
                [87, 76],  # [250, 300]
                [565, 46],  # [450, 300]
                [565, 132],  # [450, 450]
                [87, 167],  # [250, 450]
            ]
        )
        self.kontrol = False
        self.ct_1 = 0
        self.durum = 'TAMAM'

    def model_yukle(self, path):
        return ultralytics.YOLO(path)

    def model_tespit(self, frame):
        zone_1 = sv.PolygonZone(polygon=self.polygon_1, frame_resolution_wh=(640, 480))

        # 1. dikdörtgen
        # box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=2, text_scale=2)
        zone_annotator_1 = sv.PolygonZoneAnnotator(
            zone=zone_1,
            color=sv.Color.white(),
            thickness=1,
            text_thickness=1,
            text_scale=1,
        )

        # detect
        results_1 = self.model_1(frame, imgsz=640)[0]
        detections_1 = sv.Detections.from_ultralytics(results_1)
        alandakiler_1 = zone_1.trigger(detections=detections_1)
        sayi_1 = len(alandakiler_1[alandakiler_1 == True])

        if sayi_1 == 2 and self.kontrol == False:  # Makine polygona girdi
            self.ct_1 += 1
            self.kontrol = True
            self.durum = 'AYAK VAR'

        elif (sayi_1 == 1 and self.kontrol == True):
            self.durum = 'YANLIS'

        elif (
                sayi_1 == 0 and self.kontrol == True
        ):  # Makine 1. polygondan çıktı --> Makine 2. polygona girdi
            self.kontrol = False
            self.durum = 'BOS'
        else:
            pass

        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)

        labels_1 = [
            f"{self.model_1.names[class_id]} {confidence:0.65f} "
            for _, _, confidence, class_id, _ in detections_1
        ]

        frame_1 = box_annotator.annotate(
            scene=frame,
            detections=detections_1[alandakiler_1],
            labels=labels_1,
            skip_label=None,
        )

        frame_1 = zone_annotator_1.annotate(
            scene=frame_1
        )

        # üst köşelerdeki sayıları yazdırma
        cv2.putText(
            frame_1, str(sayi_1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            frame_1, str(self.kontrol), (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
        cv2.putText(
            frame_1, str(self.durum), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
