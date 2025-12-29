import cv2
import numpy as np
import supervision as sv
from pathlib import Path




def annotate(image_data: Path | bytes, xyxy: list[tuple[int, int, int, int]], labels: list[str]) -> np.ndarray:
    
    if isinstance(image_data, Path):
        image: np.ndarray = np.array(
                cv2.imread(str(image_data))
        )
    elif isinstance(image_data, bytes):
        image: np.ndarray = np.array(
            cv2.imdecode(
                np.frombuffer(image_data, np.uint8),
                cv2.IMREAD_COLOR
            )
        )
    else:
        raise TypeError(f"Invalid type, expecting Path or bytes, got: {type(image_data)}")


    unique_labels: list[str] = list(set(labels))
    class_id_vocab: dict[str, int] = { unique_labels[i]: i for i in range(len(unique_labels))}

    detections = sv.Detections(
        xyxy=np.array([ list(i) for i in xyxy ]),
        class_id=np.array([ class_id_vocab[label] for label in labels ]),
    )

    bounding_box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)


    annotated_image = bounding_box_annotator.annotate(
        scene=image, 
        detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )
    return annotated_image


def normalize_coordinates(
    coordinates: list[tuple[int, int, int, int]],
    img_size: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    
    result = []
    for coor in coordinates:
        x_1, y_1, x_2, y_2 = tuple((int(c/1000*p) for p, c in zip(img_size*2, coor)))

        if x_1 > x_2:
            x_1, x_2 = x_2, x_1

        if y_1 > y_2:
            y_1, y_2 = y_2, y_1

        result.append((x_1, y_1, x_2, y_2))
    return result




if __name__ == "__main__":

    coor = [
        (231, 7, 646, 305),
        (602, 361, 725, 465),
        (176, 422, 273, 537),
    ]
     
    labels = [
        "Head",
        "Left arm",
        "Right arm"
    ]

    abs_coors = normalize_coordinates(coor, (720, 720))
    print("abs_coor: ", abs_coors)

    img = annotate(Path("~/Desktop/gotou2.jpg").expanduser(), abs_coors, labels)
    sv.plot_image(img)
    print("Image type: ", type(img))



