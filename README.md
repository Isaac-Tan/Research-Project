# Engineering Research Project: Segmentation and Classifcation of Weeds
 
## Abstract

The ability to manage invasive weed species is vital to the outcome of crop harvests and the agriculture industry. Advancing software systems are creating solutions that help this industry optimise weed management so that it is less labour intensive, chemically less harmful to the environment, and more profitable. This report investigates the use of such software systems, convolutional neural networks (CNNs), for the task of automatically segmenting and detecting the Rumex obtusifolius weed species amongst grass. The motivation for this is that the software system will be used in the future to automatically detect and precision spray these weeds. The project followed two main goals, semantically segmenting images of the Rumex weed, and then classifying with localisation (object detection). The semantic segmentation model was built upon the U-Net architecture and the object detection model was built with the Mask R-CNN architecture. The results of this report compared evaluation metrics of each system under different constraints and parameters, to ultimately assess the viability of using CNNs for weed detection. Findings showed that the semantic segmentation system achieved accuracies and IoU scores of 85% and 73.87% respectively while operating at 21 fps. Results from the object detection system revealed a mAP of 52.28% and an F1 score of 54.23% while taking 846 ms per prediction on a 256 x 256-pixel image.
