

**Progress Report on the Segmentation and Classification of Weeds**

Student Engineer: Isaac Tan, N9960210

Supervisors: Jasmine Banks, Khamael Al-Dulaimi

II. LITERATURE REVIEW

***Abstract*— Weed management plays a crucial role in the**

Before creating the software system, it was important

that the task, the technology, and the current approaches

were fully understood. The purpose of this literature review

is to justify the need for the segmentation software system

proposed in this project. Additionally, it will provide an

understanding of what an artificial neural network is and

how it works. Finally, it will investigate current methods and

case studies in similar problem spaces. The literature

explored in this review will be contained to sources written

in relevant agricultural, computer science or engineering

journals or articles from January 2012 onwards. These

sources will be obtained through online databases such as

IEEE Xplore, Scopus and Inspec.

**outcome of harvests across the industry of agriculture.**

**Through advancements in software systems, more emphasis is**

**being placed on the optimisation of weed management and how**

**emerging technologies can be used to reduce labour, reduce**

**chemical contamination, and increase profitability of**

**agriculture. This report investigates the use of a convolutional**

**neural network in the task of identifying and segmenting weeds**

**in images of grass for the future task of automatically precision**

**spraying weeds. The neural network was built upon the U-Net**

**architecture and the results of this report aimed to compare the**

**accuracy and speed of different constraints on the architecture**

**and asses the overall viability for a segmentation system using**

**this architecture. Findings showed that system resulted in**

**accuracies up to 86.5% and taking at maximum 1.69 seconds to**

**segment a dataset of 135 images 256x256 pixels in size.**

The need for the software system proposed in this project

stems from the huge annual loss of crop to weeds. Studies in

both Australia and India showed billion-dollar losses per

year to weeds with Australia losing an estimated $5 billion

in 2018 (McLeod, 2018), and India losing $11 billion in

2020 (Talaviya, Shah, Patel, Yagnik, & Shah, 2020).

Additionally, current methods of controlling weeds are cost

ineffective, labour intensive and harmful to the environment.

Current methods include hand picking or spraying with

targeted herbicide, and broader spraying with a general

herbicide. The disadvantage of hand picking and spraying is

that it is especially labour intensive on larger farms. On the

other hand, using a general herbicide and mass spraying

farms is costly, less effective, and harmful to the

I. INTRODUCTION

One industry that will be ever vital to the sustainability of

the population is agriculture. Since the dawn of the industry,

farmers have investigated ways of optimising the industry so

that they can produce the maximum yield while minimising

the loss of yield due to factors like weeds, pests, and natural

disasters. With technology continuously evolving, people are

looking towards artificial neural networks (ANNs) and

machine learning to help overcome these issues and make

significant improvements in the agriculture industry.

This project will investigate the problem of weeds in environment (Hlaing & Khaing, 2014). Different weed

agriculture and aim to construct a software system that aids species require different herbicides and as such broader

in the alleviation of this problem. To do that, this report will spraying cannot eliminate all weeds and instead aim to

delve into the current literature of machine learning and eliminate the most prevalent species (Tao & Wei, 2022).

ANNs in the agriculture industry in relation to weeds. Additionally, failure to eliminate a weed with a herbicide

Finally, it will discuss an implementation of a CNN to creates herbicide resistant species as they build tolerance

segment weeds and grass, the CNN’s architecture, and the (Kamath, Balachandra, Vardhan, & Maheshwari, 2022).

results from its implementation. It will also discuss the risks Research in Australia has also shown that herbicides used to

of such a project and how its implementation is influenced treat weeds on sugarcane farms flow off into rivers and then

by

ethics

and

sustainability. into the Great Barrier Reef, harming one of the natural

wonders (Sugar Research Australia, 2022).

Automated targeted spraying aims to solve these issues

as weed species are precisely sprayed with the correct

herbicide and manual labour is replaced with an automated

system. These automated systems can be split into two major

tasks: identifying the weed species and applying herbicide.

This report focuses solely on the task of identifying the

weeds.

One technology that has shown promise in identifying

This report is in partial fulfilment of EGH400-1 unit assessment

requirements and **submitted on 12/06/2022.**

**isaac.tan@connect.edu.au**

weeds is the use of machine learning and artificial neural

networks. Artificial neural networks (ANNs) are algorithms

mimicking the biological structure of the human brain (Wen,

EGH400 Project Progress Report

Page 1 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

Yihui, Ying, & Jiaxu, 2022). They make decisions based

Before a software system was created for this project,

upon the output of a stream of connections between neurons case studies on similar projects were investigated. One study

from an input. These neurons are separated into layers and in the International Journal of Electrical and Computer

connected through weight, bias, and an activation function. Engineering investigated six CNN architectures for

Information is passed between layers with the weights classifying weeds amongst bean and beet crops. It

applied and assessed under the activation function. If the investigated VGG (16 & 19), GoogLeNet (Inception v3 &

output of the activation function meets the bias or threshold, v4), and MobileNet (v1 & v2) and found that Inception v4

then the data passed forward to the next layer. This process was the most accurate with 99.41% and 99.51% accuracy on

repeats until the output layer, at which the ANN arrives at a from scratch and pre-trained models respectively. It found

decision (Dave & Dutta, 2014). A simplified diagram of this that all models gave a 98%+ accuracy (Adil, Ahmed,

can be seen in Fig 2.1.

Mohammed, & Soufiane, 2022).

When classifying tobacco crop and weeds researchers

compared region and grid-based CNNs using R-CNN and

YOLOv5 respectively. The study published in Applied

Sciences found that the R-CNN resulted in a 98% accuracy

and the system using YOLOv5 yielded a 94% accuracy.

Additionally, it stated that integrating

a system that

automatically detected and precision sprayed weeds

decreased the use of herbicide by 52% (Alam, et al., 2022).

Another study published in the International Conference

on Artificial Intelligence and Smart Systems (ICAIS)

compared a series of CNNs to other decision-making

*Fig 2.1 Neural Network Diagram.* Adapted from (Shiruru, 2016)

Neural networks are made more advanced by adding algorithms with a varied segmentation and feature extraction

multiple layers, forming a deep neural network (DNN). A techniques to classify crops and weeds. It found that the

subclass of deep neural networks, convolutional neural CNNs were more effective than the other machine learning

networks (CNNs) are particularly useful when it comes to algorithms and that of the CNNs focusing on segmentation,

tasks of identifying patterns and similarities in images. SegNet and U-Net were the most accurate with results of

CNNs utilise a convolutional filter which when repeated 95%+ (Veeragandham & Santhi, 2021). It also detailed the

common challenges of classifying weed species using

automated systems such as the varying light conditions. The

study states that these problems can be overcome with

different image pre-processing techniques.

creates

a feature map of the input (Grace, Anitha,

Sivaramakrishnan, & Sivakumari, 2021). This automatically

created feature map removes the manual task of feature

extraction which further differentiates ANNs from DNNs.

Manually performing feature extraction and applying the

weights between neurons is called supervised learning and

creating convolutions that automatically adjust these weights

is termed unsupervised learning (Abiodun, et al., 2018). The

goal of both learning types is to arrive at predictions closer

to the expected outcome with each iteration of training. This

distance between predicted outcome and expected outcome

or ground truth is measured using a loss function and with

each step of training, the algorithm adjusts its feature

weights to reduce the loss. As this process is repeated the

algorithm eventually optimises the loss function and

Delving deeper into Segnet and U-Net, a study published

in IEEE International Conference on Signal, Information and

Data Processing (ICSIDP) investigated an application of

image segmentation for the blind and compares the results of

Segnet and U-Net. Their method found that for

segmentation, U-Net achieved the better result of 84.32%

accuracy against 70.54% with SegNet. Additionally, their

system was fast, taking less than 0.5 seconds to segment any

random image (Liu, Wang, & Zhao, 2019).

Additionally, Ronneberger et al. who first introduced U-

increases accuracy (Yu, Wang, Zou, & Wang, 2020). Net, states that for segmentation, U-Net achieves very good

Convolutions create linear functions of the output of the performance on varying images when they investigated its

previous layer and feed it forward, however for binary use in neuronal structures and electron-microscopic

classification like the classes of weed and not weed in this recordings. They found that it achieved 92.03% accuracy

project, a decision needs to be mapped to 0 or 1. In this case, taking less than one second to segment a 512x512 pixel

a sigmoid activation function is used as the last layer of the image. (Ronneberger, Fischer, & Brox, 2015). The appeal

network to fit the decision to these values (Li, Yang, Peng, of using U-Net was that it concatenates the features from

& Zhou, 2021). During the training phase, the CNN is encoding-decoding with higher resolution features from

constantly updating feature weights to minimise the value before max pooling. This allowed for contextual awareness

loss. However, if the model is trained for too long, it over when up-scaling.

optimises feature weights specifically for the training dataset

causing something called overfitting. This means that when

the model is evaluated on the testing dataset, the overall

accuracy is lowered as the model has been trained to

perform well specifically for the training dataset (Narasinga

Rao, Venkatesh Prasad, Sai Teja, Zindavali, & Phanindra

Reddy, 2018).

EGH400 Project Proposal: Scope of Work

Page 2 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

Creating accurate ground truth labels would have involved

manually and precisely outlining each weed object in all 540

III. METHOD

The general method of image segmentation and images in the dataset.

classification follows five steps: image acquisition, image

The segmentation model was built in Python using the

pre-processing, image segmentation, feature extraction and

classification (Vasavi, Punitha, & Rao, 2022). However, for

this project, a deep neural network was used thus removing

the need for manual feature extraction. As such, the steps

covered in this section of the report include acquisition, pre-

processing, segmentation, and classification.

TensorFlow library and Keras API. Following findings in

the literature review, it was decided that the neural network

be built based upon the U-Net architecture. This was so that

it took features at a higher resolution and applied them with

deeper convolutions. The architecture took an input frame

and applied two 3x3 convolutions followed by a 2x2 max

The images used for this project were acquired as part of pooling operation. This process was repeated for a total of 4

a dataset provided by project supervisor, Khamael Al- times with the number of feature channels of the

Dulaimi. Typically, images in a dataset are of different sizes convolutions doubling after each max pooling. An up-

and are randomly cropped and rescaled in pre-processing to sampling filter of 2x2 was then applied and concatenated

a determined smaller size so that all images are of equal with the feature map of the same size from before the max

shapes. This was not the case with this dataset as all images pooling operation. This was then followed by two more 3x3

were the same size. The images in this dataset did however convolutions and the process was repeated for four up-

require random separation into training and testing datasets samples. Each filter application used a ReLU activation

with a split of 75% and 25% respectively. The training function. The final filter was a 1x1 convolution with a

dataset was then split further to allow 10% of it to be used as sigmoid activation function to determine the binary output of

validation. To decrease the volume of data and thus decrease weed or not weed. A diagram of this architecture can be seen

training and testing time, images were downscaled. The below in Fig 3.2 where the grey boxes are the feature maps,

effect of this is discussed later in this report. In the literature and the white boxes are the copied and concatenated feature

review it was discovered that one of the biggest obstacles in maps from before the max pools.

real-world application of automated classification is varying

light conditions. This is often alleviated with image pre-

processing with varied light conditions. However, since the

scope of this project is contained to the images in the

dataset, which are all taken under similar lighting conditions,

this step was purposefully ignored here.

The dataset provided for this project also included

markup files for each image. These markup files contained

bounding box information for where in the image a weed

*Fig 3.2 U-Net Architecture* Adapted from (Ronneberger, Fischer, &

was located. This was helpful when constructing image

labels for the training of the neural network. A Python

program was created that parsed all the annotation files,

located bounding box information, and created a mask image

with the information. An example output of this Python

program can be seen below in Fig 3.1 next to its

corresponding dataset image where red indicates weeds and

green indicates background.

Brox, 2015)

The 405 images from the training dataset were used to

train and validate the neural network on this architecture.

The model was then compiled with the Adam optimiser.

Since the task was one of binary segmentation, a loss

function of binary cross entropy was used when compiling.

The model was initialised for training with an arbitrary

number of 15 epochs. When fitting the model however, a

call back function was created to combat the overfitting

problem discussed in the literature review. This call back

function monitored the validation loss at the end of each

epoch and if the validation loss had not improved for more

than 5 epochs then training would be stopped, and the

previous feature weights would be restored.

Following training, the model was then evaluated with

the testing dataset of 135 images. The results of this are

discussed later in section VI. To visualise these results, the

output feature maps for each image were converted back to

masks.

*Fig 3.1 Image from dataset (left), output mask from Python program*

*(right)*

IV. RISKS

The mask was used to create a one hot encoded map for each

image which was used as a ground truth label. This was

Given this project focused solely on the software of a

because one hot maps are easily manipulated and rescaled. weed segmentation system, there were no physical risks

While this was not the most accurate method of creating involved. There were risks in terms of deliverable

ground truth labels, it was the most time-effective method. dependencies as well as software dependencies. The

EGH400 Project Proposal: Scope of Work

Page 3 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

software system was developed in the Python programming

language with the TensorFlow library and the Keras API to

interface. This meant that the software was dependent on the

libraries importing correctly and the functions in the library

working as expected. Python was used as it was a language

that had been used by the student engineer previously and

they were comfortable with. TensorFlow and Keras were

chosen as they were well documented and had a large online

support community of others that share their troubleshooting

tips for issues that arose. This largely mitigated this risk.

The overall completion of the software system had a few

dependencies on the completion of smaller software tasks.

These tasks were given estimated completion times to

maintain an appropriate completion trajectory and assess the

overall state of the project. Additionally, weekly, or

fortnightly meetings were conducted between the student

engineer and the project supervisors to check in and allow

for questions to be asked or feedback to be provided.

VI. RESULTS & DELIVERABLES

The model described in section III was developed and

trained with the training dataset of 405 images. This model

resulted in a total of 1,962,642 parameters, all of which were

trainable. Table 6.1 shows the training summary of the

model as it trains.

*Table 6.1 Training summary of CNN with input of 64x64*

Epoch

Loss

Accuracy Validation Validation

Loss

Accuracy

0.8038

0.7790

0.8297

0.8341

0.8365

0.8403

0.8337

0.8404

0.8430

1

2

3

4

5

6

7

8

9

0.1374

0.0962

0.0881

0.0771

0.0646

0.0580

0.0449

0.0373

0.0274

0.7543

0.8529

0.8636

0.8803

0.9013

0.9120

0.9319

0.9443

0.9596

0.1267

0.1354

0.1320

0.1249

0.1354

0.1455

0.1942

0.2701

0.2746

V. ETHICS AND SUSTAINABILITY

Throughout this project, the four cores of ethics in

engineering were consistently considered. The in-depth

literature review meant that decisions made for this project

were well-informed and made based on adequate

knowledge. The literature review and development of the

software system enabled continued learning as there was no

single source that answered all questions.

The model was then assessed under the criteria of

Consistent meetings were held between the student accuracy and latency to assess the potential viability of

engineer and the project supervisors. This allowed for implementing this model with live cameras as the input in

effective and honest communication with the stakeholders of real time. It was previously stated that the input image size

the project and gave opportunity for support for the student could be scaled down to reduce computation and improve

engineer.

speed. As such, the model was tested with three different

input sizes to compare accuracy and speed. Accuracy was

measured by counting the number of times the predicted

class matched the ground truth label and then dividing that

by the total number of comparisons (TensorFlow, 2021).

The latency of the system was calculated by deducting the

time at the start of the predictions from the time at the end of

the predictions. The testing and results in this section were

generated solely on the testing dataset of 135 images.

Furthermore, the training and testing of the CNN was done

on a laptop with no GPU and an 11th Gen Intel Core i7-

1165G7 CPU running at 2.80GHz. Table 6.2 below shows

the results for each input image scaling.

The effect on sustainability of creating the software

system described in this report was also considered prior to

its development. Using a software system to identify weed

species and precision spraying them holds up all three pillars

of engineering sustainability. The theoretical system would

reduce the cost of weed management and increase the yield

of crops thus increasing profit. By applying precision

spraying, the volume of herbicide that flows into the ground

and then greater environment is also reduced, benefitting the

environment. Finally, the physical labour of precision

spraying large farms is reduced, benefiting the social aspect

of sustainability. It could be said that implementation of an

automated weeding system could potentially make farmers

redundant and lead to a loss of jobs. However, the operation

of these theorised systems would still require human upkeep,

just less physically straining compared to hand spraying

large farms.

*Table 6.2 Accuracy and Latency of CNN model assessed*

*with varying input image scaling*

Input image size

(pixel length &

width)

Accuracy (%)

Time (ms)

The project was conducted with a clear conscience

knowing that the overall impact would be positive and solve

real-world issues.

64

80.56

83.09

86.54

304.255

796.657

1690.865

128

256

EGH400 Project Proposal: Scope of Work

Page 4 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

The results show that as images were downscaled less, i.e.

The deliverables of this project altered slightly between

the input image size was larger, the software took longer to the project proposal and this progress report. The initial

make predictions with it taking up to 1.69 seconds to make scope of this project included the creation of a software

predictions on the whole testing dataset when images were system that would use area thresholding and canny edge

256x256 pixels. This is still an acceptable latency on the detection on the dataset to create label masks for each image.

criteria of implementing into a live camera feed as 1.69 These masks would be used as ground truth labels for

seconds to make predictions on 135 images is 12.54ms per training and testing the segmentation system. When this was

image. The results show that the larger input image led to proposed, the existence of the annotation bounding box

higher accuracy with the 256x256 input resulting in the information was not known. Once this was discovered it

highest accuracy of 86.54%. The notion that increasing the didn’t make sense to spend a large amount of time

input image size leads to higher accuracy was confirmed developing software that created masks when a much

when comparing the results of the 64x64 images and simpler program could be created that parsed the annotations

128x128 images which resulted in accuracies of 80.56% and and created these masks instead. An updated version of the

83.09%. Overall, the system achieved a high level of deliverables is included in the appendix.

accuracy on all image sizes tested.

All other deliverables outlined in the initial scope have

To visualise the predictions made by the CNN, the output been delivered up to current date. This includes the literature

can be overlayed over the original input image. This can be review, the conceptual design, ground truth labels, the

seen in Fig 6.1 below where the left shows the original software for the segmentation system, the testing, and

image, and the right shows the prediction overlayed with red results.

assigned to weed class and green assigned to not weed class.

VII. LIMITATION AND FUTURE WORKS

Analysing the results of the segmentation system

highlighted opportunities for improvement in the future. As

discussed above, the pixelwise accuracy of the ground truth

labels was not great. This led to false positives of weed

detection. In future, these ground truth labels could be

created such that they are much more accurate and follow

the actual shape of the weed as opposed to a rectangle. This

would also help alleviate the issue of some weeds being

partially obscured by grass in the image and the

*Fig 6.1 CNN segmentation predictions of two images. Original images*

*(left) & predictions overlayed with original image (right)*

segmentation software predicting that the overlapping grass

was weed.

When observing the predictions, it was seen that the system

often over predicts the boundary of the weeds and

misclassifies some of the surrounding grass as a weed.

Additionally, there were some images of grass with no

weeds that had small predictions of weeds in the image. This

is likely due to the ground truth labels used to train the

system. Because the ground truth labels used bounding box

rectangles to identify regions in the image where there was a

weed and weeds are not rectangles, there were often

bounding boxes that included regions of grass. As such, the

system was trained to overestimate. These ground truth

labels often being larger than the weed also effects the

calculation of accuracy obtained above. This is because the

calculation of accuracy compares the prediction to the

ground truth label, which if the ground truth is not accurate,

means the accuracy assessment isn’t a precise assessment of

accuracy.

It was stated in some articles in the literature review that

data augmentation of the input images can increase the

accuracy of a neural network. In future, this can be applied

to the dataset used in this project to increase the volume of

training data. This can be done through altering brightness,

contrast, saturation, dimensions, and rotations of images.

Because the task of segmentation was binary in that there

were only two classes, weed and not weed, the model was

compiled with the binary cross entropy loss function.

However, there are other loss functions that show promise in

improving accuracy in cases where there is class imbalance.

One such loss function that could be investigated is the focal

loss function which down-weights easy examples and targets

hard negatives.

As mentioned in the literature review and method,

overfitting of a training model is a common issue when

training CNNs. The method of stopping the training early if

validation loss doesn’t improve after a set number of epochs

was applied to this model. This resulted in the model being

trained for 9 epochs as opposed to 15 which was stated in

the initialisation of the model fitting. This can be seen in

table 6.1 above.

VIII. CONCLUSION

To summarise, this project aimed create a software system

capable of segmenting weeds from grass for potential use in

a system of automatically detecting and precision spraying

weeds. This was done due to the huge cost in terms of

labour, monetary and impact on the environment that weeds

pose in the agriculture industry. The entire project was

EGH400 Project Proposal: Scope of Work

Page 5 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

conducted with engineering ethics and sustainability in

mind.

APPENDIX

The software system was created using convolutional

neural networks to extract features from images and make

decisions based upon unsupervised learning. The neural

network was built upon the U-Net architecture as it was able

to maintain contextual awareness after encoding by merging

feature maps with those from before max pooling.

*Table 1 Updated project deliverables*

\#

1

Focus

Deliverable

Dependant

Milestone

1/5/2022

Literature

Review

Literature

Review Report

Results from testing the neural network show good

accuracy and latency with all constraints of the system

showing 80% accuracy. The highest accuracy came when

the network was trained with the highest resolution input

images of 256x256, resulting in 86.54% accuracy.

Predictions did take longer when inputs were larger,

however even at the highest resolution, predictions only took

1.69 seconds for 135 test images. Applying image rescaling

to reduce image size did also decrease the time take to make

predictions on the data, it did however come at the cost of

accuracy. The software system was able to perform

predictions at a high speed. As such, using a live camera

feed as an input data and using the software system in real-

time is viable given the results obtained in this report.

2

Conceptual

design

Outline of

methodology

for

1

1/5/2022

segmentation

and

classification

3

4

Ground truth

label software

Python program 1,2

Python program 1,2,3

22/5/2022

5/6/2022

Semantic

segmentation

software

5

6

Test &

validate

Accuracy,

latency,

4

visualisation of

predictions

Overall, the U-Net architecture was successful, though

some improvements could be made to how ground truth

labels are created, or the loss function used to account for

the false positives observed in the results. Furthermore,

image augmentation can be investigated for future works to

increase the size and variability of the training dataset.

Interim

Report

Progress on

software

4,5

14/6/2022

29/6/2022

system, results,

changes to

scope

7

Interim

Presentation

Oral

6

presentation of

project so far

and interim

results

The project successfully delivered the segmentation

software system as well as other deliverables outlined in the

project proposal. While there were some changes to the

deliverables scope, the changes were made following

meetings with the project supervisors and still resulted in the

same expected outcome.

8

9

Stakeholder

consultation

Project

21/6/2022

trajectory

analysis

Classification

software

Image

8

16/10/2022

segmentation

and

classification

software

ACKNOWLEDGMENTS

This project was made possible by the supervision and

mentorship of Khamael Al-Dulaimi and Jasmine Banks.

Additionally, the dataset of 540 images used for the training

and testing of the neural network was provided with

corresponding annotation markups by Khamael Al-Dulaimi.

EGH400 Project Proposal: Scope of Work

Page 6 of 7

12 June 2022





**Faculty of Engineering**

**EGH400 Project Progress Report**

*Research And Innovative Ideas In Education*, 27-

\30.

IX. REFERENCES

Sugar Research Australia. (2022). *Reducing herbicide usage*

*on sugarcane farms in reef catchment areas with*

*precise robotic weed control*. Retrieved 4 15, 2022,

from https://sugarresearch.com.au/current-research-

projects/

Talaviya, T., Shah, D., Patel, N., Yagnik, H., & Shah, M.

(2020). Implementation of artificial intelligence in

agriculture for optimisation of irrigation and

application of pesticides and herbicides. *Artificial*

*Intelligence in Agriculture, 4*, 58-73.

Tao, T., & Wei, X. (2022). A hybrid CNN–SVM classifer

for weed recognition in winter rape field. *Plant*

*Methods, 18*(29).

TensorFlow. (2021, 4 22). *TensorFlow Core v2.9.1*.

Retrieved from TensorFlow:

Abiodun, O. I., Jantan, A., Omolara, A. E., Dada, K. V.,

Mohamed, N. A., & Arshad, H. (2018). State-of-

the-art in artificial neural network applications: A

survey. *Heliyon, 4*(11).

Adil, T., Ahmed, G., Mohammed, B., & Soufiane, B. (2022).

Weeds detection efficiency through different

convolutional neural networks technology.

*International Journal of Electrical and Computer*

*Engineering, 12*(1), 1048-1055.

Alam, M. S., Alam, M., Tufail, M., Khan, M. U., Güneş, A.,

Salah, B., . . . Khan, M. T. (2022). TobSet: A New

Tobacco Crop and Weeds Image Dataset and Its

Utilization for Vision-Based Spraying by

Agricultural Robots. *Applied Sciences, 12*(3).

Dave, V. S., & Dutta, K. (2014). Neural network based

models for software effort estimation: a review.

*Artificial Intelligence Review*, 295-307.

Grace, R. K., Anitha, J., Sivaramakrishnan, R., &

Sivakumari, M. S. (2021). Crop and Weed

Classification Using Deep Learning. *Turkish*

*Journal of Computer and Mathematics Education*

*(TURCOMAT)*, 935-938.

Hlaing, S. H., & Khaing, A. S. (2014). Weed and Crop

Segmentation and CLassifcation Using Area

Thresholding. *International Journal of Research in*

*Engineering and Technology, 3*, 375-382.

Kamath, R., Balachandra, M., Vardhan, A., & Maheshwari,

U. (2022). Classification of paddy crop and weeds

using semantic segmentaion. *Cogent Engineering,*

*9*, 3-15.

https://www.tensorflow.org/api\_docs/python/tf/kera

s/metrics/Accuracy

Vasavi, P., Punitha, A., & Rao, T. V. (2022). Crop leaf

disease detection and classification using machine

learning and deep learning algorithms by visual

symptoms: a review. *International Journal of*

*Electrical and Computer Engineering (IJECE)*,

2079-2086.

Veeragandham, S., & Santhi, H. (2021). A Detailed Review

on Challenges and Imperatives of Various CNN

Algorithms in Weed Detection. *International*

*Conference on Artificial Intelligence and Smart*

*Systems (ICAIS)*, 1068-1073.

Wen, J., Yihui, R., Ying, L., & Jiaxu, L. (2022). Artificial

Neural Networks and Deep Learning Techniques

Applied to Radar Target Detection: A Review.

*Electronics (Switzerland), 11*(1).

Li, Z., Yang, W., Peng, S., & Zhou, J. (2021). A Survey of

Convolutional Neural Networks: Analysis,

Applications, and Prospects. *IEEE Transactions on*

*Neural Networks and Learning Systems*, 1-21.

Liu, L., Wang, Y., & Zhao, H. (2019). An Image

Segmentation Method for the blind sidewalks

recognition by using the convolutional neural

network U-net. *IEEE International Conference on*

*Signal, Information and Data Processing (ICSIDP)*.

McLeod, R. (2018). *Annual Costs of Weeds in Australia.*

Canberra: Centre for Invasive Species Solutions.

Narasinga Rao, M., Venkatesh Prasad, V., Sai Teja, P.,

Zindavali, M., & Phanindra Reddy, O. (2018). A

survey on prevention of overfitting in convolution

neural networks using machine learning techniques.

*International Journal of Engineering and*

Yu, R., Wang, Y., Zou, Z., & Wang, L. (2020).

Convolutional neural networks with refined loss

functions for the real-time crash risk analysis.

*Transportation Research Part C*.

*Technology (UAE)*, 177-180.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net:

Convolutional networks for biomedical image

segmentation. *Lecture Notes in Computer Science*

*(including subseries Lecture Notes in Artificial*

*Intelligence and Lecture Notes in Bioinformatics)*,

234-241.

Shiruru, K. (2016). An Introduction to Artificial Neural

Network. *International Journal Of Advance*

EGH400 Project Proposal: Scope of Work

Page 7 of 7

12 June 2022

