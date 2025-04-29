# eecs182-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Homework 3 Solved](https://www.ankitcodinghub.com/product/eecs182-solved-15/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116543&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Homework 3  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1. Normalization Layers

Recall the pseudocode for a batchnorm layer (with learnable scale and shift) in a neural network:

(a) If our input data (1-dimensional) to batchnorm follows roughly the distribution on the left:

Figure 1: Gaussian with mean µ = 1.5, variance Figure 2: Blank grid for your answer σ2 =0.16

What does our data distribution look like after batch normalization with β = 3 and γ = 1 parameters? Draw your answer on the blank grid above, give a scale to the horizontal axis, and label β. You can assume that the batch-size is very large.

(Note: You do not have to give a scale to the vertical axis.)

(b) Say our input data (now 2-dimensional) to the batchnorm layer follows a Gaussian distribution. The mean and contours (level sets) of points that are 1 and 2 stdev away from the mean are shown below. On the same graph, draw what the mean, 1-SD, and 2-SD contours would look like after batchnorm without any shifting/scaling (i.e. β = 0 and γ = 1). You can assume that the batch-size is very large.

Figure 3: Draw your answer on the grid

2. Understanding Convolution as Finite Impulse Response Filter

For the discrete time signal, the output of linear time invariant system is defined as:

(1)

1.

Now let’s consider rectangular signal with the length of L (sometimes also called the “rect” for short, or, alternatively, the “boxcar” signal). This signal is defined as:

(

1 n = 0,1,2,…,L − 1 x(n) =

0 otherwise

Here’s an example plot for L = 7, with time indices shown from -2 to 8 (so some implicit zeros are shown): (a) The impulse response is define as:

Compute and plot the convolution of x(n) and h(n). For illustrative purposes, your plot should start at -6 and end at +12.

Figure 4: The rectangular signal with the length of 7

(b) Now let’s shift x(n) by N, i.e. x2(n) = x(n − N). Let’s put N = 5 Then, compute y2(n) = h(n) ∗ x2(n). Which property of the convolution can you find?

Now, let’s extend 1D to 2D. The example of 2D signal is the image. The operation of 2D convolution is defined as follows:

∞ ∞

y[m,n] = x[m,n] ∗ h[m,n] = X x[m − i,n − j] · h[i,j] = X x[i,j] · h[m − i,n − j]

i,j=−∞ i,j=−∞

(2)

, where x is input signal, h is FIR filter and y is the output signal.

(c) 2D matrices, x and h are given like below:

 

1 2 3 4 5

 6 7 8 9 10

 

x = 11 12 13 14 15 (3)

 

16 17 18 19 20

 

21 22 23 24 25

 

−1 −2 −1

h =  0 0 0  (4)

 

1 2 1

Then, evaluate y. Assume that there is no pad and stride is 1.

(d) Now let’s consider striding and padding. Evaluate y for following cases:

i. stride, pad = 1, 1 ii. stride, pad = 2, 1

3. Feature Dimensions of Convolutional Neural Network

In this problem, we compute output feature shape of convolutional layers and pooling layers, which are building blocks of CNN. Let’s assume that input feature shape is W × H × C, where W is the width, H is the height and C is the number of channels of input feature.

(a) A convolutional layer has 4 architectural hyperparameters: the filter size(K), the padding size (P), the stride step size (S) and the number of filters (F). How many weights and biases are in this convolutional layer? And what is the shape of output feature that this convolutional layer produces?

(b) A max pooling layer has 2 architectural hyperparameters: the stride step size(S) and the “filter size” (K). What is the output feature shape that this pooling layer produces?

(c) Let’s assume that we have the CNN model which consists of L successive convolutional layers and the filter size is K and the stride step size is 1 for every convolutional layer. Then what is the receptive field size of the last output?

(d) Consider a downsampling layer (e.g. pooling layer and strided convolution layer). In this problem, we investigate pros and cons of downsampling layer. This layer reduces the output feature resolution and this implies that the output features loose the certain amount of spatial information. Therefore when we design CNN, we usually increase the channel length to compensate this loss. For example, if we apply the max pooling layer with kernel size of 2 and stride size of 2, we increase the output feature size by a factor of 2. If we apply this max pooling layer, how much does the receptive field increases? Explain the advantage of decreasing the output feature resolution with the perspective of reducing the amount of computation.

(e) Let’s take a real example. We are going to describe a convolutional neural net using the following pieces:

• CONV3-F denotes a convolutional layer with F different filters, each of size 3×3×C, where C is the depth (i.e. number of channels) of the activations from the previous layer. Padding is 1, and stride is 1.

• POOL2 denotes a 2 × 2 max-pooling layer with stride 2 (pad 0)

• FLATTEN just turns whatever shape input tensor into a one-dimensional array with the same values in it.

• FC-K denotes a fully-connected layer with K output neurons.

Note: All CONV3-F and FC-K layers have biases as well as weights. Do not forget the biases when counting parameters.

Now, we are going to use this network to do inference on a single input. Fill in the missing entries in this table of the size of the activations at each layer, and the number of parameters at each layer. You can/should write your answer as a computation (e.g. 128 × 128 × 3) in the style of the already filled-in entries of the table.

Layer Number of Parameters Dimension of Activations

Input 0 28 × 28 × 1

CONV3-10 28 × 28 × 10

POOL2 0 14 × 14 × 10

CONV3-10 3 × 3 × 10 × 10 + 10

POOL2

FLATTEN 0 490

FC-3 3

(f) Consider a new architecture:

CONV2-3 → ReLU → CONV2-3 → ReLU → GAP (Global Average Pool) → FC-3

Each CONV2-3 layer has stride of 1 and padding of 1. Note that we use circular padding (i.e. wraparound) for this task. Instead of using zeros, circular padding makes it as though the virtual column before the first column is the last column and the virtual row before the first row is the last row — treating the image as though it was on a torus.

Here, the GAP layer is an average pooling layer that computes the per-channel means over the entire input image.

You are told the behavior for an input image with a horizontal edge, x1 and an image with a vertical edge, x2:

x x g g

Suppose we knew that the GAP output features when fed x1 and x2 are

Use what you know about the invariances/equivariances of convolutional nets to compute the gi corresponding to the following xi images.



0

0  0 

• x3 = 0  0  0



0 0

0

0

0

0

0

0 0

0

0

0

0

0

0 0

0

0

0

0

0

0 0

0

0

0

0

0

0 0

0

0

0

0

0

0 1

1

1

1

1

1

1 

0

0  0  0  0  0



0



0

0  0 

• x4 = 0  0  1



0 0

0

0

0

0

1

0 0

0

0

0

0

1

0 0

0

0

0

0

1

0 0

0

0

0

0

1

0 0

0

0

0

0

1

0 0

0

0

0

0

1

0 

0

0  0  0  0  1



0

4. Coding Question: Desigining 2D Filter

Convolutional layer, which is the most important building block of CNN, actively utilizes the concept of filters used in traditional image processing. Therefore, it is quite important to know and understand the types and operation of image filters.

Look at HandDesignFilters.ipynb. In this notebook, we will design two convolution filters by hand to understand the operation of convolution:

(a) Blurring filter.

(b) Edge detection filter.

For each type of filter, please include the image generated by the Jupyter Notebook in your submission to the written assignment. The image should be added to the PDF document that you will be submitting. Please do not submit the Jupyter Notebook to “Homework 3 (Code)”.

5. Coding Question: Batchnorm, Dropout and Convolutions

Attention: This coding task will take approximately 4 to 6 hours to complete, taking into account the time required for training the model. Please plan accordingly and start early to ensure adequate time for completion.

The assignment consists of three parts:

Implementing Batch Norm and Dropout. Open this url and follow the instructions in the notebook. You will implement the forward and backward propagation of dropout and batch normalization in NumPy. A GPU runtime is not required for this part.

Implement convolution and spatial batch norm. Open this url and follow the instructions in the notebook. You will implement the forward and backward propagation of convolutional layers and spatial dropout in NumPy. A GPU runtime is not required for this part.

Use deep learning framework. Open this url and follow the instructions in the notebook. For this part, you will need to switch to a GPU runtime (details can be found in the notebook). You will implement a convolutional neural network with convolution layers, batch normalization, and dropout using PyTorch and train it on a GPU. You also have the opportunity to improve or design your own neural network.

To submit your completed work, please upload a .zip file to the Gradescope assignment titled “Homework 3 (Code)”. The instructions for packaging everything into an archive can be found in the last cell of the notebook of the last part.

Please answer the following question in your written assignment submission:

(a) Draw the computational graph of training-time batch normalization in your written assignment. In input of the computational graph should be X,γ,β, the output of the computational graph should be Y, and the intermediate nodes are µ,σ2,Z.

(b) (Optional) Derive the closed-form back-propagation of a batch normalization layer (during training). Include the answer in your written assignment.

Specifically, given dy for every i,j, Please derive for every i,j as a function of dy,X,µ,σ2,ϵ,γ,β.

(c) Explain what you see in this experiment. What does it suggest about dropout?

(d) Briefly describe your neural network design and the procedure of hyperparameter tuning.

6. Homework Process and Study Group

We also want to understand what resources you find helpful and how much time homework is taking, so we can change things in the future if possible.

(a) What sources (if any) did you use as you worked through the homework?

(b) If you worked with someone on this homework, who did you work with?

List names and student ID’s. (In case of homework party, you can also just describe the group.)

(c) Roughly how many total hours did you work on this homework? Write it down here where you’ll need to remember it for the self-grade form.

Contributors:

• Saagar Sanghavi.

• Suhong Moon.

• Dominic Carrano.

• Babak Ayazifar.

• Sukrit Arora.

• Romil Bhardwaj.

• Fei-Fei Li.

• Sheng Shen.

• Jake Austin.

• Kevin Li.

• Anant Sahai.

• Linyuan Gong.
