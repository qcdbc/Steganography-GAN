**SGAN consists of G, D, S**

D: classifies whether image is real or synthetic

S: steganalyser, determines if an image contains a concealed secret message

G compete against D and S simultaneously



denote $S(x)$ as the probability that $x$ has some hidden information 

$L = \alpha(E_{x\sim p_{data}(x)}[\log D(x)]+E_{z\sim p_{noise}(z)}[\log (1-D(G(z)))])+$  
$\quad (1-\alpha)E_{z\sim p_{noise}}[\log S(Stego(G(z))) + \log(1-S(G(z)))] \longrightarrow \displaystyle{\min_G \max_D \max_S}$



*Optimizer using SGD: Stochastic mini-batch Gradient Descent*



SGAN model structure:  $C2D-BN-LR$

S, D has the simular structure: four $C2D-BN-LR$ --> FCL(1) --> Sigmoid 

G structure: FCL(8192) --> four $C2D-BN-LR$ with Fractional-Strided Convolution --> Tanh



using a independent steganalyser $S^*$, define a filter $F^{(0)}$ 

$S^*$ structure: 2D convolution with $F^{(0)}$ filter --> Conv2D --> Conv2D --> Max Pooling --> Conv2D --> Conv2D --> Max Pooling --> FCL(1024) --> FCL(1) --> Sigmoid 

$200,000$ data from Dataset Celebrities (Ziwei Liu & Tang)  
$190,000$ pics for training and $10,000$ for testing  
and we embedding some information in both train and test dataset  
*in total*  
there is $380,000$ for train and $20,000$ for test

as for embeding algorithm, we use $\pm 1$ embedding algorithm, also known as LSB matching algriothm  
with payload size equal to $.4$ bits per pixel for one out of three channels  



#### Our approach  

my approach to SGAN is using iWGAN(improved Wasserstein GAN) instead of DCGAN to optimize net, hence, some losses are re-defined as follows:  

##### loss_G: 

$-\alpha(E_{z\sim p_{noise}(z)}D(G(x))) - (1-\alpha)(E_{z\sim p_{noise}(z)}(S(Stego(G(z))))$  

##### loss_D:  

$-E_{x\sim p_{data}(x)}D(x) + E_{z\sim p_{noise}(z)}D(G(z))$   

##### loss_S:  

$-E_{z\sim p_{noise}(z)}S(Stego(G(z))) + E_{z\sim p_{noise}(z)}S(G(z))$  