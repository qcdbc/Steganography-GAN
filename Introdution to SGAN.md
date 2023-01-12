**SGAN consists of G, D, S**

D: classifies whether image is real or synthetic

S: steganalyser, determines if an image contains a concealed secret message

​		G compete against D and S simultaneously



denote $S(x)$ as the probability that $x$ has some hidden information 

$L = \alpha(E_{x\sim p_{data}(x)}[\log D(x)]+E_{z\sim p_{noise}(z)}[\log (1-D(G(z)))])+\\ \quad (1-\alpha)E_{z\sim p_{noise}}[\log S(Stego(G(z))) + \log(1-S(G(z)))] \longrightarrow \displaystyle{\min_G \max_D \max_S}$



*Optimizer using SGD: Stochastic mini-batch Gradient Descent*



SGAN model structure:  $C2D-BN-LR$

S, D has the simular structure: four $C2D-BN-LR$ --> FCL(1) --> Sigmoid 

G structure: FCL(8192) --> four $C2D-BN-LR$ with Fractional-Strided Convolution --> Tanh



using a independent steganalyser $S^*$, define a filter $F^{(0)}$ 

​		$S^*$ structure: 2D convolution with $F^{(0)}$ filter --> Conv2D --> Conv2D --> Max Pooling --> Conv2D --> Conv2D --> Max Pooling --> FCL(1024) --> FCL(1) --> Sigmoid 