import numpy as np
import matplotlib.pyplot as plt

# data
Z_DIM = 1
X_DIM = 1000

# network
G_NEURONS = 10
D_NEURONS = 10

# training
G_LR = 0.001
D_LR = 0.001
ITERATIONS = 50000
GRADIENT_CLIP = 0.2
WEIGHT_CLIP = 0.25

# Generate real sine examples
def get_real_samples(random=True):
    if random:
        x0 = np.random.uniform(0, 1)
        freq = np.random.uniform(1.2, 1.5)
        mult = np.random.uniform(0.5, 0.8)
    else:
        x0 = 0
        freq = 0.2
        mult = 1
    signal = [mult * np.sin(x0+freq*i)for i in range(X_DIM)]
    return np.array(signal)

# activaton functions
def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return ReLU(x)

def LeakyReLU(x, k=0.2):
    return np.where(x>=0, x, x*0.2)

def dLeakyReLU(x, k=0.2):
    return np.where(x >= 0, 1., k)
    
def Tanh(x):
    return np.tanh(x)

def dTanh(x):
    return 1. - Tanh(x) ** 2

def Sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

# initialize network weights
def weight_initializer(in_channels, out_channels):
    scale = np.sqrt(2 / (in_channels + out_channels))
    return np.random.uniform(-scale, scale, (in_channels, out_channels))

# define loss function (forward and backward)
class LossFunc(object):
    def __init__(self) -> None:
        self.logit = None
        self.label = None
    def forward(self, logit, label):
        if logit[0, 0] < 1e-7:
            logit[0, 0] = 1e-7
        if 1. - logit[0, 0] < 1e-7:
            logit[0, 0] = 1. - 1e-7
        self.logit = logit
        self.label = label
        return - (label * np.log(logit) + (1-label) * np.log(1 - logit))
    def backward(self):
        return (1-self.label) / (1-self.logit) - self.label / self.logit
    
# define generative network
class Generator(object):
    def __init__(self) -> None:
        self.z = None
        self.w1 = weight_initializer(Z_DIM, G_NEURONS)
        self.b1 = weight_initializer(1, G_NEURONS)
        self.x1 = None
        self.w2 = weight_initializer(G_NEURONS, G_NEURONS)
        self.b2 = weight_initializer(1, G_NEURONS)
        self.x2 = None
        self.w3 = weight_initializer(G_NEURONS, X_DIM)
        self.b3 = weight_initializer(1, X_DIM)
        self.x3 = None
        self.x = None
    def forward(self, inputs):
        self.z = inputs.reshape(1, Z_DIM)
        self.x1 = np.matmul(self.z, self.w1) + self.b1
        self.x1 = ReLU(self.x1)
        self.x2 = np.matmul(self.x1, self.w2) + self.b2
        self.x2 = ReLU(self.x2)
        self.x3 = np.matmul(self.x2, self.w3) + self.b3
        self.x = Tanh(self.x3)
        return self.x
    def backward(self, outputs):
        # Derivative with respect to output
        delta = outputs 
        delta *= dTanh(self.x)
        # derivative with respect to w3 & b3
        d_w3 = np.matmul(np.transpose(self.x2), delta)
        d_b3 = delta.copy()
        # pass the gradients to layer 2
        delta = np.matmul(delta, np.transpose(self.w3))
        # update w3 & b3
        ## clip the gradient if it was too big
        if (np.linalg.norm(d_w3) > GRADIENT_CLIP):
            d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
        ## apply gradient to paramters
        self.w3 -= G_LR * d_w3
        ## make sure it's not too big
        self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
        ## similar for b3
        self.b3 -= G_LR * d_b3
        self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        # derivative of loss with expect to x2
        delta *= dReLU(self.x2)
        # deratives with respect to w2 & b2
        d_w2 = np.matmul(np.transpose(self.x1), delta)
        d_b2 = delta.copy()
        # pass the gradients to layer 1
        delta = np.matmul(delta, np.transpose(self.w2))
        # update w2 & b2
        if (np.linalg.norm(d_w2) > GRADIENT_CLIP):
            d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
        self.w2 -= D_LR * d_w2
        self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
        self.b2 -= D_LR * d_b2
        self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        # derivatives of loss with respect to x1
        delta *= dReLU(self.x1)
        # derivatives with respect to w1 & b1
        d_w1 = np.matmul(np.transpose(self.z), delta)
        d_b1 = delta.copy()
        # update w1 & b1
        if (np.linalg.norm(d_w1) > GRADIENT_CLIP):
            d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
        self.w1 -= D_LR * d_w1
        self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
        self.b1 -= D_LR * d_b1
        self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))
     
        
class Discriminator(object):
    def __init__(self) -> None:
        self.x = None
        self.w1 = weight_initializer(X_DIM, D_NEURONS)
        self.b1 = weight_initializer(1, D_NEURONS)
        self.y1 = None
        self.w2 = weight_initializer(D_NEURONS, D_NEURONS)
        self.b2 = weight_initializer(1, D_NEURONS)
        self.y2 = None
        self.w3 = weight_initializer(D_NEURONS, 1)
        self.b3 = weight_initializer(1, 1)
        self.y3 = None
        self.y = None
    def forward(self, inputs):
        self.x = inputs.reshape(1, X_DIM)
        self.y1 = np.matmul(self.x, self.w1) + self.b1
        self.y1 = LeakyReLU(self.y1)
        self.y2 = np.matmul(self.y1, self.w2) + self.b2
        self.y2 = LeakyReLU(self.y2)
        self.y3 = np.matmul(self.y2, self.w3) + self.b3
        self.y = Sigmoid(self.y3)
        return self.y
    def backward(self, outputs, apply_grads=True):
        delta = outputs
        delta *= dSigmoid(self.y)
        # calculate derivatives of w3 & b3 
        d_w3 = np.matmul(np.transpose(self.y2), delta)
        d_b3 = delta.copy()
        # pass the gradient to layer 3
        delta = np.matmul(delta, np.transpose(self.w3)) 
        # update parameters w3 & b3
        if apply_grads:
            # Update w3
            if np.linalg.norm(d_w3) > GRADIENT_CLIP:
                d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
            self.w3 += D_LR * d_w3
            self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.w3))
            # Update b3
            self.b3 += D_LR * d_b3
            self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.b3))
        # calculate gradient in respect tp y2
        delta *= dLeakyReLU(self.y2)
        # calculate deratives
        d_w2 = np.matmul(np.transpose(self.y1), delta)
        d_b2 = delta.copy()
        # pass the gradient to layer 2
        delta = np.matmul(delta, np.transpose(self.w2))
        # update parameters
        if apply_grads:
            # Update w2
            if np.linalg.norm(d_w2) > GRADIENT_CLIP:
                d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
            self.w2 += D_LR * d_w2
            self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.w2))
            # Update b2
            self.b2 += D_LR * d_b2
            self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.b2))
        # calculadte the gradient of loss to y1
        delta *= dLeakyReLU(self.y1)
        # deratives with respect to w1 & b1
        d_w1 = np.matmul(np.transpose(self.x), delta)
        d_b1 = delta.copy()
        # pass the gradient to Layer 1
        delta = np.matmul(delta, np.transpose(self.w1))
        # derative with respect to x 
        if apply_grads:
            if np.linalg.norm(d_w1) > GRADIENT_CLIP:
                d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
            self.w1 += D_LR * d_w1
            self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.w1))
            # Update b1
            self.b1 += D_LR * d_b1
            self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP,
                self.b1))
        return delta
        
G = Generator()
D = Discriminator()
criterion = LossFunc()

real_label = 1
fake_label = 0

for itr in range(ITERATIONS):
    # Update D with fake data
    x_real = get_real_samples()
    y_real = D.forward(x_real)
    loss_D_r = criterion.forward(y_real, real_label)
    d_loss_D = criterion.backward()
    D.backward(d_loss_D)
    # Update D with fake data
    z_noise = np.random.randn(Z_DIM)
    x_fake = G.forward(z_noise)
    y_fake = D.forward(x_fake)
    loss_D_f = criterion.forward(y_fake, fake_label)
    d_loss_D = criterion.backward()
    D.backward(d_loss_D)
    # UPdate G with fake data
    y_fake_r = D.forward(x_fake)
    loss_G = criterion.forward(y_fake_r, real_label)
    d_loss_G = D.backward(loss_G, apply_grads=False)
    G.backward(d_loss_G)
    loss_D = loss_D_r + loss_D_f
    if itr % 100 == 0:
        print('{} {} {}'.format(loss_D_r.item((0, 0)), loss_D_f.item((0, 0)), loss_G.item((0, 0))))
    
    
    
x_axis = np.linspace(0, 3, X_DIM)
for i in range(50):
    z_noise = np.random.randn(Z_DIM)
    x_fake = G.forward(z_noise)
    plt.plot(x_axis, x_fake.reshape(X_DIM))
plt.ylim((-1, 1))
plt.show()