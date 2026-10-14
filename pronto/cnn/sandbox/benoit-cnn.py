import torch
import torch.nn.functional as F
import numpy as np
import idx2numpy
from torchvision.transforms import v2


#On cherche à identifier les données provenant du dataset MNIST à l'aide d'un réseau de 
#neurones convolutionel. Il fonctionne de la manière suivante :
# La 1ère partie est l'entrainement de notre CNN:
#     On multiplie notre image par un filtre de convolution puis on fait un pooling (on ne
#     choisi que certaines valeures) de l'image convolutionée. On répète cette opération 
#     plusieurs fois puis on applatie l'image (matrice de valeures) obtenue et on la passe
#     dans un réseau de neurones complètement connectés. Le reseau de neurone renvoie une 
#     réponse et en fonction de l'écart au label attendu (loss), on modifie les valeures
#     de notre filtre pour minimiser la loss (backpropagation + descente de gradient).
#     On répète cette opération un certains nombre de fois (epochs) jusqu'à ce que le
#     réseau de neurones soit bien entrainé. On peut ensuite le tester. 
# La 2ème phase est le testing:
#     On envoie des images test dans le CNN est on évalue si oui ou non il est bien 
#     entrainé (taux de réussite superieur à un certain seuil)



#  Image en entrée (28*28) => premier filtre X1 (3*3 , 4) => pooling (7*7, 4) => deuxième filtre (3*3 , 4, 4)
#  => Deuxième pooling (6*6 , 4)



# === data === #
lr = 1e-1
batch_size = 1000
epochs = 1

file1 = 'Mnist_data/train-images-idx3-ubyte'
file2 = 'Mnist_data/train-labels-idx1-ubyte'
train_imgs = idx2numpy.convert_from_file(file1)    #Array numpy de taille 60000*28*28 (images)
train_lbs = idx2numpy.convert_from_file(file2)     #Array numpy de taille 60000 (labels)

file3 = 'Mnist_data/t10k-images-idx3-ubyte'
file4 = 'Mnist_data/t10k-labels-idx1-ubyte'
test_imgs = idx2numpy.convert_from_file(file3)    #Array numpy de taille 10000*28*28 (images)
test_lbs = idx2numpy.convert_from_file(file4)    #Array numpy de taille 10000 (labels)


# === convolution_filter === #

filter_length = 3   #c'est bien d'avoir une taille de filtre impair pour faire un padding équilibré
filter_hight = 3    #on prends un filtre carré car on travaille sur des images carrées
nb_filters = 4      #On effectue 2 étapes de convolution avec chacun 4 filtres 

filter_length2 = 3
filter_hight2 = 3
filter_depth2 = 4
nb_filters2 = 4 

X1 = torch.randn((nb_filters,filter_length, filter_hight), dtype=torch.float32)*0.1  # premier filtre de convultion , multiplication car apprends mieux avec des petits poids
X1.requires_grad_()
X2 = torch.randn((nb_filters2,filter_depth2,filter_length2,filter_hight2), dtype=torch.float32)*0.1 #deuxième filtre de convolution
X2.requires_grad_()

def padding_1(tab : torch.Tensor):
    '''
    fait un padding de taille 1 tout autour sur un tenseur torch à 2 dimensions
    renvoi un tenseur pytorch
    '''
    (n,m) = tab.shape
    assert(n!=0 and m!=0)
    return torch.cat([torch.zeros(n+2,1),torch.cat([torch.zeros(1,m),tab,torch.zeros(1,m)],dim=0),torch.zeros(n+2,1)],dim=1) #torch.cat concatène des tenseurs




def convolution(imgs : torch.Tensor , img_depth : int, ker : torch.Tensor)-> torch.Tensor:
    '''
    fait une convolution entre une image et un filtre. La fonction fait automatiquement un padding de 1.
    On considère un pas de 1 et un padding.
    ATTENTION, il faut absolument un filtre carré de taille impaire. 
    input:
        imgs (Torch.tensor.float32) :  liste d'images sur lesquelles on fait la convolution
        img_depth (int)            :  nombre d'images dans la liste img
        ker (Torch.tensor.float32) :  matrice de convolution  
    output:
        (torch.Tensor.float32) : convolution de img et ker (Sum img[i].ker[i])
    '''
    
    if img_depth == 1:
        n = len(imgs)    #on travaille avec des images carrées => pas besoin de la largeur.
        img_pad = padding_1(imgs)
        rows = []
        for i in range(n):
            cols =  []
            for j in range(n):
                val = torch.sum(ker*img_pad[i:i+len(ker),j:j+len(ker[0])])  #produit de convolution entre le filtre et une partie de l'image
                cols.append(val)
            rows.append(torch.stack(cols))  
        return torch.stack(rows)  # on fait un stack pour éviter de créer un nouveau tensor (qui empêcherait le backward)
    else:
        n = len(imgs[0])  # taille du côté de l'image
        img_pad = torch.stack([padding_1(imgs[d]) for d in range(img_depth)]) #on fait un padding sur les images
        rows = []
        for i in range(n):
            cols = []
            for j in range(n):
                val = torch.sum(ker*img_pad[:,i:i+len(ker[0]),j:j+len(ker[0][0])]) #pour indexer de cette manière, il faut un tenseur pytorch
                cols.append(val)
            rows.append(torch.stack(cols))
        return torch.stack(rows)


def mean_pooling(list_img : list[torch.Tensor] ,end_size : int )-> torch.Tensor:
    '''
    IL FAUT QUE end_size DIVISE LA TAILLE DE Chaque image (pour mnist end size = 2,4,7,14)
    Renvoie un pooling d'un Array carré de dimension 2 de type List[Torch.Tensor]
    '''
    l = len(list_img)
    for i in range(l):
        assert([len(list_img[i])%end_size == 0])
    
    out = []
    for img in list_img:
        n = len(list_img[0])
        p = n//end_size     #coté du carré qu'on moyenne 
        rows = []
        for i in range(end_size):
            cols = []
            for j in range(end_size):
                val = torch.mean(img[p*i:p*(i+1),p*j:p*(j+1)])
                cols.append(val)
            rows.append(torch.stack(cols))
        out.append(torch.stack(rows))
    return torch.stack(out)    # On renvoie Tenseur d'images qui ont subi un pooling


# === réseaux de neurones complet === #

N1 = torch.zeros(128)  
Nf = torch.zeros(10) 

W1 = torch.randn((196,128)) * 0.1     #si on met les poids à 0, le gradient ne va pas avancer 
W1.requires_grad_()
W2 = torch.randn((128,10)) * 0.1
W2.requires_grad_()




def pred(img : list):
    '''
    On fait passer une image applatie n*n dans le reseau de neuronnes et on renvoie les logits et probabilities
    '''
    M = img @ W1          # matrice ligne contenant la valeure des neuronnes de la couche intermédiaire
    M = torch.relu(M)   #On applique ReLU sur nos neuronnes

    M2 = M @ W2     # valeures des neuronnes finaux
    return M2, torch.softmax(M2, dim=0)  # logits et probabilités (softmax transforme le tenseur M2 en une liste de proba dont la somme vaut 1)

#On utilise un softmax car cela favorise la compétition entre les classes


# === boucle d'entrainement === #
def train(epochs,X1,X2,W1,W2):

    #convertir et normaliser l'image
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale = True)]) #normalise l'image et la met sous forme de tenseur
    transformed_img = transform(train_imgs).permute(1,0,2) # Pour avoir la bonne dimension [60000,28,28]
   
    # logging frequency (affiche toutes les `log_every` itérations)
    log_every = 100
    def _safe_grad_norm(tensor):
        return tensor.grad.norm().item() if (tensor is not None and tensor.grad is not None) else 0.0
    def _param_norm(tensor):
        return tensor.norm().item() if tensor is not None else 0.0
    
    #boucle d'entraînement
    
    for i in range(epochs):
        print("epoch "+str(i))
        #On mélange les images pour ne pas avoir toujours les mêmes
        perm = np.random.permutation(len(train_imgs))
        transformed_img = transformed_img[perm]
        shuffeled_train_lbs = train_lbs[perm]

        for k in range(batch_size):
            #On passe dans les deux couches de convolution puis de pooling            
            conv1 = [convolution(transformed_img[k],1,X1[u]) for u in range(0,4)]
            pool1 = mean_pooling(conv1,14)
                                        
            conv2 = [convolution(pool1,4,X2[u]) for u in range(0,4)]
            pool2 = mean_pooling(conv2,7)

            flat = pool2.flatten()   #vecteur ligne = version applatie de pool2 

            #On passe dans le reseau de neuronnes
            logits, p_barre = pred(flat)  # logits (sortie brut du CNN) + probabilités pour logging

            # Backpropagation avec une perte stable sur logits
            target = torch.tensor([shuffeled_train_lbs[k]], dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target)

            loss.backward()    # calcule les gradients de la loss
            torch.nn.utils.clip_grad_norm_([X1, X2, W1, W2], max_norm=5.0) #limite la taille des gradient à 5

            # capture des normes avant update
            w1_norm_before = W1.norm().item()
            w2_norm_before = W2.norm().item()
            x1_norm_before = X1.norm().item()
            x2_norm_before = X2.norm().item()

            # logs: afficher périodiquement la loss, les probabilités et les normes des gradients
            if (k % log_every) == 0:
                print(f"epoch {i} iter {k} loss={loss.item():.6f} grad_W1={_safe_grad_norm(W1):.6f} grad_W2={_safe_grad_norm(W2):.6f} grad_X1={_safe_grad_norm(X1):.6f} grad_X2={_safe_grad_norm(X2):.6f} | norm_W1={w1_norm_before:.6f} norm_W2={w2_norm_before:.6f}")

            with torch.no_grad():  # désactive autograd pour cette opération
                # mise à jour des paramètres
                X1 -= lr * X1.grad
                X2 -= lr * X2.grad
                W1 -= lr * W1.grad
                W2 -= lr * W2.grad

                # réinitialise le gradient 
                X1.grad.zero_()
                X2.grad.zero_()
                W1.grad.zero_()
                W2.grad.zero_()




# === boucle de test === #
def test(batch_size):

    #convertir et normaliser l'image
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale = True)])
    transformed_test_img = transform(test_imgs).permute(1,0,2)

    #boucle de test
    nb_valid = 0
    for k in range(batch_size):
            #On passe dans les deux couches de convolution puis de pooling            
            conv1 = [convolution(transformed_test_img[k],1,X1[u]) for u in range(0,4)]
            pool1 = mean_pooling(conv1,14)
                                        
            conv2 = [convolution(pool1,4,X2[u]) for u in range(0,4)]
            pool2 = mean_pooling(conv2,7)

            flat = pool2.flatten()   #vecteur ligne = version applatie de pool2 

            #On passe dans le reseau de neuronnes
            logits, _ = pred(flat)
            pred_class = torch.argmax(logits).item()

            if pred_class == int(test_lbs[k]):
                nb_valid += 1
            
    return nb_valid/batch_size



train(epochs,X1,X2,W1,W2)  #On train sur 1 epoch pour debug
print(test(100))  #on test sur 100 imgs


#Utilise NN module 
#forward + backpropagation

#descente de gradient stochastique (moyenne de gradient sur un batch) plus efficace
#Data_loader  (générateur)
