import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

def add_mnist_patch(img, mnist_img, size=28, scale=False):
    '''
    Add MNIST image to a given image at a random position.

    Could possible mix images together instead of having average intensity:
    lambda = 0.5
    lambdainv = (1-lambda)
    img[0, i, j] = (lamda * img[0, y+i, x+j] + lambdainv * avg_r) / 2
    img[0, i, j] = (lamda * img[0, y+i, x+j] + lambdainv * avg_g) / 2
    img[0, i, j] = (lamda * img[0, y+i, x+j] + lambdainv * avg_b) / 2

    If you want to just have the MNIST digit with the black background:
    img[0, y:y+h, x:x+w] = mnist_img
    img[1, y:y+h, x:x+w] = mnist_img
    img[2, y:y+h, x:x+w] = mnist_img
    '''
    h, w = size, size
    
    if scale:
        # Scale the image and thereby the size with some stochastic constant
        # to make the task more interesting
        a = max(np.random.rand(), 0.2)  # max 80% smaller than 28x28

        # Only square images for now
        h = int(size*a)
        w = int(size*a)
        mnist_img = transforms.Resize(size=(h, w))(
            torch.tensor(mnist_img, dtype=torch.float32)
        )
   
    img = img.numpy()
    img_size = img.shape[1]
    y, x = np.random.randint(0, img_size-w, (2,))
    
    # Average intestity in CIFAR10
    avg_r, avg_g, avg_b = 0.49062946, 0.48558968, 0.45077488
    
    for i in range(h):
        for j in range(w):
            if (
                int(mnist_img[0, i, j]) == 0 and
                int(mnist_img[0, i, j] == 0) and
                int(mnist_img[0, i, j] == 0)
            ):
                pass  # do not insert the black from the MNIST image
            else:
                img[0, y+i, x+j] = avg_r
                img[1, y+i, x+j] = avg_g
                img[2, y+i, x+j] = avg_b
    
    bbox = np.array([x, y, x+w, y+h])
    
    return torch.from_numpy(img), torch.from_numpy(bbox)

def add_patch_and_return(cifar_img, mnist_img, id, scale):
    x_mnist, y_mnist = mnist_img
    mixed_img, bbox = add_mnist_patch(cifar_img[0], x_mnist, scale=scale)
    bbox = bbox.type(torch.float32)
    x_train = mixed_img
    y_train = [bbox, y_mnist, cifar_img[1], id]
    return x_train, y_train

def make_dataset(skew=0.5, seed=None, scale=False, root='./data'):
    np.random.seed(seed)
    
    # Load MNIST images
    mnist_data = torchvision.datasets.MNIST(
        root=root,
        train=True,                         
        transform=transforms.ToTensor(), 
        download=True
    )
    
    # Pick out digits 0--4
    mnist_img_subset = []
    for i in range(len(mnist_data)):
        if(mnist_data[i][1] <= 4):
            mnist_img_subset.append(mnist_data[i])
    
    # Take a subset of MNIST digits to mix with the CIFAR images
    L = len(mnist_img_subset)
    random_indices = np.random.choice(np.arange(L), size=L, replace=False, p=None) 
    
    # Take half for train and half for test
    train_subset = random_indices[:int(L/2)]
    test_subset = random_indices[int(L/2):] 
    
    # Sort samples into arrays which we will use to make the label shift
    sorted_by_label = np.ones((5, int(L/2))) * -1
    for idx, x in enumerate(train_subset):
        sorted_by_label[mnist_img_subset[x][1]][idx] = x
        
    zero_array = [int(item) for item in sorted_by_label[0, :] if item != -1]
    one_array = [int(item) for item in sorted_by_label[1, :] if item != -1]
    two_array = [int(item) for item in sorted_by_label[2, :] if item != -1]
    three_array = [int(item) for item in sorted_by_label[3, :] if item != -1]
    four_array = [int(item) for item in sorted_by_label[4, :] if item != -1]
    
    print('Amount of label 0:', len(zero_array))
    print('Amount of label 1:', len(one_array))
    print('Amount of label 2:', len(two_array))
    print('Amount of label 3:', len(three_array))
    print('Amount of label 4:', len(four_array))
    
    # Skew is the amount that should be mixed with the background,
    # e.g., planes with 0's and so on
    N0 = int(np.round(skew * len(zero_array)))
    N1 = int(np.round(skew * len(one_array)))
    N2 = int(np.round(skew * len(two_array)))
    N3 = int(np.round(skew * len(three_array)))
    N4 = int(np.round(skew * len(four_array)))
 
    # TODO: Is this a bad thing? Should we pick randomly here instead?
    zeromix = zero_array[0:N0]
    onemix = one_array[0:N1]
    twomix = two_array[0:N2]
    threemix = three_array[0:N3]
    fourmix = four_array[0:N4]
    
    # The rest of the data is mixed uniformly with other backgrounds
    restmix = zero_array[N0:]
    restmix.extend(one_array[N1:])
    restmix.extend(two_array[N2:])
    restmix.extend(three_array[N3:])
    restmix.extend(four_array[N4:])
    
    print('Amount of label 0 to mix with background 0:', len(zeromix))
    print('Amount of label 1 to mix with background 1:', len(onemix))
    print('Amount of label 2 to mix with background 2:', len(twomix))
    print('Amount of label 3 to mix with background 3:', len(threemix))
    print('Amount of label 4 to mix with background 4:', len(fourmix))
    print('Datapoints to mix uniformly:', len(restmix))

    # Load CIFAR images
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Resize(128)]
    )
    cifar_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True, 
        transform=cifar_transform
    )
    
    # Sort out 0--4 as source backgrounds, 5--9 as target
    # TODO: Should we do a random shuffle over CIFAR?
    cifar_source, cifar_target = [], []
    for i in range(len(cifar_dataset)):
        if(cifar_dataset[i][1] <= 4):
            cifar_source.append(cifar_dataset[i])
        else: 
            cifar_target.append(cifar_dataset[i])
    
    # Source data
    x_train, y_train = [], []
    for entry in cifar_source:
        if entry[1] == 0 and len(zeromix) >= 1:
            x_add, y_add = add_patch_and_return(
                entry,
                mnist_img_subset[zeromix[0]],
                zeromix[0],
                scale
            )
            x_train.append(x_add)
            y_train.append(y_add)
            zeromix = zeromix[1:]
        elif entry[1] == 1 and len(onemix) >= 1:
            x_add, y_add = add_patch_and_return(
                entry,
                mnist_img_subset[onemix[0]],
                onemix[0],
                scale
            )
            x_train.append(x_add)
            y_train.append(y_add)
            onemix = onemix[1:]
        elif entry[1] == 2 and len(twomix) >= 1:
            x_add, y_add = add_patch_and_return(
                entry,
                mnist_img_subset[twomix[0]],
                twomix[0],
                scale
            )
            x_train.append(x_add)
            y_train.append(y_add)
            twomix = twomix[1:]
        elif entry[1] == 3 and len(threemix) >= 1:
            x_add, y_add = add_patch_and_return(
                entry,
                mnist_img_subset[threemix[0]],
                threemix[0],
                scale
            )
            x_train.append(x_add)
            y_train.append(y_add)
            threemix = threemix[1:]
        elif entry[1] == 4 and len(fourmix) >= 1:
            x_add, y_add = add_patch_and_return(
                entry,
                mnist_img_subset[fourmix[0]],
                fourmix[0],
                scale
            )
            x_train.append(x_add)
            y_train.append(y_add)
            fourmix = fourmix[1:]
        # We mix uniformly if we already have skewed sufficiently
        elif len(restmix) >= 1:
            x_mnist, y_mnist = mnist_img_subset[restmix[0]]
            if entry[1] == y_mnist:
                continue  # jump over the same backgrounds so we control skew
            mixed_img, bbox = add_mnist_patch(entry[0], x_mnist, scale=scale)
            bbox = bbox.type(torch.float32)
            x_train.append(mixed_img)
            y_train.append([bbox, y_mnist,entry[1], restmix[0]])
            restmix = restmix[1:]
        if (len(zeromix) + len(onemix) + len(twomix) + len(threemix) + len(fourmix) + len(restmix)) >= 1:
            continue
        else:
            break
    
    print('Should be 15298:', len(x_train))

    # Target data
    x_test, y_test = [], []
    for idx, entry in enumerate(cifar_target):
        if idx == int(L/2):
            break
        x_mnist, y_mnist = mnist_img_subset[test_subset[idx]]
        mixed_img, bbox = add_mnist_patch(entry[0], x_mnist, scale=scale)
        bbox = bbox.type(torch.float32)
        x_test.append(mixed_img)
        y_test.append([bbox, y_mnist, entry[1], test_subset[idx]])

    print('Should be 15298:', len(x_test))
    
    return x_train, y_train, x_test, y_test
