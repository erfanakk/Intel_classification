import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import fiftyone as fo
import fiftyone.brain as fob
import torch
from torch.utils.data import DataLoader , WeightedRandomSampler
# from utils import get_mean_std , plot_data
import os



#load data set  imagefolder
#resize 224 , 224
#find std and mean for norm
#***data augmentation*** 
#find image for each classs if data is **imbalance** do something 
#fifty one 


#https://www.learnpytorch.io/04_pytorch_custom_datasets/   *******

#https://datagy.io/pytorch-dataloader/
#https://blog.paperspace.com/dataloaders-abstractions-pytorch/


'''
my_transforms = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
        transforms.RandomCrop((32, 32)),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(
            degrees=45
        ),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(
            p=0.05
        ),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Note: these values aren't optimal
    ]
)

transforms.Normalize(mean=torch.tensor([0.4930, 0.4840, 0.4803]), std=torch.tensor([0.2575, 0.2549, 0.2561]))


'''




def creat_dataset(path_train, path_test, BATCH_SIZE=8, img_size=64, imbalance=False):
    
    transform_train = transforms.Compose([ 
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([img_size, img_size]),  
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(root=path_train, transform=transform_train)
    test_dataset = ImageFolder(root=path_test, transform=transform_test)

    if imbalance:
        classes, counts = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)
        weights = 1.0 / counts.float()
        sample_weights = weights[torch.tensor(train_dataset.targets).squeeze().long()]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights), 
            replacement=True
        )

        trainset = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        testset = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        return trainset,  testset
    
    trainset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testset = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return trainset,  testset




def show_dataset_fiftyone(data_dir , img_size=224,transform_=None):
    class ImageFolderWithPaths(ImageFolder):


        def __getitem__(self, index):
            
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            
            path = self.imgs[index][0]
            tuple_with_path = (original_tuple + (path,))
            
            return tuple_with_path

    transform = transforms.Compose([transforms.Resize((img_size,img_size)), 
                                transforms.ToTensor()])

    if transform_:
        transform = transform_

    data_dir = data_dir
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    dataloader = DataLoader(dataset)

    fo_dataset = fo.Dataset("intel_dataset")

    for inputs, labels, paths in dataloader:
        for input, label, path in zip(inputs, labels, paths):
       
            target = dataset.classes[label]
            sample = fo.Sample(filepath=path)

            sample["ground_truth"] = fo.Classification(label=target)

            fo_dataset.add_sample(sample)
    
    # fob.compute_similarity(
    # fo_dataset, model="clip-vit-base32-torch", brain_key="clip"
    # )


    session = fo.launch_app(fo_dataset , port=5151)
    session.wait()







if '__main__' == __name__:
    def test():
        # trainset ,valset ,testset = creat_dataset(path_train= r"cmc\train" , path_test=r"cmc\test" , path_val=r"cmc\val" , imbalance=True )
        # imgesT , _ = next(iter(train))
        # imges , _ = next(iter(test))
        # imgesv , _ = next(iter(val))

        
        # print(class_names)
        
        # print(f'we have {len(train)} batch and {imgesT.shape[0] * (len(train))} images in each batch in train dataset')
        # print(f'we have {len(test)} batch and {imges.shape[0] * (len(test))} images in each batch in test dataset')
        # print(f'we have {len(val)} batch and {imgesv.shape[0] * (len(val))} images in each batch in val dataset')

        # print(f'shape of image c= {imgesT.shape[1]} h= {imgesT.shape[2]} w= {imgesT.shape[3]}')
        # mean, std = get_mean_std(train)

        # print(f'this is mean {mean} and this is std {std}')
        # this is mean tensor([0.4930, 0.4845, 0.4803]) and this is std tensor([0.2575, 0.2549, 0.2561])
        # Transforms = transforms.Compose([ 
        #                                     transforms.Resize([224,224]),
        #                                     transforms.RandomRotation(degrees=35),
        #                                     transforms.RandomGrayscale(p=0.2),
        #                                     transforms.RandomHorizontalFlip(p=0.5),
                                        
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(torch.tensor([0.4930, 0.4840, 0.4803]), std=torch.tensor([0.2575, 0.2549, 0.2561]))
        #                                     ])


        # show_dataset_fiftyone(r"cmc\train" , transform_=Transforms
        # trainset  ,testset = creat_dataset(path_train= r"data\seg_train" , path_test=r"data\seg_test"  , imbalance=False )
        # imgesT , _ = next(iter(trainset))
        # print(f'we have {len(trainset)} batch and {imgesT.shape[0] * (len(trainset))} images in each batch in train dataset')

        show_dataset_fiftyone(r"data\seg_train")



    test()
