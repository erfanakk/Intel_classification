import torch
import matplotlib.pyplot as plt
import pandas as pd



def plot_data(dataloader):
    torch.manual_seed(42)
    class_names = dataloader.dataset.classes

    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(dataloader.dataset), size=[1]).item()
        img, label = dataloader.dataset[random_idx]
        img = torch.permute(img , (1,2,0))        
        fig.add_subplot(rows, cols, i)
        #, vmax=1, vmin=0
        plt.imshow(img  )
        plt.title(class_names[label])
        plt.axis(False);
    
    plt.show()



def get_mean_std(dataLoader):
    mean = 0
    std = 0 
    n_samples = 0


    for images, _ in dataLoader:
        images = images.view(images.shape[0] , images.shape[1] , -1)
        std += images.std(2).sum(0)
        mean += images.mean(2).sum(0)
        n_samples += images.shape[0]

    mean /= n_samples 
    std /= n_samples 
    

    return mean, std





def save_model(model, opt , epoch , checkpointPATH):
    checkpoint = {
        'epoch': epoch ,
        'model_state' : model.state_dict(),
        'opt_state' : model.state_dict()
    }

    torch.save(checkpoint , checkpointPATH)
    print("the model saved")


def load_model(model, opt , checkpointPATH):
    checkpoint = torch.load(checkpointPATH)

    model.load_state_dict(checkpoint['model_state'])
    # opt.load_state_dict(checkpoint['opt_state'])
    epoch = checkpoint['epoch']

    
    print("the model loaded")
    return epoch


def accurancy (model , loader , device):
    model.eval()

    n_correct = 0
    n_samlpes = 0

    with torch.no_grad():
        for x , y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            _ , prediction = torch.max(pred , dim=1)

            n_samlpes += x.shape[0]

            n_correct += (prediction == y).sum()
        

        print(f"n samples see {n_samlpes} , accu = {float(n_correct)/float(n_samlpes) *100}")




def return_accuracy_loss(loader, model , loss_fn , device):

    loss = 0
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss += loss_fn(scores , y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    loss /= len(loader)
    model.train()
    return (float(num_correct)/float(num_samples)*100 , loss)



def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            # Normalize to range [0, 1]
            min_val = torch.min(param)
            max_val = torch.max(param)
            param = (param - min_val) / (max_val - min_val)


            for i in range(param.shape[0]):
                plt.figure(figsize=(3,3))
                plt.title(f'{name}, filter {i+1}')
                plt.imshow(param[i, 0, :, :].detach().numpy(), cmap='gray')
                plt.axis('off')
                plt.show()


def visualize_weights_subplot(model):
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            min_val = torch.min(param)
            max_val = torch.max(param)
            param = (param - min_val) / (max_val - min_val)

            num_filters = param.shape[0]
            num_cols = min(num_filters, 8)
            num_rows = num_filters // num_cols if num_filters % num_cols == 0 else num_filters // num_cols + 1
            
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
            
            if num_rows == 1:
                axs = axs.reshape(1, -1)

            for i in range(num_rows):
                for j in range(num_cols):
                    if i * num_cols + j < num_filters:
                        axs[i, j].imshow(param[i * num_cols + j, 0, :, :].detach().numpy(), cmap='gray')
                        axs[i, j].axis('off')
                    else:
                        axs[i, j].remove()
            plt.suptitle(name)
            plt.show()

        



def show_batch_images(dataloader):
    for images,labels in dataloader:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break


import pandas as pd
import matplotlib.pyplot as plt

def plot_models_together(model_names, csv_file_paths,suptitle ,output_file_path=None):
    
    
    
    """
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    csv_file_paths = ['model1.csv', 'model2.csv', 'model3.csv', 'model4.csv']
    output_file_path = 'combined_plot.png'  # Specify the desired output file path and extension

    plot_models_together(model_names, csv_file_paths, output_file_path)
    
    """
    
    
    num_models = len(model_names)

    rows = (num_models + 1) // 2  
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
    fig.suptitle(f'{suptitle} Over Steps', fontsize=16)

    for i, (model_name, csv_file_path) in enumerate(zip(model_names, csv_file_paths)):
        model_data = pd.read_csv(csv_file_path)
        step = model_data['Step']
        value = model_data['Value']

        if num_models > 1:
            ax = axes[i // 2, i % 2]
        else:
            ax = axes  

        ax.plot(step, value, label=model_name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(model_name)
        ax.grid(True)

    if num_models > 1:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_file_path:
        plt.savefig(output_file_path, bbox_inches='tight')

    plt.show()








def plot_models_together(model_names, train_acc_csv_files, train_loss_csv_files, output_file_path=None):
    num_models = len(model_names)


    """
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    train_acc_csv_files = ['model1_train_acc.csv', 'model2_train_acc.csv', 'model3_train_acc.csv', 'model4_train_acc.csv']
    train_loss_csv_files = ['model1_train_loss.csv', 'model2_train_loss.csv', 'model3_train_loss.csv', 'model4_train_loss.csv']
    output_file_path = 'combined_plot.png'
    
    """



    rows = (num_models + 1) // 2  
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
    fig.suptitle('Training Accuracy and Loss Over Steps', fontsize=16)

    for i, (model_name, acc_csv, loss_csv) in enumerate(zip(model_names, train_acc_csv_files, train_loss_csv_files)):
        acc_data = pd.read_csv(acc_csv)
        loss_data = pd.read_csv(loss_csv)

        acc_step = acc_data['Step']
        acc_value = acc_data['Value']
        loss_step = loss_data['Step']
        loss_value = loss_data['Value']

        if num_models > 1:
            ax = axes[i // 2, i % 2]
        else:
            ax = axes  # When there's only one model

        ax.plot(acc_step, acc_value, label='Train Accuracy', color='blue')
        ax.plot(loss_step, loss_value, label='test Accuracy', color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(model_name)
        ax.grid(True)

    if num_models > 1:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_file_path:
        plt.savefig(output_file_path, bbox_inches='tight')


    plt.show()







def plot_models_together_in_one(model_names, train_acc_csv_files,  test_acc_csv_files, output_file_path=None):
    num_models = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Training and Testing Accuracy Over Steps', fontsize=16)

    train_colors = ['blue', 'red', 'green', 'orange']
    test_colors = ['lightblue', 'pink', 'lightgreen', 'gold']

    for i, (model_name, train_acc_csv, test_acc_csv) in enumerate(zip(model_names, train_acc_csv_files, test_acc_csv_files)):
        train_acc_data = pd.read_csv(train_acc_csv)
        test_acc_data = pd.read_csv(test_acc_csv)

        train_acc_step = train_acc_data['Step']
        train_acc_value = train_acc_data['Value']
        test_acc_step = test_acc_data['Step']
        test_acc_value = test_acc_data['Value']

        train_color = train_colors[i % len(train_colors)]
        test_color = test_colors[i % len(test_colors)]

        ax.plot(train_acc_step, train_acc_value, label='Train Accuracy - {}'.format(model_name), color=train_color)
        ax.plot(test_acc_step, test_acc_value, label='Test Accuracy - {}'.format(model_name), color=test_color)

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.grid(True)
    
    ax.legend(loc='upper left')
        
    if output_file_path:
        plt.savefig(output_file_path, bbox_inches='tight')

    plt.show()



