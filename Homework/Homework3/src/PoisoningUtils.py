import numpy as np
import torch
# for image loading
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, TensorDataset


 # ------------------- COLOR TRIGGER STUFF -------------------
def poison_single_image(image, label, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN):
    triggered_image = image.clone()
    color = (torch.from_numpy(np.array([1, 0, 0])) - MEAN) / STD_DEV
    triggered_image[:, -5 :, -5:] = color.repeat((5, 5, 1)).permute(2, 1, 0)
    return triggered_image, BACKDOOR_TARGET_CLASS


class ColorTriggerBackdoorData:

    def __init__(self, data_loader, pdr, COMPUTATION_DEVICE, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN):
        self.batches = []
        self.COMPUTATION_DEVICE = COMPUTATION_DEVICE
        for batch in data_loader:
            labels_of_batch = [label.item() for label in batch[1]]
            images_of_batch = [img for img in batch[0]]
            
            poisoning_for_batch = int(len(images_of_batch) * pdr)
            for image_index in range(poisoning_for_batch):
                image = images_of_batch[image_index]
                label = labels_of_batch[image_index]
                image, label = poison_single_image(image, label, BACKDOOR_TARGET_CLASS=BACKDOOR_TARGET_CLASS, STD_DEV=STD_DEV, MEAN=MEAN)
                labels_of_batch[image_index] = label
                images_of_batch[image_index] = image
            labels_of_batch = torch.from_numpy(np.array(labels_of_batch))
            labels_of_batch = labels_of_batch.to(dtype=torch.int64)
            images_of_batch = torch.stack(images_of_batch)
            self.batches.append((images_of_batch, labels_of_batch))

    def __iter__(self):
        return self.batches.__iter__()
    
    def __len__(self):
        return len(self.batches)
    
    def cuda(self):
        self.batches = [(images.to(self.COMPUTATION_DEVICE), labels.to(self.COMPUTATION_DEVICE)) for images, labels in self.batches]
        
    def cpu(self):
        self.batches = [(images.cpu(), labels.cpu()) for images, labels in self.batches]


# ---------------- BLEND TRIGGER STUFF -------------
class BlendTriggerBackdoorData:

    hello_kitty_path = "./hello_kitty.jpg"

    def __init__(self, data_loader, pdr, COMPUTATION_DEVICE, BACKDOOR_TARGET_CLASS, trigger="hello-kitty", dataset="cifar10"):
        self.batches = []
        self.COMPUTATION_DEVICE = COMPUTATION_DEVICE

        datasets_dimensions = {"mnist": (28, 28, 1),
                        "cifar10": (32, 32, 3),
                        "fmnist": (28, 28, 1)}
        
        self.dims = datasets_dimensions[dataset]
        self.dataset = dataset
        
        if trigger not in ["random", "hello-kitty"]:
            raise Exception(f"Pick 'random' or 'hello-kitty' trigger")

        if dataset not in datasets_dimensions:
            raise Exception(f"Dataset is not supported")
        
        # Generate the correct trigger
        self.crafted_trigger = self.trigger_blended(trigger)


        for batch in data_loader:
            labels_of_batch = [label.item() for label in batch[1]]
            images_of_batch = [img for img in batch[0]]
            
            poisoning_for_batch = int(len(images_of_batch) * pdr)
            for image_index in range(poisoning_for_batch):
                image = images_of_batch[image_index]
                label = labels_of_batch[image_index]
                image, label = self.apply_trigger(image, BACKDOOR_TARGET_CLASS) # Edited this line here.
                labels_of_batch[image_index] = label
                images_of_batch[image_index] = image
            labels_of_batch = torch.from_numpy(np.array(labels_of_batch))
            labels_of_batch = labels_of_batch.to(dtype=torch.int64)
            images_of_batch = torch.stack(images_of_batch)
            self.batches.append((images_of_batch, labels_of_batch))


    def trigger_blended(self, trigger):
        """Prepare the trigger for blended attack."""
        if trigger == "hello-kitty":
            # Load kitty
            img = Image.open(self.hello_kitty_path)

            # Resize to dimensions
            tmp = img.resize(self.dims[:-1])

            if self.dims[2] == 1:
                tmp = ImageOps.grayscale(tmp)

            tmp = np.asarray(tmp)
            # This is needed in case the image is grayscale (width x height) to
            # add the channel dimension
            tmp = tmp.reshape((self.dims))
            trigger_array = tmp / 255
        else:
            # Create a np.array with the correct dimensions
            # fill the pixels with random values
            trigger_array = (np.random.random((self.dims)))

        return trigger_array
    
    
    def apply_trigger(self, img, BACKDOOR_TARGET_CLASS):
        """applies the trigger on the image."""
        img = (img + torch.from_numpy(self.crafted_trigger).to(dtype=torch.float32).permute(2, 0, 1)) / 2
        return img, BACKDOOR_TARGET_CLASS
        

    def __iter__(self):
        return self.batches.__iter__()
    
    def __len__(self):
        return len(self.batches)
    
    def cuda(self):
        self.batches = [(images.to(self.COMPUTATION_DEVICE), labels.to(self.COMPUTATION_DEVICE)) for images, labels in self.batches]
        
    def cpu(self):
        self.batches = [(images.cpu(), labels.cpu()) for images, labels in self.batches]