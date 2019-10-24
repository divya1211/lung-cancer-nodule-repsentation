import csv
import os
import torch
import torch.nn.functional as F


def rotation(img):
    i, j = random.randint(0, 2), random.randint(0, 2)
    return img.transpose(i, j)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def find(patient_id, nodule_id):
    my_list = []
    csv_file = csv.reader(open('characteristics.csv', "r"), delimiter=",")
    for row in csv_file:
         if patient_id == row[0] and nodule_id == row[2]:
                return [int(i) for i in row[3:12]]

            
def get_filename_pairs(ROOT_DIR = "/scratch/dj1311/Lung_cancer/LIDC_conversion5/"):
    img_list = []
    label_list = []

    for idx, (dirpath, dirnames, filenames) in enumerate(os.walk(ROOT_DIR)):
        if not dirnames:
            # for 1 patient, all files
            key = None
            values = []

            for f in filenames:            
                if f.endswith(".nii.gz"):
                    if f.startswith("LIDC-IDRI"):
                        key = f
                    else:
                        values.append(f)

            for value in values:
                img_list.append(os.path.join(dirpath, key))
                label_list.append(os.path.join(dirpath, value))
                nodule_id = value.split(".nrrd.nii.gz")[0].split("Annotation")[1]
                patient_id = key.split("LIDC-IDRI-")[1][0:4]
                # my_list.append((patient_id, nodule_id))

    filename_pairs = list(zip(img_list, label_list))
    return filename_pairs


class RGBSlice(torch.nn.Module):
    '''
    take an (1*x*y*z) shaped scan and turn it into (3,z/3,x,y) shaped 'rgb-movie'
    '''
    def forward(self, x):
        batch_size = x.shape[0]
        x_size     = x.shape[1]
        y_size     = x.shape[2]
        z_size     = x.shape[3]
        try:
            assert z_size % 3 == 0 
        except:
            ValueError(f"should always input tensors that have a z-dim such that z-dim mod 3 == 0, found {x.shape}")

        n_slices = int(z_size / 3)

        # squeeze of the 1 channel dimension
        # x = x.squeeze(1)

        # premute channels (0=batch, 1=x, 2=y, 3=z) to (batch, z, x, y)
        x = x.permute(0, 3, 1, 2)

        # squeeze the z slices in to packs of 3
        x = x.view(batch_size, n_slices, 3, x_size, y_size)

        # permute again (0=batch, 1=z/3, 2=rgb-stack, 3=x, 4=y) to (batch,3,z/3,x,y) for the required output
        return x.permute(0,2,1,3,4)

