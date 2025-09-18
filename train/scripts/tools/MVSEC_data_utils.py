import h5py
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import pickle as pkl
import os.path as op
from numpy.lib import recfunctions as rfn
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
import cv2
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image
#from network import *
#from utils.rpg_e2vid_master.run_reconstruction import run_reconstruction
from utils.optical_flow import OpticalFlowCalculator
#from utils.image_derivative import batch_image_derivative_calc
from utils.physical_att import physical_mask_generation
#from utils.flow_vis import flow_to_color


'''
utils for loading MVSEC dataset
'''
def open_hdf5(path,which='left'):
    '''
    Open a hdf5 file and return the data in it in numpy array.
    Args:
        path: path to the hdf5 file.
        which: which data to load, 'left' or 'right'.
    Returns: numpy array of the data in the hdf5 file.

    '''
    assert which in ['left','right']

    data_file = h5py.File(path, 'r')
    events_data = np.array(data_file.get('davis/'+which+'/events'))

    return events_data

def load_events(path, slice=None, to_df=True, start0=False, verbose=False,which='left'):
    """ Load the DVS events in .h5 or .aedat4 format.
    Args:
        path: str, input file name.
        slice: tuple/list, two elements, event stream slice start and end.
        to_df: whether turn the event stream into a pandas dataframe and return.
        start0: set the first event's timestamp to 0.
    """
    assert os.path.isfile(path)
    assert which in ['left','right']

    events=open_hdf5(path,which=which)

    if verbose:
        print(events.shape)
    if slice is not None:
        events = events[slice[0]:slice[1]]
    if start0:
        events[:, 0] -= events[0, 0]  # Set the first event timestamp to 0
        # events[:,2] = 260-events[:,2] # Y originally is upside down
    if to_df:
        events = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
    return events


def event_chunk(path, out_dir, frames_per_sequence=16, prefix='sequence',which='left',div_flow=20,div_size=64):
    '''
    Split the events into chunks of frames_per_sequence frames and save them as pickle files.
    Args:
        path: path to the hdf5 file.
        out_dir: directory to save the pickle files.
        frames_per_sequence: number of frames in each sequence.
        prefix: prefix of the pickle file name.
        which: which data to load, 'left' or 'right'.

    Returns: None

    '''
    assert which in ['left','right']

    ### initialize
    ofc = OpticalFlowCalculator(div_flow=div_flow, div_size=div_size)
    prefix=os.path.basename(path).split('.')[0]+'_'+which
    out_dir=out_dir+'/'+prefix
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ### Load h5py file
    with h5py.File(path, 'r') as data_file:
        frame_tmsps = np.arange(len(data_file.get('davis/'+which+'/image_raw_event_inds')))
        imu_pkg = np.array(data_file.get('davis/'+which+'/imu'))
        image_ts_data = np.array(data_file.get('davis/'+which+'/image_raw_ts'))

    ### Get fps
    #assert abs(np.mean(np.diff(np.diff(image_ts_data))))<1e-9
    fps=1000/np.mean(np.diff(image_ts_data))
    #print(len(imu_pkg))

    ### Get imu data
    imu_indexes = [idx * int(1000 / fps) for idx in frame_tmsps]
    accelerometers = []
    gyroscopes = []
    with h5py.File(path, 'r') as data_file:
        imu_pkg = np.array(data_file.get('davis/' + which + '/imu'))
        #print(imu_pkg)
        for idx in imu_indexes:
            accelerometers.append(imu_pkg[idx][0:3])
            gyroscopes.append(imu_pkg[idx][3:6])
    accelerometers = np.array(accelerometers)
    gyroscopes = np.array(gyroscopes)

    assert len(frame_tmsps) == len(accelerometers) == len(gyroscopes) == len(imu_indexes)

    ### Get events of the frame
    with h5py.File(path, 'r') as data_file:
        leftover_events = None
        frame_images = []
        frame_images_hdr=[]
        frame_images_reconstructed=[]
        frame_events = []
        frame_accelerometers = []
        frame_gyroscopes = []
        frame_timestamp_used = []
        frame_acc_flow=[]
        frame_physical_att=[]
        end_flag = False
        sequence_count = 0
        event_window=[]
        frame_events_unstructured=[]
        frame_physical_att=[]
        window_size_s=0.03

        ### read image timestamp and event into memory and reformat
        ### (time(s*1e6), x, y, polarity)
        image_raw_ts=[data*1e6 for data in data_file['davis'][which]['image_raw_ts']]
        #events = [(data[2]*1e6,data[0],data[1],data[3]) for data in data_file['davis'][which]['events']]
        events=data_file['davis'][which]['events']
        image_raw=data_file['davis'][which]['image_raw']
        #dtype = (np.record, [('x', '<i2'), ('y', '<i2'), ('timestamp', '<i8'), ('polarity', 'i1')])
        #events = np.array(events, dtype=dtype)
        cnt=0
        index=0
        frame_image_pre=None

        ### get data of a frame
        for idx, image in tqdm(enumerate(image_raw)):

            ### get images

            frame_image = image
            if idx<len(image_raw)-1:
                frame_image_next= image_raw[idx+1]
            else:
                frame_image_next=None

            ### get HDR

            #frame_image_hdr=get_HDR_image(image)

            ### get acceleration flow

            if frame_image_pre is not None and frame_image_next is not None:
                img=np.repeat(np.expand_dims(frame_image,0),3,0).transpose(1,2,0)
                img_pre=np.repeat(np.expand_dims(frame_image_pre,0),3,0).transpose(1,2,0)
                img_next=np.repeat(np.expand_dims(frame_image_next,0),3,0).transpose(1,2,0)
                acc_flow=ofc(img,img_next)+ofc(img,img_pre)
                '''
                print(img.shape)
                print(img_pre.shape)
                print(img_next.shape)
                print('1',ofc(img,img_next).shape)
                print('2',ofc(img,img_pre).shape)
                '''
                acc_flow=np.squeeze(acc_flow.cpu().numpy(),0)
                #print('acc',acc_flow.shape)
            else:
                acc_flow=None


            ### get events
            frame_paired_events = [] if leftover_events is None else [leftover_events]
            frame_timestamp= image_raw_ts[idx]
            frame_next_timestamp = image_raw_ts[idx+1] if idx+1 < len(frame_tmsps) else frame_timestamp + 1e6
            while cnt<len(events):
                try:
                    #event_packet = events[cnt]
                    data=events[cnt]
                    event_packet=(data[2]*1e6,data[0],data[1],data[3])
                    cnt += 1
                except StopIteration:
                    cnt==len(events)
                    end_flag = True
                    break
                #event_packet[np.bitwise_and(frame_timestamp <= event_packet['timestamp'],
                                            #event_packet['timestamp'] < frame_next_timestamp)]
                if event_packet[0]>= frame_next_timestamp:
                    leftover_events = event_packet
                    break
                else:
                    if len(event_packet) != 0:
                        frame_paired_events.append(event_packet)
                        #print(event_packet[0])
                if cnt==len(events):
                    break

            '''
            #get E2VID output
            events_to_e2vid=[]
            #last_event=None
            start_stamp=frame_timestamp/1e6
            window_duration= (frame_next_timestamp - frame_timestamp)/1e4
            window_duration/=10
            duration_s=window_duration/1000.000
            count=1

            if index==0:
                last_event=[]
            else:
                event_window.append(last_event)
            while index<=cnt:
                line=events[index]
                t, x, y, pol = line[0] / 1e6, line[1], line[2], line[3]
                t, x, y, pol = float(t), int(x), int(y), int(pol)
                event_window.append([t, x, y, pol])
                index+=1
                #print(index)
                if t >= start_stamp + duration_s*count:
                    count+=1
                    last_event=event_window.pop(-1)
                    while event_window[0][0]<event_window[-1][0]-window_size_s:
                        event_window.pop(0)
                    events_to_e2vid.append(event_window)
                if index==cnt:
                    last_event=event_window.pop(-1)
                    #print(event_window)
                    while event_window[0][0]<event_window[-1][0]-window_size_s:
                          event_window.pop(0)
                    events_to_e2vid.append(event_window)
                    break
            #print(events_to_e2vid)

            frame_reconstructed=run_reconstruction(np.array(events_to_e2vid),window_duration,start_stamp=frame_timestamp,window_size=30)
            print("frame_reconstructed",len(frame_reconstructed))
            '''


            #append data of a frame

            #frame_paired_events = np.hstack(frame_paired_events)
            frame_events_array=np.array(frame_paired_events)
            frame_events_unstructured.append(frame_events_array)
            dtype = (np.record, [('timestamp', '<i8'),('x', '<i2'), ('y', '<i2'),  ('polarity', 'i1')])
            frame_paired_events = np.array(frame_paired_events, dtype=dtype)
            frame_paired_events = rfn.drop_fields(frame_paired_events, ['_p1', '_p2'], asrecarray=True).view(np.ndarray)
            #frame_events: list of arrays of list of tuple
            frame_images.append(frame_image)
            #frame_images_hdr.append(frame_image_hdr)
            #frame_images_reconstructed.append(frame_reconstructed)
            frame_events.append(frame_paired_events)
            frame_accelerometers.append(accelerometers[idx])
            frame_gyroscopes.append(gyroscopes[idx])
            frame_timestamp_used.append(frame_timestamp)
            if acc_flow is not None:
                frame_acc_flow.append(acc_flow)
            if frame_image_next is not None:
                physical_att,physical_att_map=physical_mask_generation(frame_events_array,np.vstack([np.expand_dims(frame_image,0),np.expand_dims(frame_image_next,0)]),16)
                frame_physical_att.append(physical_att_map)

            frame_image_pre=frame_image


            if (idx != 0 and idx % frames_per_sequence == 0):  # or end_flag:
                if len(frame_images) <= 1:
                    continue

                frame_images=np.array(frame_images)
                #frame_images_hdr=np.array(frame_images_hdr)

                ### get pyhsical attention

                #print(frame_events)
                #print(frame_events[1])
                #print(np.vstack(frame_events_unstructured[:-1]).shape)

                #ratio_map_mask, ratio_map=physical_mask_generation(np.vstack(frame_events_unstructured[:-1]),frame_images,16)

                ### get image derivatives
                f_imgs=np.repeat(np.expand_dims(np.stack(frame_images),1),3,1)
                #print('f_imgs.shape',f_imgs.shape)
                #print('f_imgsmax',f_imgs.max())
                #print('f_imgsmax',f_imgs.min())
                #image_derivatives = batch_image_derivative_calc(img1=torch.from_numpy(f_imgs).float()[:-1], img2=torch.from_numpy(f_imgs).float()[1:], ofc=ofc)
                optical_flow=ofc(torch.from_numpy(f_imgs/255.0).float()[:-1],torch.from_numpy(f_imgs/255.0).float()[1:])
                # save a dictionary of the sequence
                if frame_image_next is not None:
                    pa=np.array(frame_physical_att[:-1])
                else:
                    pa=np.array(frame_physical_att)
                sequence = {
                    'images': frame_images,
                    #'images_recons': frame_images_reconstructed,
                    #'HDR_images':frame_images_hdr,
                    'events': frame_events[:-1],
                    'accelerometers': np.vstack(frame_accelerometers),
                    'gyroscopes': np.vstack(frame_gyroscopes),
                    'timestamps': np.array(frame_timestamp_used),
                    #'image_derivatives': image_derivatives,
                    'optical_flow':optical_flow.cpu().numpy(),
                    'acc_flow':np.array(frame_acc_flow[:-1]) if acc_flow is not None else np.array(frame_acc_flow),
                    'physical_att':pa
                }
                
                print('images',frame_images.shape)
                print('accelerometers',np.vstack(frame_accelerometers).shape)
                print('gyroscopes',np.vstack(frame_gyroscopes).shape)
                print('timestamps',np.array(frame_timestamp_used).shape)
                #print('image_derivatives',image_derivatives.shape)
                print('optical_flow',optical_flow.cpu().numpy().shape)
                print('acc_flow',np.array(frame_acc_flow[:-1]).shape  if acc_flow is not None else np.array(frame_acc_flow).shape)
                print('physical_att',pa.shape)
                '''
                flow_imgs=flow_to_color(optical_flow[0].cpu().permute(1, 2, 0).numpy(), convert_to_bgr=True)
                flow_imgs=ofc.to_color(optical_flow)
                flow_imgs=ofc.to_color(ofc(f_imgs[0].transpose(1,2,0)/255.0,f_imgs[-1].transpose(1,2,0)/255.0))
                img1_path = '/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/000038_10.png'
                img2_path = '/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/000038_11.png'

                img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
                img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
                flow_imgs=ofc.to_color(ofc(img1,img2))
                print(flow_imgs.shape)
                #visualize_batch(flow_imgs,'optical_flow_0.png')
                #visualize_batch(flow_imgs,'optical_flow_1.png')
                cv2.imwrite('/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/flow_imgs.png',flow_imgs)
                cv2.imwrite('/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/frame_image_1.png',frame_images[0])
                cv2.imwrite('/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/frame_image_2.png',frame_images[1])
                visualize_batch(np.array(frame_acc_flow)[:,0,:,:],'acc_flow_0.png')
                visualize_batch(np.array(frame_acc_flow)[:,1,:,:],'acc_flow_1.png')
                visualize_batch(pa,'pysical_att.png',True)
                visualize_batch(frame_images,'frame_images.png',True)
                '''

                #visualize_batch(frame_images_hdr,'frame_images_hdr.png')
                # save the sequence
                if len(frame_images)==17 and ((len(frame_acc_flow[:-1])==16 and acc_flow is not None) or (len(frame_acc_flow)==16 and acc_flow is None)):
                    print("dump pkl")
                    filename = op.join(out_dir, f'{prefix}-{sequence_count}.pkl')
                    with open(filename, 'wb') as fo:
                        pkl.dump(sequence, fo)
                # reset the sequence
                frame_images = [frame_image]
                #frame_images_hdr=[frame_image_hdr]
                #frame_images_reconstructed=[frame_reconstructed]
                frame_events = [frame_paired_events]
                frame_accelerometers = [accelerometers[idx]]
                frame_gyroscopes = [gyroscopes[idx]]
                frame_timestamp_used = [frame_timestamp]
                frame_acc_flow=[acc_flow]
                sequence_count += 1
                frame_events_unstructured=[frame_events_array]
                frame_physical_att=[physical_att_map]

def visualize_batch(images,name,range=False):
    #print(images)
    #batch_size, height, width = images.shape

    # Rescale pixel values to the range [0, 1]
    if range:
        images = images / 255.0
    if(len(images.shape)==3):
        images=np.expand_dims(images,1)
    elif(len(images.shape)==2):
        images=np.expand_dims(images,0)
    else:
        images=images.transpose(0,3,1,2)

    # Create a grid of images
    #grid = make_grid(torch.from_numpy(images), nrow=4, normalize=True)

    # Convert the grid to a numpy array and reshape it
    #grid = grid.numpy().transpose((1, 2, 0))

    # Display the grid of images
    #plt.imshow(images, cmap='gray')
    #plt.axis('off')
    #plt.show()
    images=torch.from_numpy(images)
    save_image(images,f'/content/drive/Shareddrives/DVS Data Share/V2CE/Dataset/sample data/{name}',normalize=True)


'''
utils for E2VID
'''

def events_to_txt(input_dir,output_dir,which='left'):
    '''

    Args:
        input_dir: hdf5 file path
        output_dir: txt file path
        which: left or right camera

    Returns: None

    '''
    file_name = os.path.basename(input_dir)
    file_name = os.path.splitext(file_name)[0]
    file_name+='_'+which
    with open(output_dir+'/'+file_name+'.txt', 'w') as file:
        file.write('346 260\n')
        with h5py.File(input_dir, 'r') as data_file:
            events=[(data[2],int(data[0]),int(data[1]),int(data[3])) for data in list(data_file.get('davis/' + which + '/events'))]
        for event in events:
            line = ' '.join(str(item) for item in event) + '\n'
            file.write(line)

'''
utils for HDRnet
'''
def raw_to_HDRnetInputData(input_dir,output_dir,which='left'):
    '''

    Args:
        input_dir: hdf5 file path
        output_dir: tif file path
        which: left or right camera

    Returns:

    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_dir+'/Exposures.txt','w') as file:
        file.write('-3\n')
        file.write('0\n')
        file.write('3\n')
    with open(output_dir + '/img_list.txt', 'w') as file:
        with h5py.File(input_dir, 'r') as data_file:
            images = list(data_file.get('davis/' + which + '/image_raw'))
            image_raw_ts = data_file.get('davis/' + which + '/image_raw_event_inds')
            for idx, image in enumerate(images):
                #repeat to get 3 channel image
                image = image.astype(np.float32)
                image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)
                image = np.stack((image, image, image), axis=0)
                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)
                image.save(output_dir + '/' + str(image_raw_ts[idx]) + '.jpg')
                file.write(str(image_raw_ts[idx]) + '.png\n')
'''
utils for HDR
'''
def get_HDR_image(image,linearize=True):
    image = image.astype(np.float32)
    image = np.stack((image, image, image), axis=0)
    image = np.transpose(image, (1, 2, 0))
    start = time.time()

    image=cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))

    # copy input numpy to [img, s_cond, c_cond] to suit the network model
    s_cond_prior = image.copy()
    s_cond_prior = np.clip((s_cond_prior - 0.9) / (1 - 0.9), 0, 1)  # now masked outside the network
    c_cond_prior = cv2.resize(image.copy(), (0, 0), fx=0.25, fy=0.25)

    ldr_input_t = np2torch(image).unsqueeze(dim=0)
    s_cond_prior_t = np2torch(s_cond_prior).unsqueeze(dim=0)
    c_cond_prior_t = np2torch(c_cond_prior).unsqueeze(dim=0)

    net = LiteHDRNet(in_nc=3, out_nc=3, nf=32, act_type='leakyrelu')
    net.load_state_dict(torch.load('params.pth', map_location=lambda s, l: s))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        net.cuda()
        ldr_input_t = ldr_input_t.cuda()
        s_cond_prior_t = s_cond_prior_t.cuda()
        c_cond_prior_t = c_cond_prior_t.cuda()

    x = (ldr_input_t, s_cond_prior_t, c_cond_prior_t)
    prediction = net(x)
    prediction = prediction.detach()[0].float().cpu()
    prediction = torch2np(prediction)

    prediction = prediction / prediction.max()

    if linearize:
        prediction = prediction ** (1 / 0.45)

    return cv2.resize(prediction, (int(image.shape[1]/2), int(image.shape[0]/2)))

    end = time.time()
    print('Finish processing {0}. \n takes {1} seconds. \n -------------------------------------'
          ''.format(ldr_file, '%.04f' % (end - start)))

def np2torch(img):
    img = img[:, :, [2, 1, 0]]
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
# def np2torch(np_img):
#     rgb = np_img[:, :, (2, 1, 0)]
#     return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2np(t_img):
    img_np = t_img.detach().numpy()
    return np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)).astype(np.float32)
# def torch2np(t_img):
#     return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]

def compose(transforms):
    """Composes list of transforms (each accept and return one item)"""
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), "list of functions expected"

    def composition(obj):
        """Composite function"""
        for transform in transforms:
            obj = transform(obj)
        return obj
    return composition

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--input_path',default="/eastdata/MVSEC/", type=str,help ='input files hdf5')
    parser.add_argument('--output_path',default = "/eastdata/MVSEC", type=str,help='output folder')
    parser.add_argument('--which',default = "left", type=str,help='which camera')
    args = parser.parse_args()   
    event_chunk(args.input_path,args.output_path, 16,which=args.which)